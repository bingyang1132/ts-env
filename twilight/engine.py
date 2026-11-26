"""The game driver.

The whole game is written as one generator. It yields a :class:`~twilight.decisions.Decision`
whenever it needs input and receives the chosen :class:`~twilight.decisions.Action` back,
which lets deeply nested card effects ask sub-questions without any explicit state
machine -- an event that says "remove 3 influence from Western Europe, no more than 2
per country" is a loop that yields a country choice three times.

The trade-off is that a running game holds live generator frames, which cannot be
deep-copied. A clone is therefore defined as a replay of the action history against the
same seed (:meth:`Game.replay_clone`): exact, since the engine is deterministic given
seed + actions, but O(history).

:meth:`Game.clone` produces the same object faster. At the two points where the driver
holds no state outside :class:`GameState` -- the start of a turn and the start of each
action round -- a game being *replayed* by a clone records a checkpoint: a copy of the
state, the history prefix, and where the driver stands. A later clone copies the latest
checkpoint whose history is a prefix of the game's, restarts the driver at that point
(``_run(resume=...)``) and replays only the actions after it. Checkpoints are made only
by replay, never by a game's own :meth:`Game.step`, so they are replay results by
construction: a state edited from outside (a determinized clone, a test fixture) is
ignored by ``clone()`` exactly as it is by a replay.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence

from . import rules, spacerace
from .data import (
    CARDS,
    CHINA_CARD,
    COUNTRIES,
    COUNTRY_ORDER,
    REGION_COUNTRIES,
    deck_for_stage,
)
from .decisions import (
    Action,
    Decision,
    DecisionType,
    NO,
    PASS,
    YES,
    cards_decision,
    confirm_decision,
    countries_decision,
    country_action,
    number_action,
    options_decision,
    use_action,
)
from .enums import OpsUse, Phase, Region, Side, Stage, WinReason
from .state import GameState, VP_LIMIT

#: Action rounds per turn. The Mid War adds a seventh.
ACTION_ROUNDS = {Stage.EARLY_WAR: 6, Stage.MID_WAR: 7, Stage.LATE_WAR: 7}

#: Cards each player holds after the deal, by stage.
HAND_SIZE = {Stage.EARLY_WAR: 8, Stage.MID_WAR: 9, Stage.LATE_WAR: 9}

#: Turn on which each stage's cards enter the deck.
STAGE_START_TURN = {Stage.EARLY_WAR: 1, Stage.MID_WAR: 4, Stage.LATE_WAR: 8}

FINAL_TURN = 10

#: Fixed opening influence, from the rulebook's setup diagram.
SETUP_INFLUENCE: dict[Side, dict[str, int]] = {
    Side.USSR: {
        "Syria": 1,
        "Iraq": 1,
        "North Korea": 3,
        "East Germany": 3,
        "Finland": 1,
    },
    Side.USA: {
        "Canada": 2,
        "Iran": 1,
        "Israel": 1,
        "Japan": 1,
        "Australia": 4,
        "Philippines": 1,
        "South Korea": 1,
        "Panama": 1,
        "South Africa": 1,
        "UK": 5,
    },
}

#: Free influence each player distributes after the fixed setup, and where.
SETUP_FREE: dict[Side, tuple[int, Region]] = {
    Side.USSR: (6, Region.EASTERN_EUROPE),
    Side.USA: (7, Region.WESTERN_EUROPE),
}

#: Optional bidding handicap (rule 11.1.4): the loser of the bid adds this many
#: influence points to countries where their side already has influence. Off by
#: default, since the base game has no handicap.
DEFAULT_HANDICAP = 0


@dataclass(slots=True)
class EventContext:
    """What an event handler is told about the card being resolved."""

    card: str
    #: The player whose event this is and who makes its choices.
    player: Side
    #: Operations value of the card, for events that reuse it.
    ops: int
    #: True when the event fired because the opponent played the card for operations.
    triggered_by_opponent: bool = False


EventHandler = Callable[["Game", EventContext], Iterator[Decision]]

#: Cards whose event can still end in a US UN Intervention event in the same action
#: round (UN Intervention itself; Missile Envy, Star Wars and Grain Sales can fetch it,
#: FAQ #50), so they leave We Will Bury You's victory points undecided.
WE_WILL_BURY_YOU_CHAIN = frozenset(
    {"UN Intervention", "Missile Envy", "Star Wars", "Grain Sales to Soviets"}
)


#: Order of the two action rounds within a round.
_ROUND_ORDER = (Side.USSR, Side.USA)


@dataclass(frozen=True, slots=True, eq=False)
class _Checkpoint:
    """A replay result at a resumable point of the driver (see :meth:`Game.clone`).

    ``state`` is private to the checkpoint and never mutated; a resumed game gets a copy.
    """

    history: tuple[str, ...]
    resume: tuple
    state: GameState
    unimplemented: tuple[str, ...]
    config: tuple


class Game:
    """A single game of Twilight Struggle.

    Drive it by reading :attr:`decision` and calling :meth:`step`::

        game = Game(seed=0)
        while game.decision is not None:
            game.step(pick(game.decision))
    """

    def __init__(
        self,
        seed: int | None = None,
        *,
        optional_cards: bool = False,
        handicap: int = DEFAULT_HANDICAP,
        strict_events: bool = False,
    ) -> None:
        self.optional_cards = optional_cards
        self.handicap = handicap
        #: When true, an event with no registered handler raises instead of being
        #: skipped. Tests turn this on so gaps cannot pass silently.
        self.strict_events = strict_events

        self.state = GameState(rng=random.Random(seed), optional_cards=optional_cards)
        self.seed = seed
        self.history: list[str] = []
        #: Cards whose event was skipped for lack of an implementation.
        self.unimplemented: list[str] = []
        #: Replay checkpoints known for this game's history (see :meth:`clone`).
        self._checkpoints: tuple[_Checkpoint, ...] = ()
        #: Set only while :meth:`clone` replays: the driver then records checkpoints.
        self._recording: list[_Checkpoint] | None = None

        self._driver = self._run()
        self.decision: Decision | None = self._advance(None)

    # ------------------------------------------------------------------ #
    # Driving
    # ------------------------------------------------------------------ #

    def _advance(self, action: Action | None) -> Decision | None:
        """Run the driver to its next decision, or to the end of the game.

        The game can end deep inside a card -- a VP award, a DEFCON drop or a Cuban
        Missile Crisis coup halfway through an event or an operations play -- and the
        generator stack would otherwise go on asking for the rest of that card. Once the
        state is over, no further decision is issued: the driver is closed and ``None``
        returned.
        """
        try:
            decision = self._driver.send(action)
        except StopIteration:
            return None
        if self.state.is_over:
            self._driver.close()
            return None
        return decision

    def step(self, choice: Action | str | int) -> Decision | None:
        """Apply one atomic action and return the next decision, or ``None`` if over."""
        if self.decision is None:
            raise RuntimeError("the game is over; no further actions are possible")
        action = self.decision.resolve(choice)
        self.history.append(action.key)
        self.decision = self._advance(action)
        return self.decision

    @property
    def is_over(self) -> bool:
        return self.decision is None or self.state.is_over

    def replay_clone(self) -> "Game":
        """An independent copy, produced by replaying this game's whole action history
        from the seed. The reference definition of a clone; O(history)."""
        twin = Game(
            self.seed,
            optional_cards=self.optional_cards,
            handicap=self.handicap,
            strict_events=self.strict_events,
        )
        for key in self.history:
            if twin.decision is None:
                break
            twin.step(key)
        return twin

    def clone(self) -> "Game":
        """An independent copy, equal to :meth:`replay_clone` in every respect.

        Resumes from the latest replay checkpoint on this game's history and replays
        only the actions after it (see the module docstring). Like a replay, it ignores
        any edit made to ``self.state`` from outside :meth:`step`.
        """
        base = self._usable_checkpoint()
        recorded: list[_Checkpoint] = []
        if base is None:
            twin = Game.__new__(Game)
            twin.optional_cards = self.optional_cards
            twin.handicap = self.handicap
            twin.strict_events = self.strict_events
            twin.state = GameState(
                rng=random.Random(self.seed), optional_cards=self.optional_cards
            )
            twin.seed = self.seed
            twin.history = []
            twin.unimplemented = []
            twin._checkpoints = ()
            twin._recording = recorded
            twin._driver = twin._run()
            twin.decision = twin._advance(None)
            start, kept = 0, ()
        else:
            twin = Game.__new__(Game)
            twin.optional_cards = self.optional_cards
            twin.handicap = self.handicap
            twin.strict_events = self.strict_events
            twin.state = base.state.copy()
            twin.seed = self.seed
            twin.history = list(base.history)
            twin.unimplemented = list(base.unimplemented)
            twin._checkpoints = ()
            twin._recording = recorded
            twin._driver = twin._run(resume=base.resume)
            twin.decision = twin._advance(None)
            start = len(base.history)
            kept = self._checkpoints[: self._checkpoints.index(base) + 1]
        history = self.history
        for i in range(start, len(history)):
            if twin.decision is None:
                break
            twin.step(history[i])
        twin._recording = None
        if base is not None:
            # The checkpoint resumed from is re-recorded at the same position.
            recorded = [c for c in recorded if len(c.history) > start]
        lineage = kept + tuple(recorded)
        self._checkpoints = lineage
        twin._checkpoints = lineage
        return twin

    def _usable_checkpoint(self) -> "_Checkpoint | None":
        """The latest checkpoint whose history is a prefix of this game's history."""
        n = len(self.history)
        for cp in reversed(self._checkpoints):
            m = len(cp.history)
            if (
                m <= n
                and cp.config == (self.seed, self.optional_cards, self.handicap, self.strict_events)
                and tuple(self.history[:m]) == cp.history
            ):
                return cp
        return None

    def _checkpoint(self, resume: tuple) -> None:
        """Driver hook at a resumable point: record a checkpoint while replaying."""
        if self._recording is not None:
            self._recording.append(
                _Checkpoint(
                    history=tuple(self.history),
                    resume=resume,
                    state=self.state.copy(),
                    unimplemented=tuple(self.unimplemented),
                    config=(self.seed, self.optional_cards, self.handicap, self.strict_events),
                )
            )

    # ------------------------------------------------------------------ #
    # The event API used by card implementations
    # ------------------------------------------------------------------ #

    def ask(self, decision: Decision) -> Iterator[Decision]:
        """Put *decision* to its player and return the chosen action."""
        action = yield decision
        assert action is not None, "driver resumed without an action"
        return action

    def choose_country(
        self,
        side: Side,
        names: Sequence[str],
        prompt: str,
        *,
        dtype: DecisionType = DecisionType.CHOOSE_COUNTRY,
        allow_pass: bool = False,
        labels: dict[str, str] | None = None,
        **context: Any,
    ) -> Iterator[Decision]:
        """Ask *side* to pick one of *names*; returns the name, or ``None`` if passed.

        Returns ``None`` immediately when *names* is empty, so callers do not have to
        guard every event that might have no legal target.
        """
        names = [n for n in dict.fromkeys(names)]
        if not names:
            return None
        if len(names) == 1 and not allow_pass:
            return names[0]
        action = yield from self.ask(
            countries_decision(
                dtype, side, prompt, names, labels=labels, allow_pass=allow_pass, **context
            )
        )
        return None if action == PASS else action.value

    def choose_card(
        self,
        side: Side,
        names: Sequence[str],
        prompt: str,
        *,
        dtype: DecisionType = DecisionType.CHOOSE_CARD,
        allow_pass: bool = False,
        labels: dict[str, str] | None = None,
        **context: Any,
    ) -> Iterator[Decision]:
        names = [n for n in dict.fromkeys(names)]
        if not names:
            return None
        action = yield from self.ask(
            cards_decision(
                dtype, side, prompt, names, labels=labels, allow_pass=allow_pass, **context
            )
        )
        return None if action == PASS else action.value

    def choose_option(
        self, side: Side, prompt: str, labels: Sequence[str], **context: Any
    ) -> Iterator[Decision]:
        """Ask *side* to pick from a short list; returns the chosen index."""
        if len(labels) == 1:
            return 0
        action = yield from self.ask(options_decision(side, prompt, labels, **context))
        return action.value

    def confirm(self, side: Side, prompt: str, **context: Any) -> Iterator[Decision]:
        action = yield from self.ask(confirm_decision(side, prompt, **context))
        return action == YES

    def roll(self) -> int:
        return self.state.rng.randint(1, 6)

    def degrade_defcon(self, steps: int, actor: Side | None = None) -> None:
        """Lower DEFCON by *steps*, ending the game if it reaches 1.

        The loss falls on the **phasing player** -- whoever is taking this action round,
        or whoever played the headline -- and not on the side that happens to own the
        event doing the degrading. Playing an opponent's card for operations triggers
        their event, but if that drops DEFCON to 1 it is still the player who played the
        card who loses.

        *actor* is only a fallback for the rare call made outside anybody's action round.
        """
        if steps <= 0:
            return
        blame = self.state.phasing_player if self.state.phasing_player is not None else actor
        self.state.change_defcon(-steps)
        self.state.note(f"DEFCON degraded to {self.state.defcon}", blame)
        if blame is not None:
            self._check_defcon_loss(blame)

    def improve_defcon(self, steps: int) -> None:
        if steps <= 0:
            return
        self.state.change_defcon(steps)
        self.state.note(f"DEFCON improved to {self.state.defcon}")

    # -- influence -------------------------------------------------------- #

    def add_influence(
        self,
        side: Side,
        points: int,
        *,
        allowed: Iterable[str],
        max_per_country: int | None = None,
        prompt: str = "",
        chooser: Side | None = None,
    ) -> Iterator[Decision]:
        """Place *points* influence for *side*, one point at a time, no ops cost.

        This is event placement: each point costs one, and the only restriction is
        *allowed* (plus *max_per_country*). Operations placement goes through
        :meth:`spend_operations_influence`, which also charges double into
        opponent-controlled countries.
        """
        chooser = side if chooser is None else chooser
        allowed = list(dict.fromkeys(allowed))
        placed: dict[str, int] = {}

        for remaining in range(points, 0, -1):
            candidates = [
                n
                for n in allowed
                if max_per_country is None or placed.get(n, 0) < max_per_country
            ]
            if not candidates:
                break
            name = yield from self.choose_country(
                chooser,
                candidates,
                prompt or f"Place {remaining} more {side.label} influence",
                dtype=DecisionType.PLACE_INFLUENCE,
                labels=self._influence_labels(candidates),
                remaining=remaining,
                for_side=side.label,
            )
            if name is None:
                break
            self.state.add_inf(side, name, 1)
            placed[name] = placed.get(name, 0) + 1

        if placed:
            summary = ", ".join(f"{n} +{v}" for n, v in placed.items())
            self.state.note(f"{side.label} influence placed: {summary}", chooser)
        return placed

    def remove_influence(
        self,
        target: Side,
        points: int,
        *,
        allowed: Iterable[str] | None = None,
        max_per_country: int | None = None,
        chooser: Side,
        prompt: str = "",
        allow_pass: bool = False,
    ) -> Iterator[Decision]:
        """Remove up to *points* of *target*'s influence, chosen by *chooser*.

        *allow_pass* lets the chooser stop early, for events worded "up to".
        """
        pool = list(allowed) if allowed is not None else list(COUNTRY_ORDER)
        removed: dict[str, int] = {}

        for remaining in range(points, 0, -1):
            candidates = [
                n
                for n in pool
                if self.state.inf(target, n) > 0
                and (max_per_country is None or removed.get(n, 0) < max_per_country)
            ]
            if not candidates:
                break
            name = yield from self.choose_country(
                chooser,
                candidates,
                prompt or f"Remove {remaining} more {target.label} influence",
                dtype=DecisionType.REMOVE_INFLUENCE,
                labels=self._influence_labels(candidates),
                allow_pass=allow_pass,
                remaining=remaining,
                for_side=target.label,
            )
            if name is None:
                break
            self.state.remove_inf(target, name, 1)
            removed[name] = removed.get(name, 0) + 1

        if removed:
            summary = ", ".join(f"{n} -{v}" for n, v in removed.items())
            self.state.note(f"{target.label} influence removed: {summary}", chooser)
        return removed

    def _influence_labels(self, names: Sequence[str]) -> dict[str, str]:
        """Annotate country options with the numbers an agent would otherwise recompute."""
        out = {}
        for name in names:
            c = self.state
            ussr, usa = c.inf(Side.USSR, name), c.inf(Side.USA, name)
            owner = rules.controller(c, name)
            tag = f"USSR {ussr} / US {usa}, stab {country_stability(name)}"
            if owner is not None:
                tag += f", {owner.label} controls"
            out[name] = tag
        return out

    # ------------------------------------------------------------------ #
    # Game flow
    # ------------------------------------------------------------------ #

    def _run(self, resume: tuple | None = None) -> Iterator[Decision]:
        """The whole game. *resume* restarts it at a checkpoint (see :meth:`clone`):
        ``("turn",)`` at the start of a turn, ``("ar", round, side_index)`` at the start
        of an action round, with the state as it was there."""
        if resume is None:
            yield from self._setup()
        mid_turn = resume is not None and resume[0] == "ar"

        while mid_turn or not self.state.is_over:
            if mid_turn:
                mid_turn = False
                yield from self._play_turn(resume=resume[1:])
            else:
                self._checkpoint(("turn",))
                yield from self._play_turn()
            if self.state.is_over or self.state.turn >= FINAL_TURN:
                break
            self.state.turn += 1

        if not self.state.is_over:
            self._final_scoring()

    # -- setup ------------------------------------------------------------ #

    def _setup(self) -> Iterator[Decision]:
        state = self.state
        state.phase = Phase.SETUP

        # Cards are dealt before influence is placed, so both players choose their
        # opening with their hand in view (rule 3.1).
        self._add_stage_cards()
        self._deal_cards(Stage.EARLY_WAR)

        state.china_card_owner = Side.USSR
        state.china_card_face_up = True
        state.note("USSR receives the China Card face up")

        for side, placements in SETUP_INFLUENCE.items():
            for name, amount in placements.items():
                state.set_inf(side, name, amount)
        state.note("fixed setup influence placed")

        for side in (Side.USSR, Side.USA):
            points, region = SETUP_FREE[side]
            yield from self.add_influence(
                side,
                points,
                allowed=REGION_COUNTRIES[region],
                prompt=f"{side.label} setup: place influence in {region}",
            )

        if self.handicap:
            yield from self.add_influence(
                Side.USA,
                self.handicap,
                allowed=[n for n in COUNTRY_ORDER if state.inf(Side.USA, n) > 0],
                prompt=f"US handicap: place {self.handicap} influence where the US already has some",
            )

    # -- turns ------------------------------------------------------------ #

    def _current_stage(self) -> Stage:
        if self.state.turn >= STAGE_START_TURN[Stage.LATE_WAR]:
            return Stage.LATE_WAR
        if self.state.turn >= STAGE_START_TURN[Stage.MID_WAR]:
            return Stage.MID_WAR
        return Stage.EARLY_WAR

    def _play_turn(self, resume: tuple | None = None) -> Iterator[Decision]:
        """One turn; *resume* = ``(round, side_index)`` re-enters it at an action round."""
        state = self.state
        if resume is None:
            stage = self._current_stage()
            state.action_rounds_this_turn = ACTION_ROUNDS[stage]
            state.action_round = 0
            state.space_attempts = [0, 0]

            self._add_stage_cards()
            self._deal_cards(stage)
            state.note(f"--- turn {state.turn} ({stage.value}) ---")

            yield from self._headline_phase()
            if state.is_over:
                return
            finished = yield from self._action_rounds(0, 0, resumed=False)
        else:
            finished = yield from self._action_rounds(resume[0], resume[1], resumed=True)
        if not finished:
            return

        yield from self._turn_end()

    def _action_rounds(
        self, round_number: int, side_index: int, *, resumed: bool
    ) -> Iterator[Decision]:
        """The action rounds of a turn. Returns False if the game ended in one of them.

        *resumed*: enter directly at *side_index* of round *round_number* (a checkpoint),
        skipping the round's header, which had already run when the checkpoint was made.
        """
        state = self.state
        # The Space Station holder may play 8 action rounds, and so may the US on the
        # turn North Sea Oil is played. Both are per-side, and both can arrive in the
        # middle of the turn -- North Sea Oil played in an action round, the Space Station
        # reached or lost by a space race roll -- so the count is re-read every round.
        while True:
            if resumed:
                resumed = False
            else:
                round_number += 1
                rounds = {side: self._action_rounds_for(side) for side in Side}
                if round_number > max(rounds.values()):
                    break
                state.action_round = round_number
                state.phase = Phase.ACTION_ROUND
                side_index = 0
            for index in range(side_index, 2):
                side = _ROUND_ORDER[index]
                self._checkpoint(("ar", round_number, index))
                if state.is_over:
                    return False
                if round_number > self._action_rounds_for(side):
                    continue
                state.defcon_moved_to_two = False
                yield from self._action_round(side)
                if state.is_over:
                    return False
                # NORAD: "at the conclusion of any Action Round in which the DEFCON
                # marker moves to the '2' box" -- at any point in the round, even if it
                # has moved off again since.
                from .events.special import norad_end_of_action_round

                yield from norad_end_of_action_round(self, state.defcon_moved_to_two)
        return True

    def _action_rounds_for(self, side: Side) -> int:
        """Action rounds *side* may take this turn, as things stand right now."""
        state = self.state
        rounds = spacerace.action_rounds_for(
            state.space_race, side, state.action_rounds_this_turn
        )
        if (
            side is Side.USA
            and state.effect_data("North Sea Oil", "extra_action_round_turn") == state.turn
        ):
            # "US may play 8 cards this turn": eight in all, not one more than the
            # Space Station already allows.
            rounds = max(rounds, spacerace.SPACE_STATION_ROUNDS)
        return rounds

    def _add_stage_cards(self) -> None:
        """Shuffle in any stage whose cards enter the deck on this turn.

        On turns 4 and 8 the new cards join the *existing* draw deck without the
        discard pile being folded back in (rule 4.4).
        """
        for stage, start in STAGE_START_TURN.items():
            if start != self.state.turn or stage in self.state.stages_in_deck:
                continue
            self.state.stages_in_deck.add(stage)
            self.state.shuffle_into_deck(
                deck_for_stage(stage, optional_cards=self.optional_cards)
            )
            self.state.note(f"{stage.value} cards shuffled into the deck")

    def _deal_cards(self, stage: Stage) -> None:
        """Top both hands up to the stage's target size.

        The China Card never counts toward hand size, which falls out of it being
        tracked separately from ``hands``.
        """
        state = self.state
        target = HAND_SIZE[stage]
        # Deal alternately so a mid-deal reshuffle cannot systematically favour a side.
        while True:
            dealt = False
            for side in (Side.USSR, Side.USA):
                if len(state.hands[side]) >= target:
                    continue
                drawn = state.draw_card()
                if drawn is None:
                    continue
                state.hands[side].append(drawn)
                dealt = True
            if not dealt:
                break
        state.note(
            f"hands dealt to {target}: USSR {len(state.hands[Side.USSR])}, "
            f"US {len(state.hands[Side.USA])}"
        )

    # -- headline --------------------------------------------------------- #

    def _headline_phase(self) -> Iterator[Decision]:
        state = self.state
        state.phase = Phase.HEADLINE
        state.action_round = 0
        state.headline = {0: None, 1: None}
        state.headline_public = set()
        state.reach_snapshot = None

        # Man in Earth Orbit: its holder chooses last, after the opponent has revealed.
        order = list(Side)
        ability = spacerace.OPPONENT_HEADLINE_FIRST
        reveal_first: Side | None = None
        if spacerace.has_ability(state.space_race, Side.USA, ability):
            order = [Side.USSR, Side.USA]
            reveal_first = Side.USSR
        elif spacerace.has_ability(state.space_race, Side.USSR, ability):
            order = [Side.USA, Side.USSR]
            reveal_first = Side.USA

        for side in order:
            candidates = [
                name
                for name in state.hand(side)
                if CARDS[name].can_headline
            ]
            if not candidates:
                continue
            chosen = yield from self.choose_card(
                side,
                candidates,
                "Choose your headline card",
                dtype=DecisionType.HEADLINE,
                labels={n: headline_label(n) for n in candidates},
            )
            if chosen is not None:
                state.headline[int(side)] = chosen
                state.hand(side).remove(chosen)
                if side is reveal_first:
                    # Rule 6.4.4: selected *and revealed* before the holder chooses.
                    state.headline_public.add(int(side))
                    state.note(
                        f"headline revealed first (Man in Earth Orbit): {chosen}", side
                    )

        state.headline_public = {int(side) for side in Side}
        picks = {side: state.headline[int(side)] for side in Side}
        shown = ", ".join(f"{s.label}: {picks[s] or '-'}" for s in Side)
        state.note(f"headlines revealed -- {shown}")

        # Higher operations value resolves first; the US card wins ties.
        def sort_key(side: Side) -> tuple[int, int]:
            name = picks[side]
            ops = CARDS[name].ops if name else -1
            forced = 1 if name and CARDS[name].resolve_headline_first else 0
            return (-forced, -ops, 0 if side is Side.USA else 1)

        # Defectors headlined by the US cancels the USSR headline outright.
        cancelled: set[Side] = set()
        if picks[Side.USA] == "Defectors" and picks[Side.USSR] is not None:
            cancelled.add(Side.USSR)
        state.headline_cancelled = set()

        for side in sorted(Side, key=sort_key):
            name = picks[side]
            if name is None or state.is_over:
                continue
            state.headline[int(side)] = None

            # Defectors played mid-headline (Grain Sales drawing it) cancels a USSR
            # headline that has not resolved yet (user ruling 2026-11-25).
            if side in cancelled or int(side) in state.headline_cancelled:
                state.note(f"{side.label} headline {name} cancelled by Defectors", side)
                self._retire_card(name, side, as_event=False)
                continue

            c = CARDS[name]
            # Headlining the opponent's event means the opponent implements it, though
            # the player who headlined it still carries the blame for DEFCON.
            actor = side if c.side is None else c.side
            if not c.is_scoring and not self._event_is_playable(name, actor):
                state.note(f"headline {name} cannot take effect; discarded", side)
                self._retire_card(name, side, as_event=False)
                continue

            # Flower Power scores a war card the US headlines, as for any US play.
            self._check_flower_power(name, side)
            if state.is_over:
                break

            note = f"headline resolves: {name}"
            if actor is not side:
                note += f" (implemented by {actor.label})"
            state.note(note, side)
            # The player who headlined the card carries the DEFCON blame even when the
            # opponent implements the event.
            state.phasing_player = side
            yield from self._resolve_event(name, actor, from_headline=True)
        state.phasing_player = None

    # -- action rounds ---------------------------------------------------- #

    def _action_round(self, side: Side) -> Iterator[Decision]:
        state = self.state
        state.player = side
        state.phasing_player = side
        state.ar_sequence += 1
        # Rule 6.1.1: operations influence goes next to markers that were in place at
        # the start of the phasing player's action round, whatever an event does first.
        state.reach_snapshot = {
            int(s): rules.reachable_countries(state, s) for s in Side
        }

        yield from self._fire_deferred(side, "start")
        if state.is_over:
            return

        yield from self._offer_cuban_missile_crisis_cancel(side)
        if state.is_over:
            return

        trap = "Quagmire" if side is Side.USA else "Bear Trap"
        if state.in_play(trap, owner=side):
            # Quagmire and Bear Trap replace the action round with a discard and roll.
            yield from self._settle_we_will_bury_you(side, None)
            if not state.is_over:
                yield from self._resolve_trap(side, trap)
        elif not state.playable_hand(side):
            yield from self._settle_we_will_bury_you(side, None)
            state.note("no cards to play; action round skipped", side)
        else:
            yield from self._take_action_round(side)

        # Missile Envy's compulsion lasts until the card is spent (it can be put off by
        # a scoring card that must be played, even into the next turn: FAQ).
        forced = state.must_play[int(side)]
        if forced is not None and forced not in state.hand(side):
            state.must_play[int(side)] = None

        if not state.is_over:
            yield from self._fire_deferred(side, "end")

    def _settle_we_will_bury_you(self, side: Side, chosen: str | None) -> Iterator[Decision]:
        """We Will Bury You scores "the moment the US does not play UN Intervention"
        (FAQ #50), before any US VP in that round.

        Called once the US has chosen what to do with its action round. A card that can
        still turn into a UN Intervention event -- UN Intervention itself, or Missile
        Envy / Star Wars / Grain Sales, which can fetch it (FAQ #50) -- leaves the
        decision open; :meth:`_settle_we_will_bury_you_before` then settles it the
        moment the chain commits to anything else, and the end of the round is the last
        resort.
        """
        if chosen in WE_WILL_BURY_YOU_CHAIN:
            return
        yield from self._pay_we_will_bury_you(side)

    def _settle_we_will_bury_you_before(
        self, card: str | None, *, as_event: bool
    ) -> Iterator[Decision]:
        """Settle a pending We Will Bury You as the US round commits to *card*.

        Hooked into every resolution inside an action round -- an event
        (:meth:`_resolve_event`), a card's use (:meth:`_perform_use`) and an
        operations play (:meth:`operations`). In the US player's round, anything but a
        UN Intervention event, or the event of a card that can still fetch one, means
        the US has not played UN Intervention: the 3 VP are scored there and then,
        before whatever that resolution awards.
        """
        state = self.state
        if state.phase is not Phase.ACTION_ROUND or state.phasing_player is not Side.USA:
            return
        if as_event and card in WE_WILL_BURY_YOU_CHAIN:
            return
        yield from self._pay_we_will_bury_you(Side.USA)

    def _pay_we_will_bury_you(self, side: Side) -> Iterator[Decision]:
        from .events import deferred_handler_for
        from .events.mid_war import WE_WILL_BURY_YOU

        state = self.state
        ready = [
            t for t in state.deferred
            if t.kind == WE_WILL_BURY_YOU and t.player is side
            and state.ar_sequence > t.not_before
        ]
        for trigger in ready:
            if state.is_over:
                return
            state.deferred.remove(trigger)
            yield from deferred_handler_for(trigger.kind)(self, trigger)

    def _fire_deferred(self, side: Side, when: str) -> Iterator[Decision]:
        """Resolve triggers a card scheduled for this point in *side*'s action round."""
        from .events import deferred_handler_for

        state = self.state
        ready = [
            t
            for t in state.deferred
            if t.player is side and t.when == when and state.ar_sequence > t.not_before
        ]
        for trigger in ready:
            if state.is_over:
                return
            state.deferred.remove(trigger)
            state.note(f"{trigger.card}: deferred effect resolves ({trigger.kind})", side)
            yield from deferred_handler_for(trigger.kind)(self, trigger)

    def _offer_cuban_missile_crisis_cancel(self, side: Side) -> Iterator[Decision]:
        """Let the threatened player buy the crisis off.

        "May be cancelled at any time if the USSR player removes two Influence from
        Cuba or the US player removes 2 Influence from either West Germany or Turkey"
        -- two from one country, not one from each (FAQ #40). Offered at the start of
        the threatened player's action round and again whenever they are about to be
        given a coup they might take, in either player's round (FAQ #40: "at any
        time"). Those are the only moments the choice can matter.
        """
        state = self.state
        effect = state.effects.get("Cuban Missile Crisis")
        if effect is None or effect.owner is None or effect.owner is side:
            return

        sources = ["Cuba"] if side is Side.USSR else ["West Germany", "Turkey"]
        payable = [n for n in sources if state.inf(side, n) >= 2]
        if not payable:
            return

        where = " or ".join(payable)
        cancel = yield from self.confirm(
            side,
            f"Cuban Missile Crisis is in effect: couping would lose you the game. "
            f"Remove 2 of your influence from {where} to cancel it?",
            card="Cuban Missile Crisis",
        )
        if not cancel:
            return

        name = yield from self.choose_country(
            side,
            payable,
            "Remove 2 influence from which country to cancel the Cuban Missile Crisis?",
            dtype=DecisionType.REMOVE_INFLUENCE,
            labels=self._influence_labels(payable),
        )
        state.remove_inf(side, name, 2)
        state.note(f"{side.label} influence removed: {name} -2", side)
        state.remove_effect("Cuban Missile Crisis")
        state.note("Cuban Missile Crisis cancelled", side)

    def offer_crisis_cancel_before_coup(self, side: Side) -> Iterator[Decision]:
        """Before *side* is offered a coup, give them the chance to cancel the crisis."""
        yield from self._offer_cuban_missile_crisis_cancel(side)

    def _rounds_left(self, side: Side) -> int:
        """Action rounds *side* has left this turn, the current one included."""
        return max(1, self._action_rounds_for(side) - self.state.action_round + 1)

    def _take_action_round(self, side: Side) -> Iterator[Decision]:
        state = self.state
        hand = state.playable_hand(side)
        scoring = [n for n in state.hand(side) if CARDS[n].is_scoring]
        # Scoring cards may never be held (rule 4.5D / 10.1.5): once the rounds left
        # cannot spare one, a scoring card has to be playable now.
        scoring_due = bool(scoring) and len(scoring) >= self._rounds_left(side)

        options = list(hand)
        if scoring_due and CHINA_CARD in options:
            # Rule 9.5: the China Card may not be played if it prevents the play of a
            # scoring card.
            options.remove(CHINA_CARD)

        # Missile Envy compels its recipient to spend the card it handed over -- unless
        # a scoring card must be played instead, which puts the compulsion off (FAQ).
        forced = state.must_play[int(side)]
        if forced is not None and forced in options:
            if not scoring_due:
                yield from self._settle_we_will_bury_you(side, forced)
                state.note(f"compelled to play {forced} for operations", side)
                yield from self.play_card(forced, side, forced_use=True)
                return
            # "would still have to play the scoring card and hold Missile Envy".
            options = list(scoring)

        # Rule 9.8: play of the China Card can never be compelled by a shortage of
        # cards, so with nothing else left the player may pass. An extra round from the
        # Space Station or North Sea Oil ("may play 8") may be forgone too (user ruling
        # 2026-11-25).
        only_china = options == [CHINA_CARD]
        extra_round = state.action_round > state.action_rounds_this_turn
        may_pass = only_china or extra_round
        chosen = yield from self.choose_card(
            side,
            options,
            "Choose a card to play"
            + (" (or pass: the China Card is never compelled)" if only_china else "")
            + (" (or pass: this extra action round is optional)" if extra_round else ""),
            dtype=DecisionType.PLAY_CARD,
            allow_pass=may_pass,
            labels={n: self._card_label(n, side) for n in options},
        )
        if chosen is None:
            state.note("passes the action round", side)
            yield from self._settle_we_will_bury_you(side, None)
            return
        yield from self._settle_we_will_bury_you(side, chosen)
        if state.is_over:
            return
        if chosen == forced:
            state.note(f"compelled to play {forced} for operations", side)
            yield from self.play_card(forced, side, forced_use=True)
            return
        yield from self.play_card(chosen, side)

    def _card_label(self, name: str, side: Side) -> str:
        c = CARDS[name]
        bits = [f"{c.ops}op" if c.ops else "scoring"]
        if c.side is None:
            bits.append("neutral")
        elif c.side is side:
            bits.append("yours")
        else:
            bits.append(f"{c.side.label} event")
        if c.remove_on_event:
            bits.append("removed if evented")
        return ", ".join(bits)

    def play_card(
        self, name: str, side: Side, *, forced_use: bool = False
    ) -> Iterator[Decision]:
        """Play *name* from *side*'s hand, resolving the chosen use.

        ``state.playing_card`` is saved and restored around the whole play, so that a
        card which causes another to be played -- Grain Sales to Soviets, Missile Envy,
        Star Wars -- reports the inner card while it resolves and the outer one after.

        *forced_use* restricts the play to operations, for a card the player is
        compelled to spend rather than free to use as they like.
        """
        state = self.state
        c = CARDS[name]

        if name == CHINA_CARD:
            pass  # handled after resolution: it passes to the opponent
        elif name in state.hand(side):
            state.hand(side).remove(name)

        # Restored explicitly on the normal path rather than in a `finally`. A `finally`
        # here also runs when a discarded Game is garbage-collected -- Python throws
        # GeneratorExit into the suspended driver -- which would mutate a GameState
        # somebody is still holding a reference to, after they read it.
        previous, state.playing_card = state.playing_card, name

        if c.is_scoring:
            state.note(f"plays {name} (scoring)", side)
            yield from self._resolve_event(name, side)
            state.resolve_card(name, as_event=True)
            state.playing_card = previous
            return

        uses = yield from self._choose_use(name, side, operations_only=forced_use)
        yield from self._perform_use(name, side, uses)
        state.playing_card = previous

    def _choose_use(
        self, name: str, side: Side, *, operations_only: bool = False
    ) -> Iterator[Decision]:
        """Ask how the card is being used, returning an :class:`OpsUse`.

        *operations_only* drops the event and space race options, for a card the player
        has been compelled to spend on operations.
        """
        state = self.state
        c = CARDS[name]
        options: list[Action] = []

        own_event = c.event_belongs_to(side)
        if not operations_only and own_event and self._event_is_playable(name, side):
            options.append(use_action(OpsUse.EVENT, "resolve the event"))

        ops = self.effective_ops(name, side)
        bonus = " (+1 more if every point goes to Asia)" if name == CHINA_CARD else ""
        if Region.SOUTHEAST_ASIA in self._ops_bonus_regions(name, side):
            bonus += " (+1 more if every point goes to Southeast Asia: Vietnam Revolts)"
        if ops > 0:
            options.append(use_action(OpsUse.INFLUENCE, f"place {ops} influence{bonus}"))
            if any(rules.can_coup(state, side, n) for n in COUNTRY_ORDER):
                options.append(use_action(OpsUse.COUP, f"coup with {ops} ops{bonus}"))
            if any(rules.can_realign(state, side, n) for n in COUNTRY_ORDER):
                options.append(
                    use_action(OpsUse.REALIGN, f"{ops} realignment rolls{bonus}")
                )

        if not operations_only and self._can_space(name, side):
            box = spacerace.next_box(state.space_race[side])
            assert box is not None
            options.append(use_action(OpsUse.SPACE, f"space race: {box.name}"))

        if not options:
            # A card with an unplayable event and no operations value can only be
            # discarded, which still uses up the action round.
            options.append(use_action(OpsUse.DISCARD, "discard with no effect"))

        if len(options) == 1:
            return OpsUse(options[0].value)

        action = yield from self.ask(
            Decision(
                DecisionType.CARD_USE,
                side,
                f"How will you use {name}? ({c.ops} ops"
                + (f", effective {ops}" if ops != c.ops else "")
                + ")",
                tuple(options),
                {"card": name, "ops": ops, "event_text": c.event_text},
            )
        )
        return OpsUse(action.value)

    def _perform_use(self, name: str, side: Side, use: OpsUse) -> Iterator[Decision]:
        state = self.state
        c = CARDS[name]

        yield from self._settle_we_will_bury_you_before(name, as_event=use is OpsUse.EVENT)
        if state.is_over:
            return
        ops = self.effective_ops(name, side)

        # Flower Power scores against war cards however they are used, except when
        # discarded to the space race.
        if use is not OpsUse.SPACE:
            self._check_flower_power(name, side)

        if use is OpsUse.EVENT:
            state.note(f"plays {name} for its event", side)
            yield from self._resolve_event(name, side)
            self._retire_card(name, side, as_event=True)
            return

        if use is OpsUse.SPACE:
            yield from self._space_race_attempt(name, side)
            self._retire_card(name, side, as_event=False)
            return

        if use is OpsUse.DISCARD:
            state.note(f"discards {name} with no effect", side)
            self._retire_card(name, side, as_event=False)
            return

        # Using a card that carries the opponent's event triggers that event too.
        opponent_event = (
            c.side is not None
            and c.side is not side
            and self._event_is_playable(name, c.side)
        )

        event_first = False
        if opponent_event:
            event_first = yield from self.confirm(
                side,
                f"{name} carries the {c.side.label} event, which will occur. "
                "Resolve the event before your operations?",
                card=name,
                event_text=c.event_text,
            )

        if opponent_event and event_first:
            yield from self._resolve_event(name, c.side, triggered_by_opponent=True)

        state.note(f"plays {name} for {ops} ops ({use.value})", side)
        yield from self.conduct_operations(
            side,
            ops,
            use,
            card_name=name,
            bonus_regions=self._ops_bonus_regions(name, side),
        )

        if opponent_event and not event_first:
            yield from self._resolve_event(name, c.side, triggered_by_opponent=True)

        self._retire_card(name, side, as_event=opponent_event)

    def _check_flower_power(self, name: str, side: Side) -> None:
        """The USSR scores 2 VP whenever the US plays a war card, while it is in play."""
        from .events.mid_war2 import WAR_CARDS

        if side is not Side.USA or name not in WAR_CARDS:
            return
        if not self.state.in_play("Flower Power"):
            return
        # "unless made unplayable by a subsequent card" (FAQ #59): Arab-Israeli War
        # after Camp David Accords does not pay out.
        owner = CARDS[name].side
        if not self._event_is_playable(name, owner if owner is not None else side):
            return
        self.state.award_vp(Side.USSR, 2)
        self.state.note(f"Flower Power: {name} played by the US, USSR +2 VP", side)
        self._check_victory_points()

    def _retire_card(self, name: str, side: Side, *, as_event: bool) -> None:
        """Send a played card where it belongs, honouring cards that stay in play."""
        from .events.special import china_card_cancels_formosan_resolution

        state = self.state
        if name == CHINA_CARD:
            state.china_card_owner = side.opponent
            state.china_card_face_up = False
            state.note("the China Card passes to the opponent", side)
            china_card_cancels_formosan_resolution(self, side)
            return
        if state.in_play(name):
            return  # the event put it into play; it is not discarded
        if state._is_accounted(name):
            return  # the event moved it somewhere already (Missile Envy changes hands)
        state.resolve_card(name, as_event=as_event)

    # -- operations ------------------------------------------------------- #

    def effective_ops(self, name: str, side: Side) -> int:
        """A card's operations value after all modifiers, floored at 1 when nonzero."""
        state = self.state
        c = CARDS[name]
        if c.ops == 0:
            return 0
        ops = c.ops

        # Containment / Brezhnev Doctrine add one to their owner's cards.
        if state.in_play("Containment", owner=side):
            ops += 1
        if state.in_play("Brezhnev Doctrine", owner=side):
            ops += 1
        # Red Scare/Purge subtracts one from the opponent's cards.
        if state.in_play("Red Scare/Purge", owner=side.opponent):
            ops -= 1

        return max(1, min(4, ops))

    def _ops_bonus_regions(self, name: str, side: Side) -> tuple[Region, ...]:
        """Regions in which spending every point of *name* earns one more point.

        The China Card is worth one more when every point goes to Asia; Vietnam Revolts
        gives the USSR one more for a card spent entirely in Southeast Asia for the rest
        of the turn. They stack (rule 7.4 example: the China Card all in Southeast Asia
        under Vietnam Revolts is worth 6).
        """
        regions: list[Region] = []
        if name == CHINA_CARD:
            regions.append(Region.ASIA)
        if side is Side.USSR and self.state.in_play("Vietnam Revolts", owner=Side.USSR):
            regions.append(Region.SOUTHEAST_ASIA)
        return tuple(regions)

    @staticmethod
    def _bonus_list(
        bonus_region: Region | None, bonus_regions: Sequence[Region]
    ) -> list[Region]:
        out = list(bonus_regions)
        if bonus_region is not None and bonus_region not in out:
            out.append(bonus_region)
        return out

    def conduct_operations(
        self,
        side: Side,
        ops: int,
        use: OpsUse,
        *,
        card_name: str | None = None,
        free: bool = False,
        bonus_region: Region | None = None,
        bonus_regions: Sequence[Region] = (),
    ) -> Iterator[Decision]:
        """Spend *ops* operations points as influence, a coup, or realignments.

        *bonus_regions* (or the single *bonus_region*) implement "+1 Operations value
        when all points are used in" a region: the China Card (Asia) and Vietnam Revolts
        (Southeast Asia). The bonus cannot be folded into :meth:`effective_ops`, which
        runs before the player has chosen any target, so it is granted at the point
        where it becomes knowable and the extra point is confined to the same region.
        """
        del card_name  # kept for future card-specific ops modifiers
        regions = self._bonus_list(bonus_region, bonus_regions)
        if use is OpsUse.INFLUENCE:
            yield from self.spend_operations_influence(side, ops, bonus_regions=regions)
        elif use is OpsUse.COUP:
            yield from self._coup_operation(side, ops, free=free, bonus_regions=regions)
        elif use is OpsUse.REALIGN:
            yield from self._realign_operation(side, ops, bonus_regions=regions)
        else:
            raise ValueError(f"{use} is not an operations use")

    @staticmethod
    def _all_in(names: Iterable[str], region: Region) -> bool:
        names = list(names)
        return bool(names) and all(region in COUNTRIES[n].regions for n in names)

    def event_ops(self, ops: int, side: Side) -> int:
        """An event's "as if they played an N Ops card" value for *side*, modified.

        The player's own modifiers apply (rule 7.4.2 Example 3: CIA Created under
        Containment is worth 2): Containment / Brezhnev Doctrine add one to a maximum
        of 4, Red Scare/Purge takes one off to a minimum of 1.
        """
        state = self.state
        value = ops
        if state.in_play("Containment", owner=side):
            value += 1
        if state.in_play("Brezhnev Doctrine", owner=side):
            value += 1
        value = min(max(value, ops), 4) if value > ops else value
        if state.in_play("Red Scare/Purge", owner=side.opponent):
            value -= 1
        return max(1, value)

    def operations(
        self,
        side: Side,
        ops: int,
        *,
        uses: Sequence[OpsUse] = (OpsUse.INFLUENCE, OpsUse.COUP, OpsUse.REALIGN),
        prompt: str = "",
        bonus_regions: Sequence[Region] = (),
    ) -> Iterator[Decision]:
        """Let *side* conduct a normal operations play worth *ops* (already modified).

        Used for every "conduct operations" that is not a free coup: events worded "as
        if they played an N Ops card", Grain Sales and UN Intervention spending a real
        card, and a card taken by Missile Envy. It is exactly an operations play made
        outside one's own action round (user ruling 2026-11-25): DEFCON geography
        applies, a coup counts toward military operations and degrades DEFCON in a
        battleground, and DEFCON 1 loses for the phasing player.
        """
        state = self.state
        yield from self._settle_we_will_bury_you_before(None, as_event=False)
        if state.is_over:
            return
        if OpsUse.COUP in uses:
            # The threatened player may cancel the Cuban Missile Crisis before choosing
            # ("at any time", FAQ #40).
            yield from self._offer_cuban_missile_crisis_cancel(side)
            if state.is_over:
                return
        available: list[Action] = []
        if OpsUse.INFLUENCE in uses and rules.placeable_countries(
            state, side, reachable=self.reachable_for(side)
        ):
            available.append(use_action(OpsUse.INFLUENCE, f"place {ops} influence"))
        if OpsUse.COUP in uses and any(rules.can_coup(state, side, n) for n in COUNTRY_ORDER):
            available.append(use_action(OpsUse.COUP, f"coup with {ops} ops"))
        if OpsUse.REALIGN in uses and any(
            rules.can_realign(state, side, n) for n in COUNTRY_ORDER
        ):
            available.append(use_action(OpsUse.REALIGN, f"{ops} realignment rolls"))

        if not available:
            state.note(f"{ops} operations point(s) wasted: nothing legal", side)
            return

        if len(available) == 1:
            use = OpsUse(available[0].value)
        else:
            action = yield from self.ask(
                Decision(
                    DecisionType.CARD_USE,
                    side,
                    prompt or f"Conduct operations as if playing a {ops} Op card",
                    tuple(available),
                    {"ops": ops},
                )
            )
            use = OpsUse(action.value)

        yield from self.conduct_operations(side, ops, use, bonus_regions=bonus_regions)

    def free_operations(
        self,
        side: Side,
        ops: int,
        *,
        uses: Sequence[OpsUse] = (OpsUse.INFLUENCE, OpsUse.COUP, OpsUse.REALIGN),
        prompt: str = "",
    ) -> Iterator[Decision]:
        """Events worded "conduct operations as if they played an *ops* Op card".

        A normal operations play (see :meth:`operations`) at the event's value as
        modified for *side*. Vietnam Revolts' +1 applies to the USSR if every point goes
        to Southeast Asia (user ruling 2026-11-25). The name is historical: these
        operations are not "free" in the sense of rule 8.2.5.
        """
        value = self.event_ops(ops, side)
        bonus: tuple[Region, ...] = ()
        if side is Side.USSR and self.state.in_play("Vietnam Revolts", owner=Side.USSR):
            bonus = (Region.SOUTHEAST_ASIA,)
        yield from self.operations(
            side, value, uses=uses, prompt=prompt, bonus_regions=bonus
        )

    def spend_operations_influence(
        self,
        side: Side,
        ops: int,
        *,
        bonus_region: Region | None = None,
        bonus_regions: Sequence[Region] = (),
    ) -> Iterator[Decision]:
        """Place influence with operations points.

        Costs 2 per point into an enemy-controlled country, re-evaluated after every
        single point so that breaking control mid-placement makes the rest cheaper.
        Legal targets are frozen at the start of the placement, so newly placed
        influence cannot be chained outward to reach further countries.

        For each of *bonus_regions* (or *bonus_region*) that every point went into, one
        extra point is granted, which must also go there.
        """
        state = self.state
        budget = ops
        placed: dict[str, int] = {}
        reachable = self.reachable_for(side)
        pending = self._bonus_list(bonus_region, bonus_regions)
        granted: list[Region] = []

        while True:
            if budget <= 0:
                earned = next((r for r in pending if self._all_in(placed, r)), None)
                if earned is None:
                    break
                pending.remove(earned)
                granted.append(earned)
                budget = 1
                state.note(f"all operations points used in {earned}: +1 bonus point", side)

            candidates = rules.placeable_countries(
                state, side, budget=budget, reachable=reachable
            )
            if granted:
                # A bonus point has to stay in the region that earned it.
                candidates = [
                    n for n in candidates
                    if all(r in COUNTRIES[n].regions for r in granted)
                ]
            if not candidates:
                if not granted:
                    state.note(
                        f"{budget} operations point(s) wasted: nowhere legal to place", side
                    )
                break
            name = yield from self.choose_country(
                side,
                candidates,
                f"Place influence ({budget} ops left)",
                dtype=DecisionType.PLACE_INFLUENCE,
                labels={
                    n: f"cost {rules.influence_cost(state, side, n)}, "
                       f"{self._influence_labels([n])[n]}"
                    for n in candidates
                },
                remaining=budget,
            )
            assert name is not None
            budget -= rules.influence_cost(state, side, name)
            state.add_inf(side, name, 1)
            placed[name] = placed.get(name, 0) + 1

        if placed:
            state.note(
                "influence: " + ", ".join(f"{n} +{v}" for n, v in placed.items()), side
            )
        return placed

    def coup(self, side: Side, name: str, ops: int, *, free: bool = False) -> rules.CoupResult:
        """Resolve a coup, with every consequence that hangs off couping.

        Events that grant coups should call this rather than :func:`rules.resolve_coup`
        directly: this is the single choke point where Yuri and Samantha pays out, the
        Cuban Missile Crisis loss is checked, and a battleground coup that drops DEFCON
        to 1 ends the game against the right player.
        """
        from .events.special import yuri_and_samantha_on_coup

        # Cuban Missile Crisis: the moment the constrained player initiates a coup they
        # lose, whoever is phasing (FAQ #40/#62; the card overrides rule 8.1.3).
        self._check_cuban_missile_crisis(side)
        yuri_and_samantha_on_coup(self, side)
        result = rules.resolve_coup(self.state, side, name, ops, free=free)
        # DEFCON 1 loses for the phasing player, not necessarily the coup maker: a
        # free coup from an opponent's event played for operations is the phasing
        # player's responsibility (rule 8.1.3, FAQ #62 Lone Gunman).
        blame = self.state.phasing_player if self.state.phasing_player is not None else side
        self._check_defcon_loss(blame)
        return result

    def _check_cuban_missile_crisis(self, actor: Side) -> None:
        """Couping while the opponent's Cuban Missile Crisis stands loses the game."""
        effect = self.state.effects.get("Cuban Missile Crisis")
        if effect is None or effect.owner is None or effect.owner is actor:
            return
        if self.state.is_over:
            return
        self.state.note(
            f"{actor.label} couped during the Cuban Missile Crisis", actor
        )
        self.state.finish(effect.owner, WinReason.CUBAN_MISSILE_CRISIS)

    def reachable_for(self, side: Side) -> frozenset[str]:
        """Where *side* may place operations influence now (rule 6.1.1 snapshot)."""
        state = self.state
        snap = state.reach_snapshot.get(int(side)) if state.reach_snapshot else None
        if snap is not None and state.phase is Phase.ACTION_ROUND:
            return snap
        return rules.reachable_countries(state, side)

    def _coup_operation(
        self,
        side: Side,
        ops: int,
        *,
        free: bool = False,
        bonus_region: Region | None = None,
        bonus_regions: Sequence[Region] = (),
    ) -> Iterator[Decision]:
        state = self.state
        targets = [n for n in COUNTRY_ORDER if rules.can_coup(state, side, n)]
        name = yield from self.choose_country(
            side,
            targets,
            f"Coup target ({ops} ops)",
            dtype=DecisionType.COUP_TARGET,
            labels={n: self.coup_label(n, ops) for n in targets},
            ops=ops,
        )
        if name is None:
            return
        # A coup spends every point on one country, so the bonus is decided by the target.
        for region in self._bonus_list(bonus_region, bonus_regions):
            if region in COUNTRIES[name].regions:
                ops += 1
                state.note(f"all operations points used in {region}: coup at {ops} ops", side)
        self.coup(side, name, ops, free=free)

    def coup_label(self, name: str, ops: int) -> str:
        stability = country_stability(name)
        need = stability * 2
        # The die is 1..6, so this is the number of faces that would succeed.
        succeeds_on = max(0, min(6, ops - need + 6))
        return (
            f"stab {stability}, needs >{need}, succeeds on {succeeds_on}/6, "
            + self._influence_labels([name])[name]
        )

    def _realign_operation(
        self,
        side: Side,
        ops: int,
        *,
        bonus_region: Region | None = None,
        bonus_regions: Sequence[Region] = (),
    ) -> Iterator[Decision]:
        state = self.state
        chosen: list[str] = []
        remaining = ops
        pending = self._bonus_list(bonus_region, bonus_regions)
        granted: list[Region] = []

        while True:
            if remaining <= 0:
                earned = next((r for r in pending if self._all_in(chosen, r)), None)
                if earned is None:
                    return
                pending.remove(earned)
                granted.append(earned)
                remaining = 1
                state.note(f"all realignments made in {earned}: +1 bonus roll", side)

            targets = [n for n in COUNTRY_ORDER if rules.can_realign(state, side, n)]
            if granted:
                targets = [
                    n for n in targets if all(r in COUNTRIES[n].regions for r in granted)
                ]
            if not targets:
                return
            name = yield from self.choose_country(
                side,
                targets,
                f"Realignment target ({remaining} roll(s) left)",
                dtype=DecisionType.REALIGN_TARGET,
                labels={
                    n: f"mod {rules.realignment_modifier(state, side, n):+d}, "
                       + self._influence_labels([n])[n]
                    for n in targets
                },
                allow_pass=True,
                remaining=remaining,
            )
            if name is None:
                return
            rules.resolve_realignment(state, side, name)
            # Each realignment roll spends one operations point (the same country may
            # be chosen again); the China Card's Asia bonus depends on every target.
            remaining -= 1
            chosen.append(name)

    # -- space race -------------------------------------------------------- #

    def _can_space(self, name: str, side: Side) -> bool:
        state = self.state
        box = spacerace.next_box(state.space_race[side])
        if box is None:
            return False
        if state.space_attempts[side] >= spacerace.attempts_allowed(state.space_race, side):
            return False
        return self.effective_ops(name, side) >= box.ops_required

    def _space_race_attempt(self, name: str, side: Side) -> Iterator[Decision]:
        state = self.state
        box = spacerace.next_box(state.space_race[side])
        assert box is not None
        state.space_attempts[side] += 1
        roll = self.roll()
        success = roll <= box.max_roll

        if success:
            gained = spacerace.victory_points(state.space_race, side, box.number)
            state.space_race[side] = box.number
            state.award_vp(side, gained)
            state.note(
                f"space race {box.name}: rolled {roll} <= {box.max_roll}, success, +{gained} VP",
                side,
            )
        else:
            state.note(
                f"space race {box.name}: rolled {roll} > {box.max_roll}, failed", side
            )
        return
        yield  # pragma: no cover - keeps this a generator for uniform `yield from`

    # -- events ------------------------------------------------------------ #

    def _event_is_playable(self, name: str, side: Side) -> bool:
        """Whether *name*'s event may currently fire for *side*."""
        from .events import is_playable

        return is_playable(self, name, side)

    def _resolve_event(
        self,
        name: str,
        side: Side,
        *,
        from_headline: bool = False,
        triggered_by_opponent: bool = False,
    ) -> Iterator[Decision]:
        from .events import handler_for

        state = self.state
        if not from_headline:
            yield from self._settle_we_will_bury_you_before(name, as_event=True)
            if state.is_over:
                return
        state.events_resolved.add(name)
        previous, state.event_player = state.event_player, side
        # An event reached directly rather than through play_card -- Five-Year Plan's
        # random discard, Star Wars pulling from the pile -- leaves its card in no pile
        # while it resolves.
        held = not state._is_accounted(name)
        if held:
            state.hold(name)

        handler = handler_for(name)
        if handler is None:
            if self.strict_events:
                raise NotImplementedError(f"no event handler registered for {name!r}")
            if name not in self.unimplemented:
                self.unimplemented.append(name)
            state.note(f"event {name} has no implementation; skipped", side)

        if handler is not None:
            ctx = EventContext(
                card=name,
                player=side,
                ops=CARDS[name].ops,
                triggered_by_opponent=triggered_by_opponent,
            )
            yield from handler(self, ctx)

        # Unwound explicitly rather than in a `finally`; see play_card for why.
        state.event_player = previous
        if held:
            state.release(name)

        if from_headline:
            # A headline card that did not put itself into play is discarded now.
            self._retire_card(name, side, as_event=True)

        self._check_victory_points()

    # -- traps -------------------------------------------------------------- #

    def _resolve_trap(self, side: Side, trap: str) -> Iterator[Decision]:
        """Quagmire / Bear Trap: discard a 2+ ops card and roll 1-4 to escape.

        Following the card and FAQ #42/#44:

        * a qualifying card is any held card (never the China Card) whose *modified*
          ops value is 2 or more (rule 7.4.2 Example 2); its event does not occur;
        * a Missile Envy the player is compelled to spend must be the next discard;
        * with no qualifying card the trap stays: the player may only play scoring
          cards, otherwise the round passes;
        * scoring cards take precedence: with as many scoring cards as rounds left, a
          scoring card is played instead of the discard. Otherwise a player holding a
          qualifying card must discard (user ruling 2026-11-25).
        """
        state = self.state
        hand = state.hand(side)
        scoring = [n for n in hand if CARDS[n].is_scoring]
        qualifying = [
            n for n in hand
            if not CARDS[n].is_scoring and self.effective_ops(n, side) >= 2
        ]
        forced = state.must_play[int(side)]
        if forced is not None and forced in qualifying:
            qualifying = [forced]

        if not qualifying:
            if not scoring:
                state.note(
                    f"{trap}: no card worth 2 or more operations; the round passes", side
                )
                return
            chosen = yield from self.choose_card(
                side,
                scoring,
                f"{trap}: no card worth 2 or more operations; play a scoring card",
                dtype=DecisionType.PLAY_CARD,
                labels={n: self._card_label(n, side) for n in scoring},
            )
            yield from self.play_card(chosen, side)
            return

        if scoring and len(scoring) >= self._rounds_left(side):
            # As many scoring cards as rounds left: one must be played now, discard or
            # not (scoring cards may never be held; user ruling 2026-11-25).
            chosen = yield from self.choose_card(
                side,
                scoring,
                f"{trap}: a scoring card must be played this round",
                dtype=DecisionType.PLAY_CARD,
                labels={n: self._card_label(n, side) for n in scoring},
            )
            yield from self.play_card(chosen, side)
            return

        chosen = yield from self.choose_card(
            side,
            qualifying,
            f"{trap}: discard a card with 2 or more operations points",
            dtype=DecisionType.DISCARD_CARD,
            labels={n: self._card_label(n, side) for n in qualifying},
        )
        assert chosen is not None
        state.hand(side).remove(chosen)
        state.discard_card(chosen)
        roll = self.roll()
        if roll <= 4:
            state.remove_effect(trap)
            state.note(f"{trap}: discarded {chosen}, rolled {roll} <= 4, escaped", side)
        else:
            state.note(f"{trap}: discarded {chosen}, rolled {roll} > 4, still stuck", side)

    # -- turn end ---------------------------------------------------------- #

    def _turn_end(self) -> Iterator[Decision]:
        state = self.state
        state.phase = Phase.TURN_END

        # Phase E: military operations. A VP win here ends the game before the held
        # cards are ever revealed (rule 4.5, FAQ 8.5).
        self._military_operations_check()
        if state.is_over:
            return

        # Phase F: holding a scoring card at the end of a turn loses the game.
        holders = [
            side
            for side in Side
            if any(CARDS[n].is_scoring for n in state.hand(side))
        ]
        if holders:
            # Both holding one is a US win by designer ruling, not a draw.
            winner = Side.USA if len(holders) == 2 else holders[0].opponent
            state.finish(winner, WinReason.HELD_SCORING_CARD)
            return

        # Space race box 6: its holder may discard one held card (rule 6.4.4). After
        # Phase F, so it can never be a scoring card (FAQ 6.4).
        yield from self._discard_held_card()

        # Effects that last only for the turn expire now.
        for name, effect in list(state.effects.items()):
            if effect.expires == "end_of_turn":
                state.remove_effect(name)
                state.note(f"{name} leaves play")
            elif effect.expires == "end_of_next_turn":
                effect.expires = "end_of_turn"

        if state.turn == FINAL_TURN:
            # The China Card is worth a point to whoever holds it at the very end. It is
            # part of final scoring, where 20 VP is not an automatic win (rule 10.3.2).
            state.final_scoring = True
            state.award_vp(state.china_card_owner, 1)
            state.note(f"{state.china_card_owner.label} holds the China Card: +1 VP")

        state.china_card_face_up = True
        state.change_defcon(+1)
        state.military_ops = [0, 0]
        state.note(f"turn {state.turn} ends; DEFCON now {state.defcon}")

        self._check_victory_points()

    def _discard_held_card(self) -> Iterator[Decision]:
        """Eagle/Bear Has Landed: the box 6 holder may discard one held card."""
        state = self.state
        if state.turn == FINAL_TURN:
            return  # nothing is held over past the last turn
        for side in Side:
            if not spacerace.has_ability(
                state.space_race, side, spacerace.DISCARD_HELD_CARD
            ):
                continue
            hand = list(state.hand(side))
            if not hand:
                continue
            chosen = yield from self.choose_card(
                side,
                hand,
                "Eagle/Bear Has Landed: discard one held card, or pass to keep it",
                dtype=DecisionType.DISCARD_CARD,
                allow_pass=True,
                labels={n: self._card_label(n, side) for n in hand},
            )
            if chosen is not None:
                state.hand(side).remove(chosen)
                state.discard_card(chosen)
                state.note(f"Eagle/Bear Has Landed: discards held card {chosen}", side)

    def _military_operations_check(self) -> None:
        """Award VP for falling short of the required military operations.

        The requirement equals the current DEFCON level. When both players are short
        only the *net* difference is scored, so the side with more military operations
        gains ``min(required, mine) - theirs`` victory points (rule 8.2.1).
        """
        state = self.state
        required = state.defcon
        ussr, usa = state.military_ops[Side.USSR], state.military_ops[Side.USA]

        ahead = Side.USSR if ussr > usa else Side.USA
        gain = min(required, state.military_ops[ahead]) - state.military_ops[ahead.opponent]
        if ussr != usa and gain > 0:
            state.award_vp(ahead, gain)
            state.note(
                f"military ops: USSR {ussr}, US {usa}, required {required} "
                f"-> {ahead.label} +{gain} VP"
            )
        else:
            state.note(f"military ops: USSR {ussr}, US {usa}, required {required} -> no VP")
        self._check_victory_points()

    # -- endings ----------------------------------------------------------- #

    def _check_victory_points(self) -> None:
        state = self.state
        if state.is_over or state.final_scoring:
            return
        if state.vp >= VP_LIMIT:
            state.finish(Side.USSR, WinReason.VICTORY_POINTS)
        elif state.vp <= -VP_LIMIT:
            state.finish(Side.USA, WinReason.VICTORY_POINTS)

    def _check_defcon_loss(self, actor: Side) -> None:
        """Dropping DEFCON to 1 loses the game for whoever did it."""
        if self.state.defcon <= 1 and not self.state.is_over:
            self.state.finish(actor.opponent, WinReason.DEFCON)

    def _final_scoring(self) -> None:
        """Score every region after turn 10, then award the game.

        Southeast Asia is deliberately excluded: its territory is already counted
        inside Asia, and its scoring card is removed from the game once played
        (rule 10.3.2). Europe control still wins outright, and every region is scored
        before victory is decided.
        """
        state = self.state
        state.final_scoring = True
        state.note("--- final scoring ---")

        for region in (
            Region.EUROPE,
            Region.ASIA,
            Region.MIDDLE_EAST,
            Region.AFRICA,
            Region.CENTRAL_AMERICA,
            Region.SOUTH_AMERICA,
        ):
            _, _, auto = rules.score_region(state, region)
            if auto is not None:
                # Europe is scored first, and controlling it still wins outright; 20 VP
                # reached here does not (rule 10.3.2), so the rest is a plain tally.
                state.finish(auto, WinReason.EUROPE_CONTROL)
                return

        if state.vp > 0:
            state.finish(Side.USSR, WinReason.FINAL_SCORING)
        elif state.vp < 0:
            state.finish(Side.USA, WinReason.FINAL_SCORING)
        else:
            state.finish(None, WinReason.DRAW)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def score_region(self, region: Region) -> Iterator[Decision]:
        """Resolve a scoring card, handling the Europe automatic victory.

        Shuttle Diplomacy is spent by the next Asia or Middle East scoring card, whether
        or not the USSR controlled a battleground there, and goes to the discard pile.
        The US chooses which USSR-controlled battleground it subtracts (asked only when
        there is more than one).
        """
        state = self.state
        candidates = rules.shuttle_diplomacy_candidates(state, region)
        excluded = yield from self.choose_country(
            Side.USA,
            candidates,
            f"Shuttle Diplomacy: choose the USSR battleground that does not count in "
            f"{region} scoring",
        )
        if excluded is not None:
            state.note(f"Shuttle Diplomacy: {excluded} does not count for the USSR", Side.USA)
        _, _, auto = rules.score_region(state, region, shuttle=True, shuttle_choice=excluded)

        if region in rules.SHUTTLE_DIPLOMACY_REGIONS and state.in_play("Shuttle Diplomacy"):
            state.remove_effect("Shuttle Diplomacy")
            state.note("Shuttle Diplomacy is discarded after scoring")

        if auto is not None:
            state.finish(auto, WinReason.EUROPE_CONTROL)
        else:
            self._check_victory_points()


def country_stability(name: str) -> int:
    return COUNTRIES[name].stability


def headline_label(name: str) -> str:
    c = CARDS[name]
    return f"{c.ops}op" + (f", {c.side.label} event" if c.side else ", neutral")


__all__ = [
    "ACTION_ROUNDS",
    "EventContext",
    "EventHandler",
    "FINAL_TURN",
    "Game",
    "HAND_SIZE",
    "WE_WILL_BURY_YOU_CHAIN",
]

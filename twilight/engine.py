"""The game driver.

The whole game is written as one generator. It yields a :class:`~twilight.decisions.Decision`
whenever it needs input and receives the chosen :class:`~twilight.decisions.Action` back,
which lets deeply nested card effects ask sub-questions without any explicit state
machine -- an event that says "remove 3 influence from Western Europe, no more than 2
per country" is a loop that yields a country choice three times.

The trade-off is that a running game holds live generator frames, which cannot be
deep-copied. :meth:`Game.clone` therefore replays the action history against the same
seed. That is exact (the engine is deterministic given seed + actions) but costs time
proportional to game length, so tree search over long games wants its own snapshotting.
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

        self.state = GameState(rng=random.Random(seed))
        self.seed = seed
        self.history: list[str] = []
        self._stages_added: set[Stage] = set()
        #: Cards whose event was skipped for lack of an implementation.
        self.unimplemented: list[str] = []

        self._driver = self._run()
        self.decision: Decision | None = self._advance(None)

    # ------------------------------------------------------------------ #
    # Driving
    # ------------------------------------------------------------------ #

    def _advance(self, action: Action | None) -> Decision | None:
        try:
            return self._driver.send(action)
        except StopIteration:
            return None

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

    def clone(self) -> "Game":
        """An independent copy, produced by replaying this game's action history.

        Exact but O(history); see the module docstring.
        """
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
    ) -> Iterator[Decision]:
        """Remove up to *points* of *target*'s influence, chosen by *chooser*."""
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

    def _run(self) -> Iterator[Decision]:
        yield from self._setup()

        while not self.state.is_over:
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

    def _play_turn(self) -> Iterator[Decision]:
        state = self.state
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

        # The Space Station holder takes an extra action round, and so does the US on
        # the turn North Sea Oil is played. Both are per-side, not per-turn.
        rounds = {
            side: spacerace.action_rounds_for(
                state.space_race, side, state.action_rounds_this_turn
            )
            for side in Side
        }
        if state.effect_data("North Sea Oil", "extra_action_round_turn") == state.turn:
            rounds[Side.USA] += 1
        for round_number in range(1, max(rounds.values()) + 1):
            state.action_round = round_number
            state.phase = Phase.ACTION_ROUND
            for side in (Side.USSR, Side.USA):
                if state.is_over:
                    return
                if round_number > rounds[side]:
                    continue
                defcon_before = state.defcon
                yield from self._action_round(side)
                if state.is_over:
                    return
                # NORAD triggers on DEFCON *arriving* at 2, once per round.
                from .events.special import norad_end_of_action_round

                yield from norad_end_of_action_round(
                    self, defcon_before > 2 and state.defcon == 2
                )

        yield from self._turn_end()

    def _add_stage_cards(self) -> None:
        """Shuffle in any stage whose cards enter the deck on this turn.

        On turns 4 and 8 the new cards join the *existing* draw deck without the
        discard pile being folded back in (rule 4.4).
        """
        for stage, start in STAGE_START_TURN.items():
            if start != self.state.turn or stage in self._stages_added:
                continue
            self._stages_added.add(stage)
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

        # Man in Earth Orbit: its holder chooses last, after the opponent has revealed.
        order = list(Side)
        ability = spacerace.OPPONENT_HEADLINE_FIRST
        if spacerace.has_ability(state.space_race, Side.USA, ability):
            order = [Side.USSR, Side.USA]
        elif spacerace.has_ability(state.space_race, Side.USSR, ability):
            order = [Side.USA, Side.USSR]

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

        for side in sorted(Side, key=sort_key):
            name = picks[side]
            if name is None or state.is_over:
                continue
            state.headline[int(side)] = None

            if side in cancelled:
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

        yield from self._offer_cuban_missile_crisis_cancel(side)
        if state.is_over:
            return

        if not state.playable_hand(side):
            state.note("no cards to play; action round skipped", side)
            return

        # Quagmire and Bear Trap force discards instead of a normal action round.
        trap = "Quagmire" if side is Side.USA else "Bear Trap"
        if state.in_play(trap, owner=side):
            yield from self._resolve_trap(side, trap)
            return

        yield from self._take_action_round(side)

    def _offer_cuban_missile_crisis_cancel(self, side: Side) -> Iterator[Decision]:
        """Let the threatened player buy the crisis off before acting.

        The USSR removes 2 influence from Cuba; the US removes 2 from West Germany or
        Turkey. Offered at the start of their action round, which is the only moment the
        choice can matter -- after this they either coup and lose, or they do not.
        """
        state = self.state
        effect = state.effects.get("Cuban Missile Crisis")
        if effect is None or effect.owner is None or effect.owner is side:
            return

        sources = ["Cuba"] if side is Side.USSR else ["West Germany", "Turkey"]
        if sum(state.inf(side, n) for n in sources) < 2:
            return

        where = " or ".join(sources)
        cancel = yield from self.confirm(
            side,
            f"Cuban Missile Crisis is in effect: couping would lose you the game. "
            f"Remove 2 of your influence from {where} to cancel it?",
            card="Cuban Missile Crisis",
        )
        if not cancel:
            return

        yield from self.remove_influence(
            side,
            2,
            allowed=sources,
            chooser=side,
            prompt=f"Remove 2 influence from {where} to cancel the Cuban Missile Crisis",
        )
        state.remove_effect("Cuban Missile Crisis")
        state.note("Cuban Missile Crisis cancelled", side)

    def _take_action_round(self, side: Side) -> Iterator[Decision]:
        state = self.state
        hand = state.playable_hand(side)
        chosen = yield from self.choose_card(
            side,
            hand,
            "Choose a card to play",
            dtype=DecisionType.PLAY_CARD,
            labels={n: self._card_label(n, side) for n in hand},
        )
        assert chosen is not None
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

    def play_card(self, name: str, side: Side) -> Iterator[Decision]:
        """Play *name* from *side*'s hand, resolving the chosen use.

        ``state.playing_card`` is saved and restored around the whole play, so that a
        card which causes another to be played -- Grain Sales to Soviets, Missile Envy,
        Star Wars -- reports the inner card while it resolves and the outer one after.
        """
        state = self.state
        c = CARDS[name]

        if name == CHINA_CARD:
            pass  # handled after resolution: it passes to the opponent
        elif name in state.hand(side):
            state.hand(side).remove(name)

        previous, state.playing_card = state.playing_card, name
        try:
            if c.is_scoring:
                state.note(f"plays {name} (scoring)", side)
                yield from self._resolve_event(name, side)
                state.resolve_card(name, as_event=True)
                return

            uses = yield from self._choose_use(name, side)
            yield from self._perform_use(name, side, uses)
        finally:
            state.playing_card = previous

    def _choose_use(self, name: str, side: Side) -> Iterator[Decision]:
        """Ask how the card is being used, returning an :class:`OpsUse`."""
        state = self.state
        c = CARDS[name]
        options: list[Action] = []

        own_event = c.event_belongs_to(side)
        if own_event and self._event_is_playable(name, side):
            options.append(use_action(OpsUse.EVENT, "resolve the event"))

        ops = self.effective_ops(name, side)
        bonus = " (+1 more if every point goes to Asia)" if name == CHINA_CARD else ""
        if ops > 0:
            options.append(use_action(OpsUse.INFLUENCE, f"place {ops} influence{bonus}"))
            if any(rules.can_coup(state, side, n) for n in COUNTRY_ORDER):
                options.append(use_action(OpsUse.COUP, f"coup with {ops} ops{bonus}"))
            if any(rules.can_realign(state, side, n) for n in COUNTRY_ORDER):
                options.append(
                    use_action(OpsUse.REALIGN, f"{ops} realignment rolls{bonus}")
                )

        if self._can_space(name, side):
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
            # The China Card is worth one more when every point is spent in Asia.
            bonus_region=Region.ASIA if name == CHINA_CARD else None,
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
        if state.effect_data(name, "do_not_discard"):
            return
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

    def conduct_operations(
        self,
        side: Side,
        ops: int,
        use: OpsUse,
        *,
        card_name: str | None = None,
        free: bool = False,
        bonus_region: Region | None = None,
    ) -> Iterator[Decision]:
        """Spend *ops* operations points as influence, a coup, or realignments.

        *bonus_region* implements the China Card's "+1 Operations value when all points
        are used in Asia". It cannot be folded into :meth:`effective_ops`, which runs
        before the player has chosen any target, so the bonus is granted at the point
        where it becomes knowable and the extra point is confined to the same region.
        """
        del card_name  # kept for future card-specific ops modifiers
        if use is OpsUse.INFLUENCE:
            yield from self.spend_operations_influence(side, ops, bonus_region=bonus_region)
        elif use is OpsUse.COUP:
            yield from self._coup_operation(side, ops, free=free, bonus_region=bonus_region)
        elif use is OpsUse.REALIGN:
            yield from self._realign_operation(side, ops, bonus_region=bonus_region)
        else:
            raise ValueError(f"{use} is not an operations use")

    @staticmethod
    def _all_in(names: Iterable[str], region: Region) -> bool:
        names = list(names)
        return bool(names) and all(region in COUNTRIES[n].regions for n in names)

    def free_operations(
        self,
        side: Side,
        ops: int,
        *,
        uses: Sequence[OpsUse] = (OpsUse.INFLUENCE, OpsUse.COUP, OpsUse.REALIGN),
        prompt: str = "",
    ) -> Iterator[Decision]:
        """Let *side* conduct operations as if they had played an *ops*-value card.

        Used by the many events worded "conduct operations as if they played an N Op
        card". A coup taken this way is free: it does not count toward the required
        military operations.
        """
        state = self.state
        available: list[Action] = []
        if OpsUse.INFLUENCE in uses and rules.placeable_countries(state, side):
            available.append(use_action(OpsUse.INFLUENCE, f"place {ops} influence"))
        if OpsUse.COUP in uses and any(rules.can_coup(state, side, n) for n in COUNTRY_ORDER):
            available.append(use_action(OpsUse.COUP, f"coup with {ops} ops"))
        if OpsUse.REALIGN in uses and any(
            rules.can_realign(state, side, n) for n in COUNTRY_ORDER
        ):
            available.append(use_action(OpsUse.REALIGN, f"{ops} realignment rolls"))

        if not available:
            state.note(f"{ops} free operations point(s) wasted: nothing legal", side)
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
                    {"ops": ops, "free": True},
                )
            )
            use = OpsUse(action.value)

        yield from self.conduct_operations(side, ops, use, free=True)

    def spend_operations_influence(
        self, side: Side, ops: int, *, bonus_region: Region | None = None
    ) -> Iterator[Decision]:
        """Place influence with operations points.

        Costs 2 per point into an enemy-controlled country, re-evaluated after every
        single point so that breaking control mid-placement makes the rest cheaper.
        Legal targets are frozen at the start of the placement, so newly placed
        influence cannot be chained outward to reach further countries.

        If *bonus_region* is set and every point went into that region, one extra point
        is granted, which must also go there.
        """
        state = self.state
        budget = ops
        placed: dict[str, int] = {}
        reachable = rules.reachable_countries(state, side)
        bonus_granted = False

        while True:
            if budget <= 0:
                if (
                    bonus_region is not None
                    and not bonus_granted
                    and self._all_in(placed, bonus_region)
                ):
                    bonus_granted = True
                    budget = 1
                    state.note(
                        f"all operations points used in {bonus_region}: +1 bonus point",
                        side,
                    )
                else:
                    break

            candidates = rules.placeable_countries(
                state, side, budget=budget, reachable=reachable
            )
            if bonus_granted:
                # The bonus point has to stay in the region that earned it.
                candidates = [
                    n for n in candidates if bonus_region in COUNTRIES[n].regions
                ]
            if not candidates:
                if not bonus_granted:
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

        yuri_and_samantha_on_coup(self, side)
        result = rules.resolve_coup(self.state, side, name, ops, free=free)
        self._check_cuban_missile_crisis(side)
        self._check_defcon_loss(side)
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

    def _coup_operation(
        self,
        side: Side,
        ops: int,
        *,
        free: bool = False,
        bonus_region: Region | None = None,
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
        if bonus_region is not None and bonus_region in COUNTRIES[name].regions:
            ops += 1
            state.note(f"all operations points used in {bonus_region}: coup at {ops} ops", side)
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
        self, side: Side, ops: int, *, bonus_region: Region | None = None
    ) -> Iterator[Decision]:
        state = self.state
        chosen: list[str] = []
        remaining = ops
        bonus_granted = False

        while True:
            if remaining <= 0:
                if (
                    bonus_region is not None
                    and not bonus_granted
                    and self._all_in(chosen, bonus_region)
                ):
                    bonus_granted = True
                    remaining = 1
                    state.note(
                        f"all realignments made in {bonus_region}: +1 bonus roll", side
                    )
                else:
                    return

            targets = [n for n in COUNTRY_ORDER if rules.can_realign(state, side, n)]
            if bonus_granted:
                targets = [n for n in targets if bonus_region in COUNTRIES[n].regions]
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

        try:
            if handler is not None:
                ctx = EventContext(
                    card=name,
                    player=side,
                    ops=CARDS[name].ops,
                    triggered_by_opponent=triggered_by_opponent,
                )
                yield from handler(self, ctx)
        finally:
            state.event_player = previous
            if held:
                state.release(name)

        if from_headline:
            # A headline card that did not put itself into play is discarded now.
            self._retire_card(name, side, as_event=True)

        self._check_victory_points()

    # -- traps -------------------------------------------------------------- #

    def _resolve_trap(self, side: Side, trap: str) -> Iterator[Decision]:
        """Quagmire / Bear Trap: discard a 2+ ops card, or all scoring-free options."""
        state = self.state
        hand = [n for n in state.hand(side) if CARDS[n].ops >= 2]

        if not hand:
            playable = [n for n in state.hand(side) if not CARDS[n].is_scoring]
            if not playable:
                state.note(f"{trap}: no discardable card; action round lost", side)
                return
            state.note(f"{trap}: no 2+ ops card, may play normally", side)
            state.remove_effect(trap)
            yield from self._take_action_round(side)
            return

        chosen = yield from self.choose_card(
            side,
            hand,
            f"{trap}: discard a card with 2 or more operations points",
            dtype=DecisionType.DISCARD_CARD,
            labels={n: self._card_label(n, side) for n in hand},
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

        # Holding a scoring card at the end of a turn loses the game.
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

        self._military_operations_check()
        if state.is_over:
            return

        # Effects that last only for the turn expire now.
        for name, effect in list(state.effects.items()):
            if effect.expires == "end_of_turn":
                state.remove_effect(name)
                state.note(f"{name} leaves play")
            elif effect.expires == "end_of_next_turn":
                effect.expires = "end_of_turn"

        if state.turn == FINAL_TURN:
            # The China Card is worth a point to whoever holds it at the very end.
            state.award_vp(state.china_card_owner, 1)
            state.note(f"{state.china_card_owner.label} holds the China Card: +1 VP")

        state.china_card_face_up = True
        state.change_defcon(+1)
        state.military_ops = [0, 0]
        state.note(f"turn {state.turn} ends; DEFCON now {state.defcon}")

        self._check_victory_points()
        return
        yield  # pragma: no cover - uniform generator interface

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
        if state.is_over:
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
        state.note("--- final scoring ---")

        europe_winner: Side | None = None
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
                europe_winner = auto

        if europe_winner is not None:
            state.finish(europe_winner, WinReason.EUROPE_CONTROL)
        elif state.vp > 0:
            state.finish(Side.USSR, WinReason.FINAL_SCORING)
        elif state.vp < 0:
            state.finish(Side.USA, WinReason.FINAL_SCORING)
        else:
            state.finish(None, WinReason.DRAW)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def score_region(self, region: Region) -> None:
        """Resolve a scoring card, handling the Europe automatic victory.

        Shuttle Diplomacy is spent by the first Asia or Middle East scoring it affects,
        so it comes off the table here rather than expiring on a timer.
        """
        state = self.state
        _, _, auto = rules.score_region(state, region)

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


__all__ = ["ACTION_ROUNDS", "EventContext", "EventHandler", "FINAL_TURN", "Game", "HAND_SIZE"]

"""The mutable game state.

:class:`GameState` is the single source of truth. Everything an agent ever sees is a
pure function of it (see :mod:`twilight.observe`), which is what keeps the numeric
view used for reinforcement learning and the text view used by language-model agents
from drifting apart.

The state deliberately holds *hidden* information too -- both hands, and the draw pile
order. Information hiding happens once, in the observation layer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable

from .data import CARDS, CHINA_CARD, COUNTRY_ORDER, COUNTRY_SLOT, NUM_COUNTRIES, card
from .enums import Phase, Side, WinReason

#: Influence needed in a country beyond the opponent's to control it equals the
#: country's stability value.
MAX_DEFCON = 5
MIN_DEFCON = 1

#: The VP track runs from 20 (USSR win) to -20 (US win) and is stored USSR-positive.
VP_LIMIT = 20


@dataclass(slots=True)
class ActiveEffect:
    """A card in play, or a lingering rules modification.

    ``expires`` is checked by the engine's turn loop:

    ``permanent``       stays until some card explicitly removes it
    ``end_of_turn``     discarded during the turn-end phase
    ``end_of_next_turn`` survives one full turn (used by a few cards)
    """

    card: str
    owner: Side | None = None
    expires: str = "permanent"
    #: Free-form per-effect payload, e.g. the region chosen for Chernobyl.
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DeferredTrigger:
    """Something a card scheduled to happen at a later action round.

    Several cards say "on your opponent's next action round ...". Resolving them needs a
    hook that fires between action rounds rather than inside an event, which is what
    :meth:`twilight.engine.Game._fire_deferred` provides.

    ``not_before`` is a value of :attr:`GameState.ar_sequence`, the monotonic count of
    action rounds taken by either player. Comparing against it is what makes "next"
    mean the next one rather than the one currently in progress.
    """

    card: str
    #: Whose action round the trigger waits for.
    player: Side
    #: ``"start"`` or ``"end"`` of that action round.
    when: str
    #: Key into the deferred-handler registry in :mod:`twilight.events`.
    kind: str
    not_before: int
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LogEntry:
    turn: int
    action_round: int
    player: Side | None
    text: str

    def __str__(self) -> str:
        who = self.player.label if self.player is not None else "--"
        return f"T{self.turn}/AR{self.action_round} {who:<4} {self.text}"


@dataclass
class GameState:
    """Complete game state, hidden information included."""

    # -- board ------------------------------------------------------------- #
    #: ``influence[side][slot]`` for each placeable country, indexed by
    #: :data:`twilight.data.COUNTRY_SLOT`.
    influence: list[list[int]] = field(
        default_factory=lambda: [[0] * NUM_COUNTRIES, [0] * NUM_COUNTRIES]
    )

    # -- tracks ------------------------------------------------------------ #
    #: Victory points, USSR-positive: +20 is a USSR win, -20 a US win.
    vp: int = 0
    defcon: int = MAX_DEFCON
    turn: int = 1
    action_round: int = 0
    #: Space race box each player has reached, 0 through 8.
    space_race: list[int] = field(default_factory=lambda: [0, 0])
    #: Space race attempts already made this turn.
    space_attempts: list[int] = field(default_factory=lambda: [0, 0])
    military_ops: list[int] = field(default_factory=lambda: [0, 0])

    # -- cards ------------------------------------------------------------- #
    hands: list[list[str]] = field(default_factory=lambda: [[], []])
    deck: list[str] = field(default_factory=list)
    discard: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    china_card_owner: Side = Side.USSR
    #: A face-down China Card cannot be played; it flips up at the end of the turn in
    #: which it changed hands.
    china_card_face_up: bool = True

    # -- effects ----------------------------------------------------------- #
    effects: dict[str, ActiveEffect] = field(default_factory=dict)
    #: Cards whose event has already resolved at some point this game. Several cards
    #: are playable only if another has (or has not) been played.
    events_resolved: set[str] = field(default_factory=set)

    # -- flow -------------------------------------------------------------- #
    phase: Phase = Phase.SETUP
    #: Whose action round it is.
    player: Side = Side.USSR
    #: The player responsible for what happens now. Normally the side taking its action
    #: round; during the headline phase it is the player who *played* the card, even if
    #: the opponent implements the event. This is who loses if DEFCON reaches 1, which
    #: is not necessarily the side that owns the event doing the degrading.
    phasing_player: Side | None = None
    #: Action rounds in the current turn, which grows from 6 to 7 in the Mid War.
    action_rounds_this_turn: int = 6
    headline: dict[int, str | None] = field(default_factory=lambda: {0: None, 1: None})
    #: Set while resolving an opponent's event through UN Intervention and similar,
    #: so that "the player" inside an event means the right side.
    event_player: Side | None = None
    #: The card currently being played, from the moment it leaves its owner's hand
    #: until it reaches the discard, the removed pile, or the table. Without this the
    #: card would exist only in an engine local, so an agent choosing where to spend
    #: operations points could not see which card it was spending.
    playing_card: str | None = None
    #: Cards an event is holding outside every pile while it asks a question -- a card
    #: taken by Missile Envy or Grain Sales, or the five Our Man In Tehran turns over.
    #: Tracked so no card is ever unaccounted for mid-decision.
    transit: set[str] = field(default_factory=set)
    #: Monotonic count of action rounds taken, by either player. Deferred triggers
    #: compare against it so "next action round" excludes the current one.
    ar_sequence: int = 0
    #: Effects waiting for a later action round.
    deferred: list[DeferredTrigger] = field(default_factory=list)
    #: A card a player is compelled to play for operations on their next action round,
    #: per side. Set by Missile Envy.
    must_play: dict[int, str | None] = field(default_factory=lambda: {0: None, 1: None})

    # -- outcome ----------------------------------------------------------- #
    winner: Side | None = None
    win_reason: WinReason | None = None

    # -- bookkeeping ------------------------------------------------------- #
    rng: random.Random = field(default_factory=random.Random)
    log: list[LogEntry] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Influence
    # ------------------------------------------------------------------ #

    def inf(self, side: Side, name: str) -> int:
        return self.influence[side][COUNTRY_SLOT[name]]

    def set_inf(self, side: Side, name: str, value: int) -> None:
        if value < 0:
            raise ValueError(f"negative influence for {side.label} in {name}: {value}")
        self.influence[side][COUNTRY_SLOT[name]] = value

    def add_inf(self, side: Side, name: str, amount: int = 1) -> None:
        slot = COUNTRY_SLOT[name]
        self.influence[side][slot] = max(0, self.influence[side][slot] + amount)

    def remove_inf(self, side: Side, name: str, amount: int = 1) -> int:
        """Remove up to *amount* influence, returning how much was actually removed."""
        slot = COUNTRY_SLOT[name]
        removed = min(amount, self.influence[side][slot])
        self.influence[side][slot] -= removed
        return removed

    def clear_inf(self, side: Side, name: str) -> int:
        removed = self.influence[side][COUNTRY_SLOT[name]]
        self.influence[side][COUNTRY_SLOT[name]] = 0
        return removed

    def countries_with_influence(self, side: Side) -> list[str]:
        row = self.influence[side]
        return [name for name in COUNTRY_ORDER if row[COUNTRY_SLOT[name]] > 0]

    # ------------------------------------------------------------------ #
    # Tracks
    # ------------------------------------------------------------------ #

    def award_vp(self, side: Side, amount: int) -> None:
        """Give *amount* VP to *side*, clamped to the ends of the track."""
        if amount == 0:
            return
        delta = amount if side is Side.USSR else -amount
        self.vp = max(-VP_LIMIT, min(VP_LIMIT, self.vp + delta))

    def vp_for(self, side: Side) -> int:
        """The VP track read from *side*'s point of view."""
        return self.vp if side is Side.USSR else -self.vp

    def change_defcon(self, delta: int) -> None:
        self.defcon = max(MIN_DEFCON, min(MAX_DEFCON, self.defcon + delta))

    # ------------------------------------------------------------------ #
    # Effects
    # ------------------------------------------------------------------ #

    def in_play(self, card_name: str, owner: Side | None = None) -> bool:
        effect = self.effects.get(card_name)
        if effect is None:
            return False
        return owner is None or effect.owner is owner

    def add_effect(
        self,
        card_name: str,
        owner: Side | None = None,
        expires: str = "permanent",
        **data: Any,
    ) -> ActiveEffect:
        effect = ActiveEffect(card=card_name, owner=owner, expires=expires, data=dict(data))
        self.effects[card_name] = effect
        return effect

    def remove_effect(self, card_name: str, *, retire: bool = True) -> ActiveEffect | None:
        """Take an effect off the table and retire its card.

        While a card is in play the engine deliberately keeps it out of the discard and
        removed piles, so taking the effect off is the moment the physical card is
        finally resolved. Skipping that is how cards used to disappear from the game
        entirely, quietly shrinking the deck.

        Cards that are already somewhere countable -- most obviously one handed to the
        opponent while its effect lingers -- are left alone.
        """
        effect = self.effects.pop(card_name, None)
        if effect is None or not retire:
            return effect
        if card_name in CARDS and not self._is_accounted(card_name):
            self.resolve_card(card_name, as_event=True)
        return effect

    def _is_accounted(self, card_name: str) -> bool:
        """Whether *card_name* already sits in a hand or one of the piles."""
        return (
            card_name in self.hands[Side.USSR]
            or card_name in self.hands[Side.USA]
            or card_name in self.deck
            or card_name in self.discard
            or card_name in self.removed
        )

    def hold(self, *names: str) -> None:
        """Mark cards as in transit: out of every pile, held by a resolving event."""
        self.transit.update(names)

    def release(self, *names: str) -> None:
        """Stop tracking cards as in transit, once they have reached a pile or hand."""
        self.transit.difference_update(names)

    def effect_data(self, card_name: str, key: str, default: Any = None) -> Any:
        effect = self.effects.get(card_name)
        return default if effect is None else effect.data.get(key, default)

    def has_resolved(self, card_name: str) -> bool:
        """Whether *card_name*'s event has ever fired this game."""
        return card_name in self.events_resolved

    # ------------------------------------------------------------------ #
    # Deferred triggers
    # ------------------------------------------------------------------ #

    def defer(
        self,
        card: str,
        kind: str,
        *,
        player: Side,
        when: str = "end",
        **data: Any,
    ) -> DeferredTrigger:
        """Schedule *kind* for *player*'s next action round.

        ``not_before`` is pinned to the current action round, so a trigger created
        during a player's own action round waits for their *following* one.
        """
        if when not in ("start", "end"):
            raise ValueError(f"deferred trigger must fire at 'start' or 'end', not {when!r}")
        trigger = DeferredTrigger(
            card=card,
            player=player,
            when=when,
            kind=kind,
            not_before=self.ar_sequence,
            data=dict(data),
        )
        self.deferred.append(trigger)
        return trigger

    def cancel_deferred(self, *, card: str | None = None, kind: str | None = None) -> int:
        """Drop pending triggers matching *card* and/or *kind*. Returns how many went.

        At least one filter is required, so an accidental bare call cannot silently
        wipe every pending trigger.
        """
        if card is None and kind is None:
            raise ValueError("cancel_deferred needs a card or a kind to match on")
        before = len(self.deferred)
        self.deferred = [
            t for t in self.deferred if not self._deferred_matches(t, card, kind)
        ]
        return before - len(self.deferred)

    def has_deferred(self, *, card: str | None = None, kind: str | None = None) -> bool:
        return any(self._deferred_matches(t, card, kind) for t in self.deferred)

    @staticmethod
    def _deferred_matches(
        trigger: DeferredTrigger, card: str | None, kind: str | None
    ) -> bool:
        return (card is None or trigger.card == card) and (
            kind is None or trigger.kind == kind
        )

    # ------------------------------------------------------------------ #
    # Cards
    # ------------------------------------------------------------------ #

    def hand(self, side: Side) -> list[str]:
        return self.hands[side]

    def holds_china_card(self, side: Side) -> bool:
        return self.china_card_owner is side

    def playable_hand(self, side: Side) -> list[str]:
        """Cards *side* may play this action round, China Card included when usable."""
        cards = list(self.hands[side])
        if self.china_card_owner is side and self.china_card_face_up:
            cards.append(CHINA_CARD)
        return cards

    def draw_card(self) -> str | None:
        """Take the top card, reshuffling the discard pile in if the deck runs out."""
        if not self.deck:
            if not self.discard:
                return None
            self.deck = list(self.discard)
            self.discard.clear()
            self.rng.shuffle(self.deck)
            self.note("draw pile exhausted; discard pile reshuffled")
        return self.deck.pop()

    def shuffle_into_deck(self, names: Iterable[str]) -> None:
        self.deck.extend(names)
        self.rng.shuffle(self.deck)

    def discard_card(self, name: str) -> None:
        if name == CHINA_CARD:
            raise ValueError("the China Card is never discarded; it passes to the opponent")
        self.discard.append(name)

    def resolve_card(self, name: str, *, as_event: bool) -> None:
        """Send a played card to the discard or removed pile.

        Cards marked "remove after use" in the data leave the game only when played
        for their event; used purely for operations they are discarded as normal.
        """
        if name == CHINA_CARD:
            return
        if as_event:
            self.events_resolved.add(name)
        if as_event and card(name).remove_on_event:
            self.removed.append(name)
        else:
            self.discard.append(name)

    def cards_in_play_or_gone(self) -> set[str]:
        """Cards that are neither in a hand nor available to be drawn."""
        return set(self.removed) | set(self.effects)

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def note(self, text: str, player: Side | None = None) -> None:
        self.log.append(LogEntry(self.turn, self.action_round, player, text))

    # ------------------------------------------------------------------ #
    # Outcome
    # ------------------------------------------------------------------ #

    @property
    def is_over(self) -> bool:
        return self.phase is Phase.GAME_OVER

    def finish(self, winner: Side | None, reason: WinReason) -> None:
        self.phase = Phase.GAME_OVER
        self.winner = winner
        self.win_reason = reason
        who = "draw" if winner is None else f"{winner.label} wins"
        self.note(f"game over: {who} ({reason.value})")

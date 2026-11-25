"""Scaffolding for the card-by-card behaviour tests.

These tests are a second, implementation-independent audit of the engine: every
expectation is taken from the printed card text or the rulebook (quoted in each
test's docstring), never from what the code currently does.

The :class:`Table` helper builds an arbitrary position on a real :class:`Game` --
influence, DEFCON, VP, hands, effects in play, turn and action round -- and then
drives one engine entry point (a whole action round, the headline phase, the turn
end, the rest of the game ...) while the test answers each decision by its
canonical action key::

    t = Table(turn=2, defcon=4)
    t.inf(usa={"West Germany": 3})
    t.play(Side.USSR, "Blockade")       # card:Blockade, then use:event
    t.answer("pass")                     # the US declines to discard
    assert t.s.inf(Side.USA, "West Germany") == 0

Dice are scripted with :meth:`Table.dice`; when the queue is empty the seeded RNG
takes over again, so a test only scripts the rolls it cares about.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Iterator

import pytest

from twilight import Game
from twilight.data import CARDS, CHINA_CARD, COUNTRY_ORDER, NUM_COUNTRIES
from twilight.decisions import Decision, DecisionType
from twilight.enums import Phase, Side, Stage

USSR = Side.USSR
USA = Side.USA


class ScriptedRandom(random.Random):
    """A seeded RNG whose die rolls and random picks can be queued by a test."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.dice: list[int] = []
        self.picks: list[Any] = []
        self.rolls_made: list[int] = []

    def randint(self, a: int, b: int) -> int:  # noqa: D401 - RNG API
        if self.dice:
            value = self.dice.pop(0)
            assert a <= value <= b, f"scripted die {value} outside {a}..{b}"
        else:
            value = super().randint(a, b)
        self.rolls_made.append(value)
        return value

    def choice(self, seq):  # noqa: D401 - RNG API
        if self.picks:
            want = self.picks.pop(0)
            assert want in seq, f"scripted pick {want!r} not among {list(seq)!r}"
            return want
        return super().choice(seq)


def _empty_gen() -> Iterator[Decision]:
    return
    yield  # pragma: no cover


class Table:
    """A real Game whose state has been wiped and rebuilt to order."""

    def __init__(
        self,
        *,
        turn: int = 1,
        defcon: int = 5,
        vp: int = 0,
        action_round: int = 1,
        optional: bool = True,
        seed: int = 0,
        china: Side | None = USSR,
        china_face_up: bool = True,
    ) -> None:
        g = Game(seed=seed, optional_cards=optional, strict_events=True)
        s = g.state
        s.influence = [[0] * NUM_COUNTRIES, [0] * NUM_COUNTRIES]
        s.hands = [[], []]
        s.deck = []
        s.discard = []
        s.removed = []
        s.effects = {}
        s.events_resolved = set()
        s.deferred = []
        s.must_play = {0: None, 1: None}
        s.transit = set()
        s.headline = {0: None, 1: None}
        s.military_ops = [0, 0]
        s.space_race = [0, 0]
        s.space_attempts = [0, 0]
        # Every stage already shuffled in, so no turn loop adds cards behind our back.
        s.stages_in_deck = set(Stage)
        s.turn = turn
        s.defcon = defcon
        s.vp = vp
        s.phase = Phase.ACTION_ROUND
        s.action_round = action_round
        s.action_rounds_this_turn = 6 if turn < 4 else 7
        s.player = USSR
        s.phasing_player = None
        s.event_player = None
        s.playing_card = None
        s.winner = None
        s.win_reason = None
        s.rng = ScriptedRandom(seed)
        s.china_card_owner = china if china is not None else USSR
        s.china_card_face_up = china_face_up
        s.log = []
        g.history = []
        self.g = g
        self.s = s

    # ------------------------------------------------------------------ #
    # Building the position
    # ------------------------------------------------------------------ #

    def inf(self, ussr: dict[str, int] | None = None, usa: dict[str, int] | None = None) -> "Table":
        for side, placements in ((USSR, ussr), (USA, usa)):
            for name, amount in (placements or {}).items():
                self.s.set_inf(side, name, amount)
        return self

    def hand(self, side: Side, *cards: str) -> "Table":
        for c in cards:
            assert c in CARDS, c
            assert c != CHINA_CARD, "use china= for the China Card"
            self.s.hands[side].append(c)
        return self

    def deck(self, *cards: str) -> "Table":
        """Set the draw pile; the *first* card named is the next one drawn."""
        self.s.deck = list(reversed(cards))
        return self

    def discard(self, *cards: str) -> "Table":
        self.s.discard.extend(cards)
        return self

    def effect(self, name: str, owner: Side | None, expires: str = "permanent", **data) -> "Table":
        self.s.add_effect(name, owner=owner, expires=expires, **data)
        self.s.events_resolved.add(name)
        return self

    def resolved(self, *names: str) -> "Table":
        """Mark events as having been played earlier in the game (and gone)."""
        for n in names:
            self.s.events_resolved.add(n)
            if CARDS[n].remove_on_event and n not in self.s.removed:
                self.s.removed.append(n)
        return self

    def filler(self, side: Side, n: int, *, exclude: Iterable[str] = ()) -> list[str]:
        """Give *side* n harmless cards not used elsewhere in the position."""
        used = set(self.s.hands[0]) | set(self.s.hands[1]) | set(self.s.deck) | set(
            self.s.discard
        ) | set(self.s.removed) | set(self.s.effects) | set(exclude)
        pool = [c for c in FILLER if c not in used]
        picked = pool[:n]
        assert len(picked) == n, "ran out of filler cards"
        self.s.hands[side].extend(picked)
        return picked

    # ------------------------------------------------------------------ #
    # Driving
    # ------------------------------------------------------------------ #

    def run(self, gen: Iterator[Decision]) -> "Table":
        self.g._driver = gen
        self.g.decision = self.g._advance(None)
        return self

    def action_round(self, side: Side) -> "Table":
        self.s.phase = Phase.ACTION_ROUND
        return self.run(self.g._action_round(side))

    def headline_phase(self) -> "Table":
        return self.run(self.g._headline_phase())

    def turn_end(self) -> "Table":
        return self.run(self.g._turn_end())

    def play_turn(self) -> "Table":
        return self.run(self.g._play_turn())

    def rest_of_game(self) -> "Table":
        """Run the real game loop from the current turn (skipping setup)."""
        self.g._setup = _empty_gen  # type: ignore[method-assign]
        return self.run(self.g._run())

    def final_scoring(self) -> "Table":
        def gen():
            self.g._final_scoring()
            return
            yield  # pragma: no cover
        return self.run(gen())

    def event(self, side: Side, card: str, *, from_headline: bool = False) -> "Table":
        """Resolve *card*'s event for *side* directly (not from a hand)."""
        self.s.phasing_player = side
        self.s.player = side
        return self.run(self.g._resolve_event(card, side, from_headline=from_headline))

    def play(self, side: Side, card: str, use: str | None = "event", *answers: str) -> "Table":
        """Start *side*'s action round, pick *card*, choose *use*, then answer on.

        *use* is the ``use:`` suffix (``event``, ``influence``, ``coup``, ``realign``,
        ``space``) or ``None`` to stop at the use decision.
        """
        if card == CHINA_CARD:
            self.s.china_card_owner = side
            self.s.china_card_face_up = True
        elif card not in self.s.hands[side]:
            self.hand(side, card)
        self.action_round(side)
        if self.d is not None and self.d.type is DecisionType.PLAY_CARD:
            self.answer(f"card:{card}")
        if use is not None and self.d is not None and self.d.type is DecisionType.CARD_USE:
            self.answer(f"use:{use}")
        return self.answer(*answers)

    @property
    def d(self) -> Decision | None:
        return self.g.decision

    @property
    def done(self) -> bool:
        return self.g.decision is None

    def keys(self) -> tuple[str, ...]:
        assert self.d is not None, "no decision pending"
        return self.d.legal_keys

    def answer(self, *keys: str) -> "Table":
        for key in keys:
            assert self.d is not None, f"wanted to answer {key!r} but nothing is being asked"
            self.g.step(key)
        return self

    def finish_with(self, policy=None, limit: int = 500) -> "Table":
        """Answer every remaining decision with *policy* (default: pass, else first)."""
        for _ in range(limit):
            if self.d is None:
                return self
            if policy is not None:
                choice = policy(self.d)
            elif "pass" in self.d.legal_keys:
                choice = "pass"
            else:
                choice = self.d.legal_keys[0]
            self.g.step(choice)
        raise AssertionError("decision loop did not end")

    def dice(self, *values: int) -> "Table":
        self.s.rng.dice.extend(values)
        return self

    def picks(self, *values) -> "Table":
        self.s.rng.picks.extend(values)
        return self

    # ------------------------------------------------------------------ #
    # Reading the position
    # ------------------------------------------------------------------ #

    def where(self, card: str) -> str:
        s = self.s
        places = []
        if card in s.hands[USSR]:
            places.append("ussr_hand")
        if card in s.hands[USA]:
            places.append("usa_hand")
        if card in s.deck:
            places.append("deck")
        if card in s.discard:
            places.append("discard")
        if card in s.removed:
            places.append("removed")
        if card in s.effects:
            places.append("in_play")
        if card in s.transit:
            places.append("transit")
        return "+".join(places) or "nowhere"

    def gone_for_good(self, card: str) -> bool:
        """A starred card after its event: out of the game (removed, or on the table
        as a permanent effect), and never back in the draw cycle."""
        w = self.where(card)
        return w in ("removed", "in_play")

    def in_cycle(self, card: str) -> bool:
        """The card went to the discard pile (will be reshuffled)."""
        return self.where(card) == "discard"


#: Cards whose events are irrelevant to most tests and that are never the subject.
FILLER = [
    "Duck and Cover",
    "Red Scare/Purge",
    "Olympic Games",
    "Decolonization",
    "Nuclear Test Ban",
    "Indo-Pakistani War",
    "Arab-Israeli War",
    "Socialist Governments",
    "Five-Year Plan",
    "Brush War",
    "Arms Race",
    "Summit",
    "Junta",
    "Missile Envy",
    "South African Unrest",
    "Muslim Revolution",
    "ABM Treaty",
    "OPEC",
    "Colonial Rear Guards",
    "Grain Sales to Soviets",
    "Latin American Death Squads",
    "Shuttle Diplomacy",
    "The Voice Of America",
    "Liberation Theology",
    "One Small Step...",
    "Terrorism",
    "Latin American Debt Crisis",
]


@pytest.fixture
def table():
    return Table


def mk(**kw) -> Table:
    return Table(**kw)

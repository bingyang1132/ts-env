"""DEFCON-moving events: direction must match the card's own data.

Nuclear Test Ban once degraded DEFCON instead of improving it. Nothing caught it because
no test pinned the direction and random play rarely fires the event at DEFCON 2. The
second test makes the whole class of mistake impossible to reintroduce: every handler
whose card data names an ImproveDEFCON / DegradeDEFCON effect must call the matching
engine method.
"""

from __future__ import annotations

import inspect

from twilight import Game, Side
from twilight.data import CARDS
from twilight.engine import EventContext
from twilight.events import EVENTS
from twilight.events.early_war import nuclear_test_ban


def _run(gen) -> None:
    try:
        while True:
            next(gen)
    except StopIteration:
        pass


def test_nuclear_test_ban_scores_then_improves_defcon():
    game = Game(seed=1)
    state = game.state
    state.defcon = 3
    before = state.vp

    _run(nuclear_test_ban(game, EventContext(card="Nuclear Test Ban", player=Side.USSR, ops=4)))

    assert state.vp == before + 1, "VP should be DEFCON - 2 = 1, on the USSR-positive track"
    assert state.defcon == 5, "the card improves DEFCON by two; it must never lower it"
    assert state.winner is None


def test_nuclear_test_ban_at_defcon_2_is_safe():
    game = Game(seed=1)
    state = game.state
    state.defcon = 2

    _run(nuclear_test_ban(game, EventContext(card="Nuclear Test Ban", player=Side.USA, ops=4)))

    assert state.defcon == 4
    assert state.winner is None


def test_defcon_direction_matches_card_data():
    mismatches = []
    for name, card in CARDS.items():
        spec = repr(card.effect_spec)
        improves = "ImproveDEFCON" in spec
        degrades = "DegradeDEFCON" in spec
        if not (improves or degrades):
            continue
        source = inspect.getsource(EVENTS[name])
        if improves != ("improve_defcon" in source) or degrades != ("degrade_defcon" in source):
            mismatches.append(name)
    assert not mismatches, f"handler DEFCON direction disagrees with card data: {mismatches}"

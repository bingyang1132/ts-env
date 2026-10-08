"""Realignment spends one operations point per roll.

Rulebook: each operations point buys one realignment roll; the same country may be
targeted again, and the player may stop early. Before this was enforced the engine's
realignment loop never counted rolls down, so a player who never passed could realign
forever (the value-greedy "loops inside an event" games).
"""

from __future__ import annotations

import random

import pytest

from twilight import Game, Side
from twilight.data import COUNTRIES
from twilight.decisions import PASS, DecisionType, use_action
from twilight.enums import OpsUse, Phase, Region


def advance_to_action_round(game: Game, seed: int = 0) -> Game:
    rng = random.Random(seed)
    while game.decision is not None and game.state.phase is not Phase.ACTION_ROUND:
        game.step(rng.choice(game.decision.options))
    return game


def _game_with_targets(seed: int = 0) -> Game:
    """A game in its first action round with plenty of influence to realign against."""
    game = advance_to_action_round(Game(seed=seed), seed=seed)
    for name in ("Chile", "Pakistan", "Japan"):
        game.state.add_inf(Side.USSR, name, 10)
        game.state.add_inf(Side.USA, name, 10)
    return game


def _drive(gen, pick):
    """Run an operations generator to completion; return the decisions it asked."""
    asked = []
    try:
        decision = next(gen)
        while True:
            asked.append(decision)
            decision = gen.send(pick(decision, len(asked)))
    except StopIteration:
        pass
    return asked


def _always(name):
    def pick(decision, _n):
        action = decision.find(f"country:{name}")
        assert action is not None, f"{name} not offered: {decision.prompt}"
        return action

    return pick


@pytest.mark.parametrize("ops", [1, 2, 3, 4])
def test_realignment_asks_exactly_ops_times(ops):
    game = _game_with_targets()
    asked = _drive(game._realign_operation(Side.USSR, ops), _always("Chile"))
    assert len(asked) == ops
    assert all(d.type is DecisionType.REALIGN_TARGET for d in asked)
    assert [d.context["remaining"] for d in asked] == list(range(ops, 0, -1))


def test_four_op_card_through_conduct_operations_stops_after_four_rolls():
    game = _game_with_targets()
    gen = game.conduct_operations(Side.USSR, 4, OpsUse.REALIGN)
    asked = _drive(gen, _always("Chile"))
    assert len(asked) == 4


def test_the_same_country_may_be_realigned_repeatedly():
    game = _game_with_targets()
    asked = _drive(game._realign_operation(Side.USA, 3), _always("Pakistan"))
    assert len(asked) == 3
    assert all(d.find("country:Pakistan") is not None for d in asked)


def test_pass_ends_realignment_early():
    game = _game_with_targets()

    def pick(decision, n):
        assert decision.find(PASS.key) is not None, "pass must always be on offer"
        return PASS if n == 2 else decision.find("country:Chile")

    asked = _drive(game._realign_operation(Side.USSR, 4), pick)
    assert len(asked) == 2


def test_china_card_bonus_roll_when_every_roll_is_in_asia():
    game = _game_with_targets()
    asked = _drive(
        game._realign_operation(Side.USSR, 4, bonus_region=Region.ASIA),
        _always("Pakistan"),
    )
    assert len(asked) == 5
    bonus = asked[-1]
    assert bonus.context["remaining"] == 1
    offered = [a.value for a in bonus.options if a.key != PASS.key]
    assert offered and all(Region.ASIA in COUNTRIES[n].regions for n in offered)


def test_no_china_card_bonus_when_a_roll_leaves_asia():
    game = _game_with_targets()

    def pick(decision, n):
        return decision.find("country:Chile" if n == 1 else "country:Pakistan")

    asked = _drive(game._realign_operation(Side.USSR, 4, bonus_region=Region.ASIA), pick)
    assert len(asked) == 4


@pytest.mark.parametrize("seed", range(12))
def test_a_player_who_never_passes_still_finishes_the_game(seed):
    """Prefer realignment whenever offered and never pass: the game must still end."""
    rng = random.Random(seed)
    game = Game(seed=seed)
    run = longest = 0
    for _ in range(20_000):
        decision = game.decision
        if decision is None:
            break
        if decision.type is DecisionType.REALIGN_TARGET:
            run += 1
            longest = max(longest, run)
            assert decision.context["remaining"] <= 5
            choices = [a for a in decision.options if a.key != PASS.key]
        else:
            run = 0
            realign = decision.find(use_action(OpsUse.REALIGN).key)
            choices = [realign] if realign is not None else list(decision.options)
        game.step(rng.choice(choices))
    assert game.decision is None, "the game did not finish within 20,000 steps"
    # Four ops plus the China Card's Asia bonus is the most one operation can give.
    assert longest <= 5

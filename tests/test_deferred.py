"""Deferred triggers, and the two cards that need them.

Several cards say "on your opponent's next action round ...". These check the timing
primitive itself, then the two real users of it: We Will Bury You's cancellable victory
points and Missile Envy's forced play.
"""

from __future__ import annotations

import random

import pytest

from twilight import Game, Side
from twilight.decisions import ActionKind
from twilight.engine import EventContext
from twilight.enums import OpsUse, Phase
from twilight.events import register_deferred
from twilight.events.mid_war import WE_WILL_BURY_YOU, missile_envy, we_will_bury_you

#: A trigger kind used only by these tests, so nothing real is disturbed.
TEST_KIND = "__test_marker__"


@register_deferred(TEST_KIND)
def _test_marker(game, trigger):
    """Leave a detectable trace that this trigger fired, and for whom."""
    game.state.add_effect("__fired__", owner=trigger.player)
    return
    yield  # pragma: no cover - generator interface only


def run_silent(gen) -> None:
    """Drive a generator that is expected to ask no decisions."""
    try:
        decision = next(gen)
    except StopIteration:
        return
    raise AssertionError(f"unexpected decision: {decision.type} / {decision.prompt}")


def advance_to_action_round(game: Game, seed: int = 0) -> Game:
    rng = random.Random(seed)
    while game.decision is not None and game.state.phase is not Phase.ACTION_ROUND:
        game.step(rng.choice(game.decision.options))
    return game


# --------------------------------------------------------------------------- #
# The timing primitive
# --------------------------------------------------------------------------- #


def test_defer_records_the_current_action_round():
    game = Game(seed=1)
    state = game.state
    state.ar_sequence = 7
    trigger = state.defer("Some Card", TEST_KIND, player=Side.USA)
    assert trigger.not_before == 7
    assert state.has_deferred(card="Some Card")
    assert state.has_deferred(kind=TEST_KIND)


def test_a_trigger_does_not_fire_in_the_action_round_that_created_it():
    """"Next action round" must exclude the one currently in progress."""
    game = Game(seed=1)
    state = game.state
    state.ar_sequence = 4
    state.defer("Some Card", TEST_KIND, player=Side.USA, when="end")

    # Same sequence number: not yet.
    run_silent(game._fire_deferred(Side.USA, "end"))
    assert state.has_deferred(kind=TEST_KIND)

    # A later action round for that player: now.
    state.ar_sequence = 5
    run_silent(game._fire_deferred(Side.USA, "end"))
    assert not state.has_deferred(kind=TEST_KIND)
    assert state.in_play("__fired__")


def test_a_trigger_ignores_the_other_players_action_rounds():
    game = Game(seed=1)
    state = game.state
    state.defer("Some Card", TEST_KIND, player=Side.USA, when="end")
    state.ar_sequence += 5

    run_silent(game._fire_deferred(Side.USSR, "end"))
    assert state.has_deferred(kind=TEST_KIND)
    run_silent(game._fire_deferred(Side.USA, "end"))
    assert not state.has_deferred(kind=TEST_KIND)


def test_a_trigger_only_fires_at_the_moment_it_asked_for():
    game = Game(seed=1)
    state = game.state
    state.defer("Some Card", TEST_KIND, player=Side.USA, when="end")
    state.ar_sequence += 1

    run_silent(game._fire_deferred(Side.USA, "start"))
    assert state.has_deferred(kind=TEST_KIND)
    run_silent(game._fire_deferred(Side.USA, "end"))
    assert not state.has_deferred(kind=TEST_KIND)


def test_cancel_deferred_requires_a_filter():
    game = Game(seed=1)
    game.state.defer("Some Card", TEST_KIND, player=Side.USA)
    with pytest.raises(ValueError, match="needs a card or a kind"):
        game.state.cancel_deferred()
    assert game.state.cancel_deferred(card="Some Card") == 1
    assert not game.state.has_deferred(kind=TEST_KIND)


def test_triggers_fire_during_a_real_action_round():
    """End to end: schedule for the opponent, then let the game run into it."""
    game = advance_to_action_round(Game(seed=5), seed=5)
    state = game.state
    target = state.player.opponent
    state.defer("Some Card", TEST_KIND, player=target, when="end")

    rng = random.Random(5)
    for _ in range(400):
        if game.decision is None or state.in_play("__fired__"):
            break
        game.step(rng.choice(game.decision.options))
    assert state.in_play("__fired__"), "the deferred trigger never fired"


# --------------------------------------------------------------------------- #
# #50 We Will Bury You
# --------------------------------------------------------------------------- #


def test_we_will_bury_you_schedules_its_vp_instead_of_awarding_them():
    game = Game(seed=1)
    state = game.state
    state.defcon = 4
    before = state.vp

    run_silent(
        we_will_bury_you(game, EventContext(card="We Will Bury You", player=Side.USSR, ops=4))
    )

    assert state.defcon == 3                      # DEFCON drops immediately
    assert state.vp == before                     # the VP do not
    assert state.has_deferred(card="We Will Bury You", kind=WE_WILL_BURY_YOU)


def test_we_will_bury_you_pays_out_at_the_end_of_the_us_action_round():
    game = Game(seed=1)
    state = game.state
    state.defcon = 4
    before = state.vp

    run_silent(
        we_will_bury_you(game, EventContext(card="We Will Bury You", player=Side.USSR, ops=4))
    )
    state.ar_sequence += 1
    run_silent(game._fire_deferred(Side.USA, "end"))

    assert state.vp == before + 3                 # USSR-positive track
    assert not state.has_deferred(kind=WE_WILL_BURY_YOU)


def test_un_intervention_cancels_we_will_bury_you():
    game = Game(seed=1)
    state = game.state
    state.defcon = 4
    before = state.vp

    run_silent(
        we_will_bury_you(game, EventContext(card="We Will Bury You", player=Side.USSR, ops=4))
    )
    # What UN Intervention's handler does when the US plays it as an event.
    assert state.cancel_deferred(card="We Will Bury You") == 1

    state.ar_sequence += 1
    run_silent(game._fire_deferred(Side.USA, "end"))
    assert state.vp == before, "cancelled victory points were still awarded"


def test_we_will_bury_you_does_not_pay_out_twice():
    game = Game(seed=1)
    state = game.state
    state.defcon = 4
    run_silent(
        we_will_bury_you(game, EventContext(card="We Will Bury You", player=Side.USSR, ops=4))
    )
    state.ar_sequence += 1
    run_silent(game._fire_deferred(Side.USA, "end"))
    after_first = state.vp
    state.ar_sequence += 1
    run_silent(game._fire_deferred(Side.USA, "end"))
    assert state.vp == after_first


# --------------------------------------------------------------------------- #
# #49 Missile Envy
# --------------------------------------------------------------------------- #


def test_missile_envy_compels_the_recipient_to_spend_it():
    game = advance_to_action_round(Game(seed=2), seed=2)
    state = game.state
    ussr, usa = Side.USSR, Side.USA

    # Give the US a single card so the exchange is deterministic.
    state.hands[usa] = ["Duck and Cover"]
    state.hands[ussr] = ["Missile Envy"]

    gen = missile_envy(game, EventContext(card="Missile Envy", player=ussr, ops=2))
    # Duck and Cover is a US event, so the USSR spends it as operations: that asks
    # questions, which is fine -- drive them randomly.
    rng = random.Random(2)
    decision = next(gen, None)
    while decision is not None:
        try:
            decision = gen.send(rng.choice(decision.options))
        except StopIteration:
            break

    assert state.must_play[int(usa)] == "Missile Envy"
    assert "Missile Envy" in state.hands[usa], "the card should have changed hands"


def test_a_compelled_card_cannot_be_used_for_its_event_or_the_space_race():
    game = advance_to_action_round(Game(seed=4), seed=4)
    side = game.state.player

    # Truman Doctrine is a US event with a playable event and enough ops for the space
    # race, so all three uses would normally be on offer.
    name = "Truman Doctrine"
    game.state.hands[Side.USA] = [name]
    game.state.hands[Side.USSR] = [name] if side is Side.USSR else []

    free = next(game._choose_use(name, Side.USA), None)
    forced = next(game._choose_use(name, Side.USA, operations_only=True), None)

    if free is not None:
        offered = {a.value for a in free.options}
        assert OpsUse.INFLUENCE.value in offered
    if forced is not None:
        offered = {a.value for a in forced.options}
        assert OpsUse.EVENT.value not in offered
        assert OpsUse.SPACE.value not in offered
        assert offered <= {
            OpsUse.INFLUENCE.value,
            OpsUse.COUP.value,
            OpsUse.REALIGN.value,
            OpsUse.DISCARD.value,
        }


def test_compulsion_expires_after_one_action_round():
    """A player who cannot satisfy the compulsion is not stuck with it forever."""
    game = advance_to_action_round(Game(seed=6), seed=6)
    state = game.state
    state.must_play[int(state.player.opponent)] = "Missile Envy"

    rng = random.Random(6)
    seen_none = False
    for _ in range(300):
        if game.decision is None:
            break
        game.step(rng.choice(game.decision.options))
        if all(v is None for v in state.must_play.values()):
            seen_none = True
            break
    assert seen_none, "the compulsion was never cleared"


# --------------------------------------------------------------------------- #
# Deferred triggers must not break the usual invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(12))
def test_random_games_still_terminate_and_drain_their_triggers(seed):
    game = Game(seed=seed)
    rng = random.Random(seed)
    steps = 0
    while game.decision is not None:
        game.step(rng.choice(game.decision.options))
        steps += 1
        assert steps < 200_000
    # A finished game may legitimately still hold a trigger whose action round never
    # arrived, but the list must never grow without bound.
    assert len(game.state.deferred) <= 4
    assert all(
        name in ("We Will Bury You",) or name.startswith("__")
        for name in (t.card for t in game.state.deferred)
    )

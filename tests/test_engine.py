"""Turn structure, setup, determinism, and the decision protocol."""

from __future__ import annotations

import random

import pytest

from twilight import data
from twilight.decisions import IllegalAction, NUM_ACTIONS
from twilight.engine import ACTION_ROUNDS, HAND_SIZE, Game, SETUP_INFLUENCE
from twilight.enums import Phase, Side, Stage, WinReason
from twilight.events import coverage, missing_handlers


def play_random(game: Game, rng: random.Random, limit: int = 200_000) -> int:
    steps = 0
    while game.decision is not None and steps < limit:
        game.step(rng.choice(game.decision.options))
        steps += 1
    return steps


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #


def test_setup_deals_before_asking_for_influence():
    """Both players see their hand before placing opening influence."""
    game = Game(seed=1)
    assert game.decision is not None
    assert game.decision.player is Side.USSR
    assert len(game.state.hands[Side.USSR]) == HAND_SIZE[Stage.EARLY_WAR]
    assert len(game.state.hands[Side.USA]) == HAND_SIZE[Stage.EARLY_WAR]


def test_setup_influence_totals():
    """15 USSR and 25 US influence, as printed in the rulebook."""
    game = Game(seed=1)
    rng = random.Random(1)
    # Play through setup only: 6 USSR + 7 US free placements.
    while game.decision is not None and game.state.phase is Phase.SETUP:
        game.step(rng.choice(game.decision.options))

    assert sum(game.state.influence[Side.USSR]) == 15
    assert sum(game.state.influence[Side.USA]) == 25


def test_fixed_setup_matches_the_rulebook():
    game = Game(seed=1)
    for side, placements in SETUP_INFLUENCE.items():
        for name, amount in placements.items():
            assert game.state.inf(side, name) >= amount, f"{side.label} {name}"
    # Canada 2 is easy to miss and changes the US total.
    assert game.state.inf(Side.USA, "Canada") == 2


def test_ussr_places_setup_influence_only_in_eastern_europe():
    game = Game(seed=3)
    assert game.decision is not None
    for action in game.decision.options:
        name = action.value
        assert data.country(name).in_region(data.Region.EASTERN_EUROPE) if False else True
    names = {a.value for a in game.decision.options}
    from twilight.enums import Region

    assert names <= set(data.REGION_COUNTRIES[Region.EASTERN_EUROPE])


def test_china_card_starts_face_up_with_the_ussr():
    game = Game(seed=1)
    assert game.state.china_card_owner is Side.USSR
    assert game.state.china_card_face_up


def test_china_card_is_not_in_the_deck_or_a_hand():
    game = Game(seed=1)
    state = game.state
    assert data.CHINA_CARD not in state.deck
    assert data.CHINA_CARD not in state.hands[Side.USSR]
    assert data.CHINA_CARD not in state.hands[Side.USA]


# --------------------------------------------------------------------------- #
# Decision protocol
# --------------------------------------------------------------------------- #


def test_every_decision_has_at_least_one_legal_option():
    game = Game(seed=5)
    rng = random.Random(5)
    while game.decision is not None:
        assert game.decision.options
        game.step(rng.choice(game.decision.options))


def test_actions_can_be_given_as_key_or_vocabulary_index():
    game = Game(seed=2)
    assert game.decision is not None
    key = game.decision.options[0].key
    index = game.decision.options[0].index
    assert 0 <= index < NUM_ACTIONS

    twin = Game(seed=2)
    game.step(key)
    twin.step(index)
    assert game.state.influence == twin.state.influence


def test_illegal_actions_are_rejected():
    game = Game(seed=2)
    with pytest.raises(IllegalAction):
        game.step("country:Chile")  # not an Eastern European setup option
    with pytest.raises(IllegalAction):
        game.step("pass")


def test_action_mask_matches_the_option_list():
    game = Game(seed=4)
    mask = game.decision.mask()
    assert sum(mask) == len(game.decision.options)
    for action in game.decision.options:
        assert mask[action.index]


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_same_seed_and_actions_give_the_same_trajectory():
    a, b = Game(seed=11), Game(seed=11)
    rng = random.Random(11)
    while a.decision is not None:
        choice = rng.choice(a.decision.options)
        a.step(choice)
        b.step(choice.key)
    assert a.state.vp == b.state.vp
    assert a.state.influence == b.state.influence
    assert a.state.winner is b.state.winner
    assert a.history == b.history


def test_clone_reproduces_the_position():
    game = Game(seed=13)
    rng = random.Random(13)
    for _ in range(80):
        if game.decision is None:
            break
        game.step(rng.choice(game.decision.options))

    twin = game.clone()
    assert twin.history == game.history
    assert twin.state.influence == game.state.influence
    assert twin.state.vp == game.state.vp
    assert twin.state.defcon == game.state.defcon
    if game.decision is not None:
        assert twin.decision is not None
        assert twin.decision.legal_keys == game.decision.legal_keys


def test_different_seeds_diverge():
    outcomes = set()
    for seed in range(6):
        game = Game(seed=seed)
        play_random(game, random.Random(seed))
        outcomes.add((game.state.turn, game.state.vp))
    assert len(outcomes) > 1


# --------------------------------------------------------------------------- #
# Turn structure and endings
# --------------------------------------------------------------------------- #


def test_action_round_counts_per_stage():
    assert ACTION_ROUNDS[Stage.EARLY_WAR] == 6
    assert ACTION_ROUNDS[Stage.MID_WAR] == 7
    assert ACTION_ROUNDS[Stage.LATE_WAR] == 7
    assert HAND_SIZE[Stage.EARLY_WAR] == 8
    assert HAND_SIZE[Stage.MID_WAR] == 9


@pytest.mark.parametrize("seed", range(24))
def test_random_games_terminate_with_a_valid_outcome(seed):
    game = Game(seed=seed)
    play_random(game, random.Random(seed))

    state = game.state
    assert state.is_over
    assert state.win_reason is not None
    assert isinstance(state.win_reason, WinReason)
    assert state.winner is None or isinstance(state.winner, Side)
    assert 1 <= state.turn <= 10
    assert 1 <= state.defcon <= 5
    assert -20 <= state.vp <= 20


@pytest.mark.parametrize("seed", range(24))
def test_invariants_hold_throughout(seed):
    game = Game(seed=seed)
    rng = random.Random(seed)
    while game.decision is not None:
        state = game.state
        assert 1 <= state.defcon <= 5
        assert -20 <= state.vp <= 20
        assert 1 <= state.turn <= 10
        for side in Side:
            assert all(v >= 0 for v in state.influence[side])
            assert 0 <= state.space_race[side] <= 8
            assert 0 <= state.military_ops[side] <= 5
        game.step(rng.choice(game.decision.options))


@pytest.mark.parametrize("seed", range(16))
def test_every_card_is_always_accounted_for(seed):
    """No card may vanish. Losing one silently shrinks the deck for the rest of the game."""
    game = Game(seed=seed)
    rng = random.Random(seed)
    # The China Card is tracked separately, and the optional cards are out of the
    # default deck.
    expected = {
        name
        for name, card in data.CARDS.items()
        if name != data.CHINA_CARD and not card.optional
    }

    while game.decision is not None:
        state = game.state
        located: set[str] = set()
        for side in Side:
            located |= set(state.hands[side])
        located |= set(state.deck) | set(state.discard) | set(state.removed)
        located |= set(state.effects)  # cards sitting in play on the table
        # A chosen headline waits here between selection and resolution, and the card
        # currently being played is in neither a hand nor a pile.
        located |= {name for name in state.headline.values() if name is not None}
        located |= state.transit  # held by a resolving event
        if state.playing_card is not None:
            located.add(state.playing_card)

        # Cards from stages not yet dealt in are legitimately absent.
        not_yet_in_play = {
            name
            for name, card in data.CARDS.items()
            if card.stage not in game._stages_added
        }
        missing = expected - located - not_yet_in_play
        assert not missing, f"cards lost from the game: {sorted(missing)}"
        game.step(rng.choice(game.decision.options))


@pytest.mark.parametrize("seed", range(16))
def test_no_card_is_ever_in_two_places(seed):
    game = Game(seed=seed)
    rng = random.Random(seed)
    while game.decision is not None:
        state = game.state
        seen: list[str] = []
        for side in Side:
            seen.extend(state.hands[side])
        seen.extend(state.deck)
        seen.extend(state.discard)
        seen.extend(state.removed)
        assert len(seen) == len(set(seen)), f"duplicated: {sorted(seen)}"
        game.step(rng.choice(game.decision.options))


def test_defcon_one_loses_for_whoever_caused_it():
    game = Game(seed=1)
    game.state.defcon = 2
    game.degrade_defcon(1, Side.USSR)
    assert game.state.is_over
    assert game.state.winner is Side.USA
    assert game.state.win_reason is WinReason.DEFCON


def test_defcon_blame_falls_on_the_phasing_player_not_the_event_owner():
    """Playing an opponent's card for ops fires their event, but you carry the blame."""
    game = Game(seed=1)
    state = game.state
    state.defcon = 2
    state.phasing_player = Side.USSR
    # The US owns the event doing the degrading, yet the USSR is phasing and loses.
    game.degrade_defcon(1, Side.USA)
    assert state.winner is Side.USA
    assert state.win_reason is WinReason.DEFCON


def test_victory_points_end_the_game_at_twenty():
    game = Game(seed=1)
    game.state.award_vp(Side.USSR, 20)
    game._check_victory_points()
    assert game.state.winner is Side.USSR
    assert game.state.win_reason is WinReason.VICTORY_POINTS


def test_military_operations_shortfall_is_netted():
    """Only the difference is scored when both players are short."""
    game = Game(seed=1)
    state = game.state
    state.defcon = 3
    state.military_ops = [2, 1]
    before = state.vp
    game._military_operations_check()
    assert state.vp - before == 1  # USSR ahead by one, not USSR +2 / US +1


def test_military_operations_equal_totals_score_nothing():
    game = Game(seed=1)
    game.state.defcon = 5
    game.state.military_ops = [0, 0]
    before = game.state.vp
    game._military_operations_check()
    assert game.state.vp == before


# --------------------------------------------------------------------------- #
# Event registry
# --------------------------------------------------------------------------- #


def test_every_card_has_an_event_handler():
    missing = missing_handlers()
    done, total = coverage()
    assert not missing, (
        f"{len(missing)} of {total} cards have no event handler: {missing}"
    )
    assert done == total


@pytest.mark.parametrize("seed", range(8))
def test_strict_mode_finds_no_unimplemented_events(seed):
    """With every event registered, strict mode must never trip."""
    game = Game(seed=seed, strict_events=True)
    play_random(game, random.Random(seed))
    assert not game.unimplemented

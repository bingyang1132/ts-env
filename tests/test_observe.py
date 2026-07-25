"""Observation correctness, information hiding, and the encoders.

Information hiding is the part worth being strict about: a leak here silently teaches
an agent to read its opponent's hand, and the resulting policy is worthless.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from twilight import Game, Side, TwilightStruggleEnv, observe, render
from twilight.encode import action_mask, encode, flatten, observation_shapes
from twilight.enums import Region


def advance(game: Game, steps: int, seed: int = 0) -> Game:
    rng = random.Random(seed)
    for _ in range(steps):
        if game.decision is None:
            break
        game.step(rng.choice(game.decision.options))
    return game


# --------------------------------------------------------------------------- #
# Information hiding
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(6))
def test_opponent_hand_is_never_leaked(seed):
    game = advance(Game(seed=seed), 120, seed)
    for player in Side:
        obs = observe(game.state, player, game.decision)
        opponent_hand = set(game.state.hand(player.opponent))
        # The count is public; the contents are not.
        assert obs.opponent_hand_size == len(opponent_hand)
        if obs.opponent_hand_revealed is None:
            leaked = opponent_hand & set(obs.hand)
            assert not leaked, f"opponent cards appeared in own hand: {leaked}"


@pytest.mark.parametrize("seed", range(6))
def test_text_view_does_not_mention_hidden_cards(seed):
    """The rendered prompt must not name a card only the opponent holds."""
    game = advance(Game(seed=seed), 120, seed)
    if game.decision is None:
        pytest.skip("game ended early")
    for player in Side:
        obs = observe(game.state, player, game.decision)
        if obs.opponent_hand_revealed is not None:
            continue
        text = render(obs)
        public = (
            set(obs.hand) | set(obs.discard) | set(obs.removed)
            | {name for name, _ in obs.effects}
        )
        for card in game.state.hand(player.opponent):
            if card in public:
                continue  # a card can be in the log for legitimate public reasons
            assert card not in text.split("RECENT EVENTS")[0], (
                f"{card!r} is only in the opponent's hand but appears in the view"
            )


def test_draw_pile_order_is_hidden_but_composition_is_inferable():
    game = advance(Game(seed=3), 60, 3)
    obs = observe(game.state, Side.USSR, game.decision)
    # Deck order is never exposed as a sequence.
    assert not hasattr(obs, "deck")
    # Everything publicly accounted for is excluded from what might still be drawn.
    possible = set(obs.deck_possible)
    assert possible.isdisjoint(set(obs.hand))
    assert possible.isdisjoint(set(obs.discard))
    assert possible.isdisjoint(set(obs.removed))
    # The opponent's hand is unknown, so those cards legitimately remain possible.
    assert set(game.state.hand(Side.USA)) <= possible


def test_a_player_only_sees_a_decision_addressed_to_them():
    game = Game(seed=8)
    assert game.decision is not None
    asked = game.decision.player
    assert observe(game.state, asked, game.decision).decision is not None
    assert observe(game.state, asked.opponent, game.decision).decision is None


# --------------------------------------------------------------------------- #
# Derived facts
# --------------------------------------------------------------------------- #


def test_vp_is_reported_from_the_observer_perspective():
    game = Game(seed=1)
    game.state.award_vp(Side.USSR, 6)
    assert observe(game.state, Side.USSR).vp == 6
    assert observe(game.state, Side.USA).vp == -6


def test_region_preview_is_zero_sum_between_the_players():
    game = advance(Game(seed=5), 100, 5)
    ussr = observe(game.state, Side.USSR)
    usa = observe(game.state, Side.USA)
    for region in (Region.EUROPE, Region.ASIA, Region.AFRICA):
        a = ussr.region(region).net_vp_for_observer
        b = usa.region(region).net_vp_for_observer
        assert a == -b


def test_country_view_reports_control_thresholds():
    game = Game(seed=1)
    state = game.state
    state.set_inf(Side.USA, "Italy", 3)   # stability 2
    state.set_inf(Side.USSR, "Italy", 0)
    view = observe(state, Side.USSR).country("Italy")
    assert view.controller is Side.USA
    # The USSR needs stability + their influence = 2 + 3 = 5.
    assert view.to_control == 5
    assert view.place_cost == 2


def test_legal_country_targets_are_marked_in_the_encoding():
    game = Game(seed=2)
    assert game.decision is not None
    obs = observe(game.state, game.decision.player, game.decision)
    matrix = encode(obs)["countries"]
    from twilight.encode import COUNTRY_FEATURES

    column = COUNTRY_FEATURES.index("is_legal_target")
    marked = {
        obs.countries[i].name for i in range(len(obs.countries)) if matrix[i, column] > 0
    }
    expected = {a.value for a in game.decision.options if str(a.kind) == "country"}
    assert marked == expected


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #


def test_encoded_shapes_are_stable_and_finite():
    shapes = observation_shapes()
    assert shapes["countries"][0] == 84
    assert shapes["hand"] == (110,)
    assert shapes["action_mask"][0] > 0

    game = advance(Game(seed=4), 70, 4)
    for player in Side:
        encoded = encode(observe(game.state, player, game.decision))
        for key, array in encoded.items():
            if array.dtype == bool:
                continue
            assert np.all(np.isfinite(array)), key
            assert np.all(np.abs(array) <= 1.0 + 1e-6), f"{key} outside [-1, 1]"
        assert flatten(encoded).shape == shapes["flat"]


def test_action_mask_matches_the_legal_options():
    game = advance(Game(seed=6), 40, 6)
    if game.decision is None:
        pytest.skip("game ended early")
    obs = observe(game.state, game.decision.player, game.decision)
    mask = action_mask(obs)
    assert mask.sum() == len(game.decision.options)
    for a in game.decision.options:
        assert mask[a.index]


def test_encoding_is_deterministic():
    a = encode(observe(advance(Game(seed=9), 50, 9).state, Side.USSR))
    b = encode(observe(advance(Game(seed=9), 50, 9).state, Side.USSR))
    for key in a:
        assert np.array_equal(a[key], b[key]), key


def test_text_render_is_deterministic():
    first = render(observe(advance(Game(seed=9), 50, 9).state, Side.USSR))
    second = render(observe(advance(Game(seed=9), 50, 9).state, Side.USSR))
    assert first == second


# --------------------------------------------------------------------------- #
# Environment wrapper
# --------------------------------------------------------------------------- #


def test_env_runs_a_full_game_and_reports_a_winner():
    env = TwilightStruggleEnv(seed=2)
    rng = random.Random(2)
    terminated = False
    info: dict = {}
    while not terminated:
        assert env.decision is not None
        _, _, terminated, truncated, info = env.step(rng.choice(env.decision.options))
        if truncated:
            pytest.fail("a random game should not need truncating")
    assert info["winner"] in ("USSR", "USA", "draw")
    assert info["rewards"]["USSR"] == -info["rewards"]["USA"]


def test_sparse_reward_is_zero_until_the_end():
    env = TwilightStruggleEnv(seed=3, reward_mode="sparse")
    rng = random.Random(3)
    while True:
        result = env.step(rng.choice(env.decision.options))
        if result.terminated:
            assert abs(result.reward) in (0.0, 1.0)
            break
        assert result.reward == 0.0


def test_vp_delta_reward_tracks_the_track():
    env = TwilightStruggleEnv(seed=4, reward_mode="vp_delta")
    rng = random.Random(4)
    total_ussr = 0.0
    while True:
        mover = env.decision.player
        vp_before = env.game.state.vp
        result = env.step(rng.choice(env.decision.options))
        delta = env.game.state.vp - vp_before
        shaped = result.reward - (
            0.0 if not result.terminated else _terminal_part(env, mover)
        )
        assert shaped == pytest.approx((delta if mover is Side.USSR else -delta) / 20.0)
        total_ussr += delta
        if result.terminated:
            break
    assert env.game.state.vp == total_ussr


def _terminal_part(env: TwilightStruggleEnv, mover: Side) -> float:
    winner = env.game.state.winner
    if winner is None:
        return 0.0
    return 1.0 if winner is mover else -1.0


def test_env_reset_is_reproducible():
    env = TwilightStruggleEnv(seed=5, encode_observations=False)
    first = env.text()
    env.reset(5)
    assert env.text() == first


def test_observation_for_works_for_both_sides():
    env = TwilightStruggleEnv(seed=7, encode_observations=False)
    for side in Side:
        obs = env.observation_for(side)
        assert obs.player is side
        assert isinstance(render(obs), str)

"""Numeric encoding of an observation, for learned policies.

The board is emitted as a ``(84, K)`` matrix -- one row per country, in a fixed order --
rather than flattened, so a network can share weights across countries and attend over
them. Everything else goes into a global vector, plus multi-hot card sets and a boolean
action mask over the closed action vocabulary.

``flatten(...)`` concatenates it all for a plain MLP baseline. All features are scaled
into roughly ``[0, 1]`` (or ``[-1, 1]`` for victory points, which are signed from the
observer's point of view).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .data import CARD_ORDER, CARD_SLOT, COUNTRIES, NUM_CARDS, NUM_COUNTRIES
from .decisions import ACTION_INDEX, ActionKind, DecisionType, NUM_ACTIONS
from .enums import AUTO_VICTORY, Phase, Region, Side
from .observe import Observation

#: Influence in one country is usually 0-5 but has no hard ceiling, so features
#: saturate at this value rather than running outside [0, 1].
INFLUENCE_CAP = 10.0
#: Hands are dealt to 8 or 9 but can carry extra cards over, so they saturate too.
HAND_CAP = 12.0


def _sat(value: float, cap: float) -> float:
    """Scale into [0, 1], saturating at *cap* instead of overflowing."""
    return min(float(value), cap) / cap

_REGIONS = tuple(Region)
_PHASES = tuple(Phase)
_DECISION_TYPES = tuple(DecisionType)
_TIERS = ("none", "presence", "domination", "control")

#: Country feature names, in column order. Exposed so callers can label or ablate them.
COUNTRY_FEATURES: tuple[str, ...] = (
    "ussr_influence",
    "usa_influence",
    "my_influence",
    "their_influence",
    "i_control",
    "they_control",
    "uncontrolled",
    "battleground",
    "stability",
    "points_to_control",
    "can_place",
    "costs_double",
    "can_coup",
    "coup_success_chance",
    "can_realign",
    "adjacent_enemy_superpower",
    "adjacent_my_superpower",
    "is_legal_target",
) + tuple(f"region_{r.name.lower()}" for r in _REGIONS)

NUM_COUNTRY_FEATURES = len(COUNTRY_FEATURES)


def _one_hot(value: Any, options: tuple, out: list[float]) -> None:
    for option in options:
        out.append(1.0 if value == option else 0.0)


def encode_countries(obs: Observation) -> np.ndarray:
    """``(NUM_COUNTRIES, NUM_COUNTRY_FEATURES)`` board matrix."""
    legal_targets = set()
    if obs.decision is not None:
        legal_targets = {
            a.value for a in obs.decision.options if a.kind is ActionKind.COUNTRY
        }

    matrix = np.zeros((NUM_COUNTRIES, NUM_COUNTRY_FEATURES), dtype=np.float32)
    for row, view in enumerate(obs.countries):
        mine = view.mine(obs.player)
        theirs = view.theirs(obs.player)
        # Faces of a d6 that would beat twice the stability, given a 4-op coup. A
        # perspective-free difficulty signal that saves the policy the arithmetic.
        coup_chance = max(0, min(6, 4 - 2 * view.stability + 6)) / 6.0

        features: list[float] = [
            _sat(view.ussr, INFLUENCE_CAP),
            _sat(view.usa, INFLUENCE_CAP),
            _sat(mine, INFLUENCE_CAP),
            _sat(theirs, INFLUENCE_CAP),
            1.0 if view.controller is obs.player else 0.0,
            1.0 if view.controller is obs.opponent else 0.0,
            1.0 if view.controller is None else 0.0,
            1.0 if view.battleground else 0.0,
            view.stability / 5.0,
            _sat(view.to_control, 10.0),
            1.0 if view.can_place else 0.0,
            1.0 if view.place_cost == 2 else 0.0,
            1.0 if view.can_coup else 0.0,
            coup_chance,
            1.0 if view.can_realign else 0.0,
            1.0 if view.adjacent_to_enemy_superpower else 0.0,
            1.0 if view.name in COUNTRIES[obs.player.label].adjacent else 0.0,
            1.0 if view.name in legal_targets else 0.0,
        ]
        regions = COUNTRIES[view.name].regions
        features.extend(1.0 if r in regions else 0.0 for r in _REGIONS)
        matrix[row] = features
    return matrix


def encode_global(obs: Observation) -> np.ndarray:
    """Flat vector of tracks, hand sizes, phase, and per-region standings."""
    out: list[float] = [
        obs.vp / 20.0,
        obs.defcon / 5.0,
        _sat(obs.action_round, 8.0),
        obs.turn / 10.0,
        obs.space_race[obs.player] / 8.0,
        obs.space_race[obs.opponent] / 8.0,
        (obs.space_race[obs.player] - obs.space_race[obs.opponent]) / 8.0,
        _sat(obs.space_attempts_left, 2.0),
        obs.military_ops[obs.player] / 5.0,
        obs.military_ops[obs.opponent] / 5.0,
        obs.military_ops_required / 5.0,
        (obs.military_ops[obs.player] - obs.military_ops_required) / 5.0,
        _sat(len(obs.hand), HAND_CAP),
        _sat(obs.opponent_hand_size, HAND_CAP),
        obs.deck_size / float(NUM_CARDS),
        len(obs.unseen) / float(NUM_CARDS),
        obs.in_deck_odds,
        len(obs.unseen_scoring_cards()) / 7.0,
        len(obs.discard) / float(NUM_CARDS),
        len(obs.removed) / float(NUM_CARDS),
        1.0 if obs.china_card_owner is obs.player else 0.0,
        1.0 if obs.china_card_face_up else 0.0,
        1.0 if obs.china_card_available else 0.0,
        1.0 if obs.player is Side.USSR else 0.0,
        1.0 if obs.to_move is obs.player else 0.0,
    ]
    _one_hot(obs.defcon, (1, 2, 3, 4, 5), out)
    _one_hot(obs.turn, tuple(range(1, 11)), out)
    _one_hot(obs.phase, _PHASES, out)
    _one_hot(
        obs.decision.type if obs.decision is not None else None, _DECISION_TYPES, out
    )

    for view in obs.regions:
        _one_hot(view.tiers[obs.player], _TIERS, out)
        _one_hot(view.tiers[obs.opponent], _TIERS, out)
        out.append(view.countries[obs.player] / 21.0)
        out.append(view.countries[obs.opponent] / 21.0)
        out.append(view.battlegrounds[obs.player] / 6.0)
        out.append(view.battlegrounds[obs.opponent] / 6.0)
        # Europe control is an automatic win, so clamp before scaling.
        net = view.net_vp_for_observer
        out.append(max(-1.0, min(1.0, net / 12.0)) if abs(net) < AUTO_VICTORY else
                   (1.0 if net > 0 else -1.0))
    return np.asarray(out, dtype=np.float32)


def _card_multi_hot(names: tuple[str, ...]) -> np.ndarray:
    vec = np.zeros(NUM_CARDS, dtype=np.float32)
    for name in names:
        slot = CARD_SLOT.get(name)
        if slot is not None:
            vec[slot] = 1.0
    return vec


def action_mask(obs: Observation) -> np.ndarray:
    """Boolean mask over the whole action vocabulary."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    if obs.decision is not None:
        for action in obs.decision.options:
            mask[ACTION_INDEX[action.key]] = True
    return mask


def encode(obs: Observation) -> dict[str, np.ndarray]:
    """Full structured encoding of *obs*."""
    return {
        "countries": encode_countries(obs),
        "global": encode_global(obs),
        "hand": _card_multi_hot(obs.hand),
        "discard": _card_multi_hot(obs.discard),
        "removed": _card_multi_hot(obs.removed),
        "effects": _card_multi_hot(tuple(name for name, _ in obs.effects)),
        # The card-counting channel: cards that could be in the opponent's hand or the
        # draw pile. Scaled by the odds of being in the deck, so a network sees both the
        # membership and how likely each is to come round.
        "unseen": _card_multi_hot(obs.unseen) * (obs.in_deck_odds or 1.0),
        "unseen_mask": _card_multi_hot(obs.unseen),
        "opponent_hand": _card_multi_hot(obs.opponent_hand_revealed or ()),
        "action_mask": action_mask(obs),
    }


def flatten(encoded: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate a structured encoding into one vector, excluding the action mask."""
    parts = [
        encoded["countries"].reshape(-1),
        encoded["global"],
        encoded["hand"],
        encoded["discard"],
        encoded["removed"],
        encoded["effects"],
        encoded["unseen"],
        encoded["opponent_hand"],
    ]
    return np.concatenate(parts).astype(np.float32)


def observation_shapes() -> dict[str, tuple[int, ...]]:
    """Shapes of each encoded component, for building network input layers.

    Computed from a real starting position rather than hand-maintained, so it cannot
    drift out of step with the encoder.
    """
    from .engine import Game

    game = Game(seed=0)
    from .observe import observe

    obs = observe(game.state, Side.USSR, game.decision)
    encoded = encode(obs)
    shapes = {key: tuple(value.shape) for key, value in encoded.items()}
    shapes["flat"] = (int(flatten(encoded).shape[0]),)
    return shapes


__all__ = [
    "COUNTRY_FEATURES",
    "NUM_COUNTRY_FEATURES",
    "action_mask",
    "encode",
    "encode_countries",
    "encode_global",
    "flatten",
    "observation_shapes",
]

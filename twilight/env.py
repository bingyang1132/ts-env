"""Agent-facing environment wrapper.

Twilight Struggle is a two-player zero-sum game with alternating -- and sometimes
consecutive -- decisions by the same side, so this follows the turn-based convention
used by PettingZoo's AEC API rather than pretending to be a single-agent Gym env:

* ``info["to_move"]`` names the side that must act now;
* :meth:`TwilightStruggleEnv.step` applies one atomic action for that side and returns
  the reward *from that side's point of view*;
* ``info["rewards"]`` gives both sides' rewards, which always sum to zero.

Reward modes:

``sparse``   0 everywhere, then +1 / -1 / 0 at the end. The honest objective.
``vp_delta`` the change in the victory point track each step, scaled by 1/20, plus the
             terminal result. Useful for bootstrapping, but note it is *not* a
             policy-invariant shaping: Europe control, DEFCON 1 and a held scoring card
             all end games without a corresponding VP move, so a policy trained purely
             on VP delta will undervalue and mishandle them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .decisions import Action, Decision
from .encode import action_mask, encode, flatten
from .engine import Game
from .enums import Side
from .observe import Observation, observe
from .render import render

RewardMode = Literal["sparse", "vp_delta"]

#: Terminal reward magnitude for a win.
WIN_REWARD = 1.0


@dataclass(slots=True)
class StepResult:
    """What one step produced, for callers who prefer attributes to a 5-tuple."""

    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def __iter__(self):
        yield from (self.observation, self.reward, self.terminated, self.truncated, self.info)


class TwilightStruggleEnv:
    """A single game, exposed as a step/reset environment.

    ``encode_observations=True`` yields the numeric dictionary from
    :mod:`twilight.encode`; ``False`` yields the :class:`~twilight.observe.Observation`
    itself, which is what a language-model agent wants (feed it through
    :func:`twilight.render.render`).
    """

    def __init__(
        self,
        seed: int | None = None,
        *,
        reward_mode: RewardMode = "sparse",
        optional_cards: bool = False,
        handicap: int = 0,
        strict_events: bool = False,
        encode_observations: bool = True,
        max_steps: int | None = 100_000,
    ) -> None:
        if reward_mode not in ("sparse", "vp_delta"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        self.reward_mode: RewardMode = reward_mode
        self.optional_cards = optional_cards
        self.handicap = handicap
        self.strict_events = strict_events
        self.encode_observations = encode_observations
        self.max_steps = max_steps

        self._seed = seed
        self.game: Game | None = None
        self.steps = 0
        self._last_vp = 0
        self.reset(seed)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            self._seed = seed
        self.game = Game(
            self._seed,
            optional_cards=self.optional_cards,
            handicap=self.handicap,
            strict_events=self.strict_events,
        )
        self.steps = 0
        self._last_vp = self.game.state.vp
        return self._observation(), self._info()

    def step(self, action: Action | str | int) -> StepResult:
        """Apply one atomic action for the side currently to move."""
        game = self._require_game()
        if game.decision is None:
            raise RuntimeError("the game is over; call reset() first")

        mover = game.decision.player
        vp_before = game.state.vp

        game.step(action)
        self.steps += 1

        reward = self._reward(mover, vp_before)
        terminated = game.decision is None
        truncated = bool(
            self.max_steps is not None and not terminated and self.steps >= self.max_steps
        )

        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=self._info(last_mover=mover, reward=reward),
        )

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #

    def _reward(self, mover: Side, vp_before: int) -> float:
        game = self._require_game()
        state = game.state
        reward = 0.0

        if self.reward_mode == "vp_delta":
            delta = state.vp - vp_before  # USSR-positive
            reward += (delta if mover is Side.USSR else -delta) / 20.0

        if game.decision is None:
            if state.winner is None:
                pass  # a draw is worth nothing to either side
            elif state.winner is mover:
                reward += WIN_REWARD
            else:
                reward -= WIN_REWARD
        return reward

    def rewards(self) -> dict[str, float]:
        """Terminal reward for each side, zero-sum. All zeros while the game runs."""
        game = self._require_game()
        if game.decision is not None or game.state.winner is None:
            return {Side.USSR.label: 0.0, Side.USA.label: 0.0}
        winner = game.state.winner
        return {
            Side.USSR.label: WIN_REWARD if winner is Side.USSR else -WIN_REWARD,
            Side.USA.label: WIN_REWARD if winner is Side.USA else -WIN_REWARD,
        }

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #

    def observation_for(self, player: Side) -> Observation:
        """*player*'s view, whether or not it is their turn.

        Useful for self-play value networks, which need both perspectives.
        """
        game = self._require_game()
        return observe(game.state, player, game.decision)

    def _observation(self) -> Any:
        game = self._require_game()
        player = game.decision.player if game.decision is not None else Side.USSR
        obs = observe(game.state, player, game.decision)
        return encode(obs) if self.encode_observations else obs

    def text(self, player: Side | None = None) -> str:
        """The text view for *player*, defaulting to whoever must act."""
        game = self._require_game()
        if player is None:
            player = game.decision.player if game.decision is not None else Side.USSR
        return render(self.observation_for(player))

    def render(self) -> str:
        return self.text()

    @property
    def decision(self) -> Decision | None:
        return self._require_game().decision

    @property
    def legal_actions(self) -> tuple[str, ...]:
        decision = self._require_game().decision
        return () if decision is None else decision.legal_keys

    def action_mask(self):
        game = self._require_game()
        player = game.decision.player if game.decision is not None else Side.USSR
        return action_mask(observe(game.state, player, game.decision))

    def flat_observation(self):
        obs = self._observation()
        if not isinstance(obs, dict):
            raise TypeError("flat_observation requires encode_observations=True")
        return flatten(obs)

    # ------------------------------------------------------------------ #
    # Info
    # ------------------------------------------------------------------ #

    def _info(
        self, last_mover: Side | None = None, reward: float | None = None
    ) -> dict[str, Any]:
        game = self._require_game()
        state = game.state
        decision = game.decision

        info: dict[str, Any] = {
            "to_move": decision.player.label if decision is not None else None,
            "decision_type": str(decision.type) if decision is not None else None,
            "prompt": decision.prompt if decision is not None else None,
            "legal_actions": self.legal_actions,
            "turn": state.turn,
            "action_round": state.action_round,
            "phase": state.phase.value,
            "vp": state.vp,
            "defcon": state.defcon,
            "steps": self.steps,
        }
        if last_mover is not None:
            info["last_mover"] = last_mover.label
            info["step_reward"] = reward
        if decision is None:
            info["winner"] = state.winner.label if state.winner is not None else "draw"
            info["win_reason"] = state.win_reason.value if state.win_reason else None
            info["rewards"] = self.rewards()
            info["unimplemented_events"] = list(game.unimplemented)
        return info

    def _require_game(self) -> Game:
        if self.game is None:  # pragma: no cover - reset runs in __init__
            raise RuntimeError("call reset() first")
        return self.game


__all__ = ["RewardMode", "StepResult", "TwilightStruggleEnv", "WIN_REWARD"]

"""Recording games, including each agent's stated reason for its choice.

A trajectory of action keys tells you *what* an agent did. Watching a replay and asking
"why on earth did it do that" is the actual work, and the answer is usually not
recoverable after the fact -- especially for a language model, whose reasoning exists only
at the moment it replies.

So an agent may annotate every choice it makes, and the annotation is stored beside the
action. :mod:`twilight.render` and ``tools/viz.py`` then show it against the board it was
made on.

The annotation channel is deliberately outside the engine: the rules do not care why a
move was made, and a rationale must never be able to affect play.

Agents opt in by either returning ``(action, note)`` from ``act`` or exposing a
``rationale`` attribute set during ``act``. Returning a bare action keeps working.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .decisions import Action, Decision
from .engine import Game
from .enums import Side

#: Longest annotation kept. Rationales are meant to be a sentence, not a transcript;
#: a runaway model reply would otherwise bloat every recording.
MAX_NOTE = 2000


@dataclass(slots=True)
class Step:
    """One atomic action, and why it was taken.

    ``player`` and ``decision_type`` are redundant -- a replay recovers them -- but they
    make a recording readable on its own, and :meth:`GameRecord.replay` cross-checks them
    so a corrupt file is caught rather than silently replayed into a different game.
    """

    action: str
    player: str
    decision_type: str
    #: The agent's own explanation, if it offered one.
    note: str | None = None
    #: Anything else the agent wants to keep: token counts, retries, raw replies.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameRecord:
    """A complete game: enough to replay it exactly, plus why each move was made."""

    seed: int | None
    optional_cards: bool = False
    steps: list[Step] = field(default_factory=list)
    winner: str | None = None
    win_reason: str | None = None
    final_vp: int | None = None
    final_turn: int | None = None
    #: Free-form: agent names, model ids, a timestamp supplied by the caller.
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- access ----------------------------------------------------------- #

    @property
    def history(self) -> list[str]:
        return [s.action for s in self.steps]

    @property
    def notes(self) -> list[str | None]:
        return [s.note for s in self.steps]

    def annotated(self) -> list[tuple[int, Step]]:
        """``(index, step)`` for the steps that carry a rationale."""
        return [(i, s) for i, s in enumerate(self.steps) if s.note]

    def __len__(self) -> int:
        return len(self.steps)

    # -- replay ----------------------------------------------------------- #

    def replay(self, *, upto: int | None = None, strict: bool = True) -> Game:
        """Rebuild the game by replaying the recorded actions.

        With *strict*, the recorded player and decision type must match what the engine
        asks at each step, which catches a recording made against different rules.
        """
        game = Game(self.seed, optional_cards=self.optional_cards)
        for index, step in enumerate(self.steps[:upto]):
            if game.decision is None:
                raise ValueError(f"recording has {len(self.steps)} steps but the game "
                                 f"ended after {index}")
            if strict:
                actual_player = game.decision.player.label
                actual_type = str(game.decision.type)
                if (actual_player, actual_type) != (step.player, step.decision_type):
                    raise ValueError(
                        f"step {index} does not match this engine: recorded "
                        f"{step.player}/{step.decision_type}, got "
                        f"{actual_player}/{actual_type}"
                    )
            game.step(step.action)
        return game

    # -- persistence ------------------------------------------------------ #

    def to_dict(self) -> dict:
        data = asdict(self)
        # Drop empty extras so hand-written recordings stay readable.
        for step in data["steps"]:
            if not step["extra"]:
                del step["extra"]
            if step["note"] is None:
                del step["note"]
        return data

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GameRecord":
        steps = [
            Step(
                action=s["action"],
                player=s["player"],
                decision_type=s["decision_type"],
                note=s.get("note"),
                extra=s.get("extra", {}),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            seed=data.get("seed"),
            optional_cards=data.get("optional_cards", False),
            steps=steps,
            winner=data.get("winner"),
            win_reason=data.get("win_reason"),
            final_vp=data.get("final_vp"),
            final_turn=data.get("final_turn"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GameRecord":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Running agents
# --------------------------------------------------------------------------- #

#: What an agent's ``act`` may return: an action, an action key, or either plus a note.
AgentReply = Action | str | int | tuple[Any, str | None]


def call_agent(agent: Any, game: Game, decision: Decision) -> tuple[Any, str | None, dict]:
    """Ask *agent* to choose, and collect any rationale it offers.

    Supports three conventions so an agent never has to care about recording:
    returning the action alone, returning ``(action, note)``, or setting
    ``agent.rationale`` (and optionally ``agent.extra``) while choosing.
    """
    if hasattr(agent, "rationale"):
        agent.rationale = None
    if hasattr(agent, "extra"):
        agent.extra = {}

    reply = agent.act(game, decision)

    note: str | None = None
    if isinstance(reply, tuple):
        if len(reply) != 2:
            raise TypeError(f"an agent returned a {len(reply)}-tuple; expected (action, note)")
        reply, note = reply
    if note is None:
        note = getattr(agent, "rationale", None)

    extra = dict(getattr(agent, "extra", {}) or {})
    if note is not None:
        note = str(note).strip()[:MAX_NOTE] or None
    return reply, note, extra


def play_game(
    agents: Mapping[Side, Any],
    seed: int | None = None,
    *,
    optional_cards: bool = False,
    max_steps: int = 200_000,
    on_step: Callable[[int, Game, Step], None] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GameRecord:
    """Play a whole game between *agents*, returning a full record.

    *on_step* is called after each action with ``(index, game, step)``, which is how
    ``tools/play.py`` prints moves as they happen without duplicating this loop.
    """
    game = Game(seed, optional_cards=optional_cards)
    record = GameRecord(
        seed=seed, optional_cards=optional_cards, metadata=dict(metadata or {})
    )

    while game.decision is not None:
        if len(record.steps) >= max_steps:
            raise RuntimeError(f"game did not finish within {max_steps} steps")
        decision = game.decision
        reply, note, extra = call_agent(agents[decision.player], game, decision)
        action = decision.resolve(reply)

        step = Step(
            action=action.key,
            player=decision.player.label,
            decision_type=str(decision.type),
            note=note,
            extra=extra,
        )
        record.steps.append(step)
        game.step(action)
        if on_step is not None:
            on_step(len(record.steps) - 1, game, step)

    state = game.state
    record.winner = state.winner.label if state.winner is not None else "draw"
    record.win_reason = state.win_reason.value if state.win_reason is not None else None
    record.final_vp = state.vp
    record.final_turn = state.turn
    return record


def record_from_game(game: Game, *, seed: int | None = None, **metadata: Any) -> GameRecord:
    """Build a record from a game that was driven some other way.

    The action history is enough to replay, so this recovers everything except the
    rationales, which were never captured.
    """
    replay = Game(seed if seed is not None else game.seed, optional_cards=game.optional_cards)
    steps: list[Step] = []
    for key in game.history:
        if replay.decision is None:
            break
        steps.append(
            Step(
                action=key,
                player=replay.decision.player.label,
                decision_type=str(replay.decision.type),
            )
        )
        replay.step(key)

    state = game.state
    return GameRecord(
        seed=game.seed,
        optional_cards=game.optional_cards,
        steps=steps,
        winner=state.winner.label if state.winner is not None else ("draw" if state.is_over else None),
        win_reason=state.win_reason.value if state.win_reason is not None else None,
        final_vp=state.vp,
        final_turn=state.turn,
        metadata=dict(metadata),
    )


__all__ = [
    "AgentReply",
    "GameRecord",
    "MAX_NOTE",
    "Step",
    "call_agent",
    "play_game",
    "record_from_game",
]

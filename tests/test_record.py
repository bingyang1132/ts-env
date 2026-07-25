"""Recording games and the agent-rationale channel.

The invariant that matters: a rationale is carried alongside a move and can never change
what the move does. If annotating an agent altered play, every annotated recording would
be of a different game than the unannotated one.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from twilight import Game, Side
from twilight.record import (
    GameRecord,
    MAX_NOTE,
    Step,
    call_agent,
    play_game,
    record_from_game,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from baselines import GreedyAgent, RandomAgent, SafeRandomAgent  # noqa: E402


class Plain:
    """Returns a bare action, the way an agent that knows nothing about notes would."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, game, decision):
        return self.rng.choice(decision.options)


class Tupled(Plain):
    """Returns ``(action, note)``."""

    def act(self, game, decision):
        return super().act(game, decision), f"because {decision.type}"


class Attributed(Plain):
    """Sets ``self.rationale`` while choosing."""

    def __init__(self, seed=0):
        super().__init__(seed)
        self.rationale = None
        self.extra = {}

    def act(self, game, decision):
        action = super().act(game, decision)
        self.rationale = f"chose {action.key}"
        self.extra = {"options": len(decision.options)}
        return action


class KeyOnly(Plain):
    """Returns a canonical key string rather than an Action."""

    def act(self, game, decision):
        return super().act(game, decision).key


# --------------------------------------------------------------------------- #
# The agent protocol
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", [Plain, Tupled, Attributed, KeyOnly])
def test_all_agent_conventions_are_accepted(cls):
    game = Game(seed=1)
    reply, note, extra = call_agent(cls(seed=1), game, game.decision)
    action = game.decision.resolve(reply)
    assert action in game.decision.options
    if cls is Plain or cls is KeyOnly:
        assert note is None
    else:
        assert note


def test_attributed_agent_supplies_extra():
    game = Game(seed=1)
    _, note, extra = call_agent(Attributed(seed=1), game, game.decision)
    assert note and note.startswith("chose ")
    assert extra["options"] == len(game.decision.options)


def test_a_stale_rationale_is_not_reused():
    """The recorder clears the attribute first, so a silent agent yields no note."""

    class SometimesSilent(Attributed):
        def act(self, game, decision):
            action = Plain.act(self, game, decision)
            if len(decision.options) % 2 == 0:
                self.rationale = "even number of options"
            return action

    agent = SometimesSilent(seed=2)
    game = Game(seed=2)
    seen = []
    while game.decision is not None and len(seen) < 40:
        reply, note, _ = call_agent(agent, game, game.decision)
        seen.append((len(game.decision.options) % 2 == 0, note))
        game.step(game.decision.resolve(reply))

    for even, note in seen:
        assert bool(note) is even, "a note leaked from a previous decision"


def test_a_bad_tuple_is_rejected():
    game = Game(seed=1)

    class Broken(Plain):
        def act(self, game, decision):
            return (super().act(game, decision), "note", "extra")

    with pytest.raises(TypeError, match="expected .action, note."):
        call_agent(Broken(), game, game.decision)


def test_long_notes_are_truncated():
    game = Game(seed=1)

    class Verbose(Plain):
        def act(self, game, decision):
            return super().act(game, decision), "x" * (MAX_NOTE * 3)

    _, note, _ = call_agent(Verbose(), game, game.decision)
    assert len(note) == MAX_NOTE


# --------------------------------------------------------------------------- #
# Annotations must not change play
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 4, 9])
def test_annotating_does_not_change_the_game(seed):
    """The same choices with and without notes must give the same trajectory."""
    quiet = play_game({s: Plain(seed=seed) for s in Side}, seed)
    loud = play_game({s: Tupled(seed=seed) for s in Side}, seed)

    assert quiet.history == loud.history
    assert quiet.winner == loud.winner
    assert quiet.final_vp == loud.final_vp
    assert all(s.note is None for s in quiet.steps)
    assert all(s.note for s in loud.steps)


def test_greedy_annotates_every_move_it_makes():
    record = play_game({s: GreedyAgent(seed=3) for s in Side}, 3)
    assert len(record) > 10
    assert all(s.note for s in record.steps)
    assert record.annotated() and len(record.annotated()) == len(record)


def test_plain_baselines_annotate_nothing():
    for cls in (RandomAgent, SafeRandomAgent):
        record = play_game({s: cls(seed=3) for s in Side}, 3)
        assert not record.annotated(), cls.__name__


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


def test_play_game_produces_a_replayable_record():
    record = play_game({s: Attributed(seed=6) for s in Side}, 6)
    assert record.winner in ("USSR", "USA", "draw")
    assert record.final_turn and 1 <= record.final_turn <= 10

    replayed = record.replay()
    assert replayed.history == record.history
    assert replayed.state.vp == record.final_vp
    assert replayed.state.winner is not None or record.winner == "draw"


def test_replay_can_stop_partway():
    record = play_game({s: Plain(seed=7) for s in Side}, 7)
    half = record.replay(upto=len(record) // 2)
    assert len(half.history) == len(record) // 2
    assert half.decision is not None


def test_record_round_trips_through_json(tmp_path: Path):
    record = play_game({s: Attributed(seed=8) for s in Side}, 8)
    path = record.save(tmp_path / "g.json")
    reloaded = GameRecord.load(path)

    assert reloaded.seed == record.seed
    assert reloaded.history == record.history
    assert reloaded.notes == record.notes
    assert reloaded.winner == record.winner
    assert reloaded.final_vp == record.final_vp
    assert reloaded.steps[0].extra == record.steps[0].extra
    assert reloaded.replay().history == record.history


def test_saved_records_omit_empty_fields(tmp_path: Path):
    """Keeps hand-written and hand-read recordings legible."""
    import json

    record = play_game({s: Plain(seed=2) for s in Side}, 2)
    payload = json.loads(record.save(tmp_path / "g.json").read_text(encoding="utf-8"))
    assert "note" not in payload["steps"][0]
    assert "extra" not in payload["steps"][0]


def test_replay_rejects_a_record_that_does_not_match_the_engine():
    record = play_game({s: Plain(seed=5) for s in Side}, 5)
    # Corrupt the recorded metadata for one step.
    record.steps[3].decision_type = "not_a_real_decision"
    with pytest.raises(ValueError, match="does not match this engine"):
        record.replay()
    # Without the cross-check it still replays, since only the action is used.
    assert record.replay(strict=False).history == record.history


def test_replay_rejects_a_record_longer_than_the_game():
    record = play_game({s: Plain(seed=5) for s in Side}, 5)
    record.steps.append(Step(action="pass", player="USSR", decision_type="confirm"))
    with pytest.raises(ValueError, match="ended after"):
        record.replay(strict=False)


def test_record_from_game_recovers_a_hand_driven_game():
    game = Game(seed=15)
    rng = random.Random(15)
    while game.decision is not None:
        game.step(rng.choice(game.decision.options))

    record = record_from_game(game, note="driven by hand")
    assert record.history == game.history
    assert record.winner == (
        game.state.winner.label if game.state.winner is not None else "draw"
    )
    assert record.metadata["note"] == "driven by hand"
    # Rationales were never captured, so there are none to invent.
    assert not record.annotated()
    assert record.replay().history == game.history


def test_metadata_is_carried_through(tmp_path: Path):
    record = play_game(
        {s: Plain(seed=1) for s in Side}, 1, metadata={"model": "test", "run": 3}
    )
    reloaded = GameRecord.load(record.save(tmp_path / "g.json"))
    assert reloaded.metadata == {"model": "test", "run": 3}

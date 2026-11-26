"""Differential test: the checkpointed ``Game.clone`` against the replay reference.

``Game.replay_clone`` is the definition of a clone (seed + whole history replayed).
``Game.clone`` resumes from a replay checkpoint and replays only the tail. At every step
of mixed-policy games this checks that the two give the same game: every state field
(the RNG state included), history, the pending decision with all its options, both
players' observations, and the same outcome over the next ``FOLLOW`` steps under one
shared action sequence. It also checks that a clone is independent of its source, that
clones of clones agree, and that an edit made to a game's state from outside ``step``
is ignored by ``clone`` exactly as by a replay.

The default run is a quick sample. ``TS_CLONE_DIFF_GAMES=300`` (and optionally
``TS_CLONE_DIFF_SHARD=i/n`` to split the games over processes) runs the full check.
"""

from __future__ import annotations

import os
import random

import pytest

from twilight.baselines import GreedyAgent, RandomAgent, SafeRandomAgent
from twilight.engine import Game
from twilight.enums import Side
from twilight.observe import observe

GAMES = int(os.environ.get("TS_CLONE_DIFF_GAMES", "4"))
FOLLOW = int(os.environ.get("TS_CLONE_DIFF_FOLLOW", "6"))
SHARD = os.environ.get("TS_CLONE_DIFF_SHARD", "0/1")
MAX_STEPS = 3000


def snapshot(game: Game, observations: bool = True) -> tuple:
    """Everything a game consists of, in comparable form (optionally with both
    players' observations, which are a function of the rest)."""
    fields = {}
    for name, value in vars(game.state).items():
        if name == "rng":
            value = value.getstate()
        elif name == "log":
            value = [(e.turn, e.action_round, e.player, e.text) for e in value]
        elif name == "effects":
            value = {k: (e.card, e.owner, e.expires, e.data) for k, e in value.items()}
        elif name == "deferred":
            value = [(t.card, t.player, t.when, t.kind, t.not_before, t.data) for t in value]
        fields[name] = value
    d = game.decision
    decision = None if d is None else (
        d.type, d.player, d.prompt, tuple((a.kind, a.value, a.label) for a in d.options), d.context
    )
    observations = tuple(observe(game.state, side, d) for side in Side) if observations else ()
    config = (game.seed, game.optional_cards, game.handicap, game.strict_events)
    return fields, list(game.history), list(game.unimplemented), decision, observations, config


def assert_same(a: Game, b: Game, where: str, observations: bool = True) -> None:
    sa, sb = snapshot(a, observations), snapshot(b, observations)
    if sa == sb:
        return
    names = ("state", "history", "unimplemented", "decision", "observations", "config")
    for name, x, y in zip(names, sa, sb):
        if name == "state":
            diff = [k for k in set(x) | set(y) if x.get(k) != y.get(k)]
            assert not diff, f"{where}: state fields differ: {diff}"
        assert x == y, f"{where}: {name} differs"


def _policy(rng: random.Random, seed: int):
    kind = rng.choice(["random", "safe_random", "greedy", "greedy", "mixed"])
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "safe_random":
        return SafeRandomAgent(seed=seed)
    greedy = GreedyAgent(seed=seed)
    if kind == "greedy":
        return greedy
    rand = RandomAgent(seed=seed + 1)

    class Mixed:  # greedy with 30% random moves
        def act(self, game, decision):
            return rand.act(game, decision) if rng.random() < 0.3 else greedy.act(game, decision)

    return Mixed()


def _act(agent, game):
    r = agent.act(game, game.decision)
    return r[0] if isinstance(r, tuple) else r


def follow(fast: Game, ref: Game, rng: random.Random, steps: int, where: str) -> None:
    """Drive both copies with one action sequence and compare after every step."""
    for i in range(steps):
        if fast.decision is None or ref.decision is None:
            assert fast.decision is None and ref.decision is None, f"{where}: one copy ended"
            return
        key = rng.choice(fast.decision.options).key
        fast.step(key)
        ref.step(key)
        assert_same(fast, ref, f"{where} +{i + 1}", observations=(i == steps - 1))


def run_game(index: int) -> int:
    seed = 90_000 + index
    rng = random.Random(seed)
    game = Game(seed=seed, optional_cards=(index % 5 == 4))
    agents = {Side.USSR: _policy(rng, seed), Side.USA: _policy(rng, seed + 7)}
    steps = 0
    while game.decision is not None and steps < MAX_STEPS:
        where = f"game {index} (seed {seed}) step {steps}"
        before = snapshot(game, False)
        fast = game.clone()
        ref = game.replay_clone()
        assert_same(fast, ref, where)
        assert_same(fast, game, where + " (vs source)", observations=False)
        follow(fast, ref, random.Random(seed * 7919 + steps), FOLLOW, where)
        # clone of a clone that went on alone
        if steps % 3 == 0 and fast.decision is not None:
            assert_same(fast.clone(), ref.replay_clone(), where + " (clone of clone)", False)
        # an external edit is ignored by clone() as it is by a replay
        if steps % 5 == 0 and fast.decision is not None:
            st = fast.state
            st.rng.seed(steps)
            rng.shuffle(st.deck)
            if st.hands[0] and st.deck:
                st.hands[0][0], st.deck[0] = st.deck[0], st.hands[0][0]
            st.vp += 1
            assert_same(fast.clone(), fast.replay_clone(), where + " (edited state)", False)
        assert snapshot(game, False) == before, f"{where}: cloning changed the source"
        game.step(_act(agents[game.decision.player], game))
        steps += 1
    return steps


def _indices() -> list[int]:
    i, n = (int(x) for x in SHARD.split("/"))
    return [g for g in range(GAMES) if g % n == i]


@pytest.mark.parametrize("index", _indices())
def test_clone_matches_replay(index):
    assert run_game(index) > 0


def test_clone_of_finished_and_fresh_games():
    game = Game(seed=5)
    assert_same(game.clone(), game.replay_clone(), "fresh")
    rng = random.Random(5)
    while game.decision is not None:
        game.step(rng.choice(game.decision.options))
    assert_same(game.clone(), game.replay_clone(), "finished")


def test_state_copy_is_independent():
    game = Game(seed=11)
    rng = random.Random(11)
    for _ in range(120):
        game.step(rng.choice(game.decision.options))
    st = game.state
    cp = st.copy()
    assert set(vars(cp)) == set(vars(st))
    before = snapshot(game)[0]
    cp.influence[0][0] += 5
    cp.hands[0].append("x")
    cp.deck.clear()
    cp.rng.random()
    cp.headline[0] = "y"
    cp.log.append(None)
    cp.stages_in_deck.clear()
    assert snapshot(game)[0] == before

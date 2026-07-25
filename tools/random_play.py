"""Play games with random agents to shake out crashes and rule-invariant violations.

Usage::

    python tools/random_play.py --games 50
    python tools/random_play.py --games 1 --log        # print the game log
"""

from __future__ import annotations

import argparse
import random
import sys
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twilight import events  # noqa: E402
from twilight.data import CARDS, CHINA_CARD  # noqa: E402
from twilight.decisions import NUM_ACTIONS  # noqa: E402
from twilight.engine import Game  # noqa: E402
from twilight.enums import Side  # noqa: E402
from twilight.state import VP_LIMIT  # noqa: E402


def check_invariants(game: Game) -> list[str]:
    """Rule invariants that must hold at every decision point."""
    problems: list[str] = []
    s = game.state

    if not 1 <= s.defcon <= 5:
        problems.append(f"DEFCON out of range: {s.defcon}")
    if not -VP_LIMIT <= s.vp <= VP_LIMIT:
        problems.append(f"VP out of range: {s.vp}")
    for side in Side:
        if any(v < 0 for v in s.influence[side]):
            problems.append(f"negative influence for {side.label}")
        if not 0 <= s.space_race[side] <= 8:
            problems.append(f"space race out of range: {s.space_race[side]}")
        if not 0 <= s.military_ops[side] <= 5:
            problems.append(f"military ops out of range: {s.military_ops[side]}")
    if not 1 <= s.turn <= 10:
        problems.append(f"turn out of range: {s.turn}")

    if game.decision is not None:
        d = game.decision
        if not d.options:
            problems.append(f"decision {d.type} has no options")
        for action in d.options:
            if not 0 <= action.index < NUM_ACTIONS:
                problems.append(f"action {action.key} outside vocabulary")

    # No card may exist in two places at once.
    seen = Counter()
    for side in Side:
        seen.update(s.hands[side])
    seen.update(s.deck)
    seen.update(s.discard)
    seen.update(s.removed)
    duplicated = [name for name, n in seen.items() if n > 1]
    if duplicated:
        problems.append(f"cards in more than one place: {duplicated}")

    # Nor may any card vanish: a lost card silently shrinks the deck for good.
    located = set(seen) | set(s.effects) | set(s.transit)
    located |= {n for n in s.headline.values() if n is not None}
    if s.playing_card is not None:
        located.add(s.playing_card)
    expected = {
        name
        for name, c in CARDS.items()
        if name != CHINA_CARD
        and not c.optional
        and c.stage in game._stages_added
    }
    missing = expected - located
    if missing:
        problems.append(f"cards lost from the game: {sorted(missing)}")

    return problems


def play_one(seed: int, *, verbose: bool = False, check: bool = True) -> dict:
    game = Game(seed=seed)
    rng = random.Random(seed ^ 0x5EED)
    steps = 0
    decision_types: Counter[str] = Counter()

    while game.decision is not None:
        if check:
            problems = check_invariants(game)
            if problems:
                raise AssertionError(
                    f"invariant violated at step {steps} "
                    f"({game.decision.type}): {problems}"
                )
        decision_types[str(game.decision.type)] += 1
        choice = rng.choice(game.decision.options)
        game.step(choice)
        steps += 1
        if steps > 200_000:
            raise AssertionError("game failed to terminate")

    if verbose:
        for entry in game.state.log:
            print(entry)

    return {
        "seed": seed,
        "steps": steps,
        "turn": game.state.turn,
        "vp": game.state.vp,
        "winner": game.state.winner.label if game.state.winner is not None else "draw",
        "reason": game.state.win_reason.value if game.state.win_reason else "?",
        "decision_types": decision_types,
        "unimplemented": list(game.unimplemented),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", action="store_true", help="print the log of game 1")
    parser.add_argument("--no-check", action="store_true", help="skip invariant checks")
    args = parser.parse_args()

    done, total = events.coverage()
    print(f"event coverage: {done}/{total} cards implemented\n")

    results = []
    failures = 0
    all_types: Counter[str] = Counter()
    unimplemented: Counter[str] = Counter()

    for i in range(args.games):
        seed = args.seed + i
        try:
            r = play_one(seed, verbose=args.log and i == 0, check=not args.no_check)
        except Exception as exc:  # noqa: BLE001 - we want the whole picture
            failures += 1
            print(f"seed {seed}: FAILED {type(exc).__name__}: {exc}")
            if failures == 1:
                traceback.print_exc()
            continue
        results.append(r)
        all_types.update(r["decision_types"])
        unimplemented.update(r["unimplemented"])

    print(f"\ncompleted {len(results)}/{args.games} games, {failures} failures")
    if not results:
        return 1

    steps = [r["steps"] for r in results]
    print(f"steps per game: min {min(steps)}, median {sorted(steps)[len(steps)//2]}, max {max(steps)}")
    print("outcomes:", dict(Counter(f"{r['winner']}/{r['reason']}" for r in results)))
    print("final turn:", dict(sorted(Counter(r["turn"] for r in results).items())))

    print("\ndecision types encountered:")
    for name, n in all_types.most_common():
        print(f"  {n:>7}  {name}")

    if unimplemented:
        print(f"\nunimplemented events actually hit ({len(unimplemented)} distinct):")
        for name, n in unimplemented.most_common(20):
            print(f"  {n:>4}  {name}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

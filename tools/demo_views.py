"""Show both agent views of the same position, and the encoded shapes.

    python tools/demo_views.py --steps 60
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twilight import Side, TwilightStruggleEnv  # noqa: E402
from twilight.encode import COUNTRY_FEATURES, encode, flatten, observation_shapes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60, help="random steps to play first")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print("=" * 78)
    print("ENCODED OBSERVATION SHAPES (for building network inputs)")
    print("=" * 78)
    for key, shape in observation_shapes().items():
        print(f"  {key:<16} {shape}")
    print(f"\n  country feature columns ({len(COUNTRY_FEATURES)}):")
    for i in range(0, len(COUNTRY_FEATURES), 4):
        print("    " + ", ".join(COUNTRY_FEATURES[i : i + 4]))

    env = TwilightStruggleEnv(seed=args.seed, encode_observations=False)
    rng = random.Random(args.seed)
    info: dict = {}
    for _ in range(args.steps):
        if env.decision is None:
            break
        choice = rng.choice(env.decision.options)
        _, _, terminated, _, info = env.step(choice)
        if terminated:
            break

    print()
    print("=" * 78)
    print(f"TEXT VIEW after {env.steps} random steps (what an LLM agent receives)")
    print("=" * 78)
    print(env.text())

    if env.decision is not None:
        obs = env.observation_for(env.decision.player)
        encoded = encode(obs)
        flat = flatten(encoded)
        print()
        print("=" * 78)
        print("SAME POSITION, NUMERIC VIEW")
        print("=" * 78)
        print(f"  countries matrix : {encoded['countries'].shape}")
        print(f"  global vector    : {encoded['global'].shape}")
        print(f"  flat vector      : {flat.shape}, "
              f"range [{flat.min():.3f}, {flat.max():.3f}]")
        print(f"  legal actions    : {int(encoded['action_mask'].sum())} "
              f"of {encoded['action_mask'].size}")
        print(f"  to move          : {info.get('to_move')}")
        print(f"  VP (USSR view)   : {env.observation_for(Side.USSR).vp:+d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

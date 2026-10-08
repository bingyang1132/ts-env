"""Baseline agents and a head-to-head runner (moved to :mod:`twilight.baselines`).

The agents now ship inside the package, so ``pip install`` makes them importable::

    from twilight.baselines import GreedyAgent, SafeRandomAgent, AGENTS

This file only re-exports that module, so the older usage with ``examples/`` on
``sys.path`` (``from baselines import GreedyAgent``) and the command line below keep
working unchanged::

    python examples/baselines.py --games 40 --ussr greedy --usa safe_random
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from a source checkout without installing: make ``twilight`` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twilight.baselines import *  # noqa: E402,F403
from twilight.baselines import (  # noqa: E402,F401
    _DEFCON_DEGRADING,
    _DEFCON_LABEL,
    _defcon_in_label,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())

"""A Twilight Struggle environment for testing and training agents.

The board topology and card database are extracted from the shipped game's own Lua
files (see ``tools/extract_lua.py``), so country stability values, battleground flags,
adjacency and card statistics are authoritative rather than hand-transcribed.

Quick start::

    from twilight import TwilightStruggleEnv

    env = TwilightStruggleEnv(seed=0, encode_observations=False)
    obs, info = env.reset()
    while not env.decision is None:
        print(env.text())                     # text view for a language model
        action = pick(info["legal_actions"])
        obs, reward, done, truncated, info = env.step(action)

Or drive the engine directly for full control::

    from twilight import Game
    game = Game(seed=0)
    while game.decision is not None:
        game.step(choose(game.decision))
"""

from .decisions import ACTION_VOCAB, Action, Decision, DecisionType, NUM_ACTIONS
from .engine import Game
from .enums import OpsUse, Phase, Region, Side, Stage, WinReason
from .env import TwilightStruggleEnv
from .observe import Observation, observe
from .record import GameRecord, Step, play_game, record_from_game
from .render import render
from .state import GameState

__version__ = "0.1.0"

__all__ = [
    "ACTION_VOCAB",
    "Action",
    "Decision",
    "DecisionType",
    "Game",
    "GameRecord",
    "GameState",
    "NUM_ACTIONS",
    "Observation",
    "OpsUse",
    "Phase",
    "Region",
    "Side",
    "Stage",
    "Step",
    "TwilightStruggleEnv",
    "WinReason",
    "__version__",
    "observe",
    "play_game",
    "record_from_game",
    "render",
]

"""The seven scoring cards.

All of them delegate to :meth:`twilight.engine.Game.score_region`, which applies the
net VP swing and handles Europe control ending the game outright.
"""

from __future__ import annotations

from ..enums import Region
from . import register
from .helpers import nothing

_SCORING_CARDS = {
    "Europe Scoring": Region.EUROPE,
    "Asia Scoring": Region.ASIA,
    "Middle East Scoring": Region.MIDDLE_EAST,
    "Central America Scoring": Region.CENTRAL_AMERICA,
    "Southeast Asia Scoring": Region.SOUTHEAST_ASIA,
    "Africa Scoring": Region.AFRICA,
    "South America Scoring": Region.SOUTH_AMERICA,
}


def _make(region: Region):
    def handler(game, ctx):
        game.score_region(region)
        return (yield from nothing())

    handler.__name__ = f"score_{region.name.lower()}"
    handler.__doc__ = f"Score the {region} region for both players."
    return handler


for _name, _region in _SCORING_CARDS.items():
    register(_name)(_make(_region))

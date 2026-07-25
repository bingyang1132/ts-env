"""Enumerations shared across the engine."""

from __future__ import annotations

from enum import Enum, IntEnum


class Side(IntEnum):
    """The two players.

    Deliberately an :class:`~enum.IntEnum` so it can index the paired arrays in
    :class:`~twilight.state.GameState` (``influence[side]``, ``military_ops[side]``).
    Because ``USSR == 0``, plain truthiness would read a USSR win as "no winner", so
    ``__bool__`` is overridden: always test ``side is None`` explicitly.
    """

    USSR = 0
    USA = 1

    def __bool__(self) -> bool:
        return True

    @property
    def opponent(self) -> "Side":
        return Side.USA if self is Side.USSR else Side.USSR

    @property
    def label(self) -> str:
        return "USSR" if self is Side.USSR else "USA"

    @classmethod
    def from_label(cls, label: str) -> "Side":
        if label == "USSR":
            return cls.USSR
        if label == "USA":
            return cls.USA
        raise ValueError(f"not a player side: {label!r}")


NEUTRAL = "Neutral"


class Region(str, Enum):
    """Scoring regions.

    ``WESTERN_EUROPE`` / ``EASTERN_EUROPE`` and ``SOUTHEAST_ASIA`` are scored by their
    own cards while also rolling up into ``EUROPE`` / ``ASIA``. Austria and Finland
    belong to both halves of Europe.
    """

    EUROPE = "Europe"
    WESTERN_EUROPE = "Western Europe"
    EASTERN_EUROPE = "Eastern Europe"
    ASIA = "Asia"
    SOUTHEAST_ASIA = "Southeast Asia"
    MIDDLE_EAST = "Middle East"
    AFRICA = "Africa"
    CENTRAL_AMERICA = "Central America"
    SOUTH_AMERICA = "South America"

    def __str__(self) -> str:  # keeps f-strings readable
        return self.value


#: Victory points for presence / domination / control in each region.
#: Europe control ends the game immediately, so its control value is a sentinel.
AUTO_VICTORY = 1_000

SCORING_VALUES: dict[Region, tuple[int, int, int]] = {
    Region.EUROPE: (3, 7, AUTO_VICTORY),
    Region.ASIA: (3, 7, 9),
    Region.MIDDLE_EAST: (3, 5, 7),
    Region.AFRICA: (1, 4, 6),
    Region.CENTRAL_AMERICA: (1, 3, 5),
    Region.SOUTH_AMERICA: (2, 5, 6),
    Region.SOUTHEAST_ASIA: (0, 0, 0),  # scored per-country, not by presence tier
}

#: Regions that have a scoring card, in the order they are scored at the end of the
#: game. Western and Eastern Europe are never scored on their own -- they exist so that
#: cards can refer to one half of Europe -- so they are absent here and from
#: :data:`SCORING_VALUES`.
SCORING_REGIONS: tuple[Region, ...] = (
    Region.EUROPE,
    Region.ASIA,
    Region.MIDDLE_EAST,
    Region.AFRICA,
    Region.CENTRAL_AMERICA,
    Region.SOUTH_AMERICA,
    Region.SOUTHEAST_ASIA,
)

#: Regions whose scoring card awards +1 VP per controlled country adjacent to the
#: enemy superpower. Only these three cards print that clause.
ADJACENCY_BONUS_REGIONS = frozenset(
    {Region.EUROPE, Region.ASIA, Region.CENTRAL_AMERICA}
)

#: Southeast Asia is scored country by country instead of by presence tier:
#: 1 VP per controlled country, and Thailand is worth 2.
SOUTHEAST_ASIA_COUNTRY_VP: dict[str, int] = {"Thailand": 2}


class Phase(str, Enum):
    """Where in the turn structure the game currently is."""

    SETUP = "setup"
    HEADLINE = "headline"
    ACTION_ROUND = "action_round"
    TURN_END = "turn_end"
    GAME_OVER = "game_over"


class CardType(str, Enum):
    EVENT = "Event"
    SCORING = "Scoring"


class Stage(str, Enum):
    EARLY_WAR = "Early War"
    MID_WAR = "Mid War"
    LATE_WAR = "Late War"


class OpsUse(str, Enum):
    """What a player chose to do with a card."""

    EVENT = "event"
    INFLUENCE = "influence"
    COUP = "coup"
    REALIGN = "realign"
    SPACE = "space"
    DISCARD = "discard"


class WinReason(str, Enum):
    VICTORY_POINTS = "victory_points"          # VP track reached +/-20
    EUROPE_CONTROL = "europe_control"          # controlled Europe when Europe Scoring fired
    DEFCON = "defcon"                          # dropped DEFCON to 1
    MILITARY_OPS = "military_ops"              # failed required military operations at DEFCON 2 ... see rules
    HELD_SCORING_CARD = "held_scoring_card"    # ended a turn holding a scoring card
    CUBAN_MISSILE_CRISIS = "cuban_missile_crisis"  # couped while the crisis was in play
    WARGAMES = "wargames"                      # Wargames event
    FINAL_SCORING = "final_scoring"            # highest VP after turn 10
    DRAW = "draw"

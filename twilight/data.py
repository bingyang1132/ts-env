"""Static game data, loaded once from the JSON extracted out of the game's Lua files.

Everything here is immutable and shared by all games in the process: the board
topology, and the card definitions. Regenerate the JSON with
``python tools/extract_lua.py`` if the game install is updated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .enums import CardType, Region, Side, Stage

DATA_DIR = Path(__file__).resolve().parent / "data"

#: Internal helper cards the game's own AI uses to stand in for a generic N-op card.
#: They are not part of any physical deck.
_AI_PROXY_PREFIXES = ("1 Op AI Proxy", "2 Op AI Proxy", "3 Op AI Proxy", "4 Op AI Proxy")

#: The China Card is handled separately: it is never shuffled into the deck, it passes
#: between players, and it cannot be a headline.
CHINA_CARD = "The China Card"


@dataclass(frozen=True, slots=True)
class Country:
    """A space on the board that influence can occupy."""

    name: str
    index: int
    stability: int
    battleground: bool
    regions: tuple[Region, ...]
    adjacent: tuple[str, ...]
    #: True for the USA / USSR spaces themselves, which hold no influence.
    superpower: bool = False

    @property
    def region(self) -> Region:
        """The most specific region this country scores in."""
        return self.regions[0]

    def in_region(self, region: Region) -> bool:
        return region in self.regions


@dataclass(frozen=True, slots=True)
class Card:
    """A card in the deck."""

    name: str
    number: int
    type: CardType
    stage: Stage
    ops: int
    #: Which player's event this is; ``None`` for neutral events playable by either.
    side: Side | None
    scoring_region: Region | None
    optional: bool
    #: Removed from the game after its event resolves, rather than discarded.
    remove_on_event: bool
    #: Scoring cards must be played the turn they are drawn.
    may_be_held: bool
    can_headline: bool
    resolve_headline_first: bool
    event_text: str
    #: Verbatim effect-function names from the game's Lua, kept as the implementation
    #: spec and used by the registry to flag cards with no handler.
    effect_spec: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def is_scoring(self) -> bool:
        return self.type is CardType.SCORING

    def event_belongs_to(self, side: Side) -> bool:
        """Whether *side* triggers this event when playing the card for its event."""
        return self.side is None or self.side is side

    def __str__(self) -> str:
        return self.name


def _load_json(name: str) -> dict:
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing -- run `python tools/extract_lua.py` to regenerate "
            "the database from the game install."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _build_countries() -> tuple[dict[str, Country], tuple[str, ...]]:
    raw = _load_json("map.json")
    countries: dict[str, Country] = {}

    for name, entry in raw.items():
        # The Chinese Civil War space only exists in the Turn Zero variant.
        if entry["chinese_civil_war"]:
            continue
        countries[name] = Country(
            name=name,
            index=entry["index"],
            stability=entry["stability"] or 0,
            battleground=entry["battleground"],
            regions=tuple(Region(r) for r in entry["regions"]),
            adjacent=tuple(
                a for a in entry["adjacent"] if not raw[a]["chinese_civil_war"]
            ),
            superpower=entry["superpower"],
        )

    # A stable, contiguous ordering for observation vectors: playable countries first
    # (grouped by region so nearby columns are related), superpowers excluded since
    # they never hold influence.
    playable = [c for c in countries.values() if not c.superpower]
    order = tuple(
        c.name
        for c in sorted(playable, key=lambda c: (c.region.value, -c.stability, c.name))
    )
    return countries, order


COUNTRIES, COUNTRY_ORDER = _build_countries()

#: Position of each placeable country in observation vectors.
COUNTRY_SLOT: dict[str, int] = {name: i for i, name in enumerate(COUNTRY_ORDER)}
NUM_COUNTRIES = len(COUNTRY_ORDER)

#: Countries in each region, in observation order.
REGION_COUNTRIES: dict[Region, tuple[str, ...]] = {
    region: tuple(n for n in COUNTRY_ORDER if COUNTRIES[n].in_region(region))
    for region in Region
}

#: Battlegrounds in each region.
REGION_BATTLEGROUNDS: dict[Region, tuple[str, ...]] = {
    region: tuple(n for n in names if COUNTRIES[n].battleground)
    for region, names in REGION_COUNTRIES.items()
}

SUPERPOWER_SPACE: dict[Side, str] = {Side.USSR: "USSR", Side.USA: "USA"}

#: Countries touching the enemy superpower, which score a bonus when controlled.
ADJACENT_TO_SUPERPOWER: dict[Side, frozenset[str]] = {
    side: frozenset(COUNTRIES[SUPERPOWER_SPACE[side]].adjacent) for side in Side
}


def _build_cards() -> dict[str, Card]:
    raw = _load_json("cards.json")
    cards: dict[str, Card] = {}

    for name, entry in raw.items():
        if entry["source"] != "base" or name in _AI_PROXY_PREFIXES:
            continue
        side = None if entry["side"] == "Neutral" else Side.from_label(entry["side"])
        cards[name] = Card(
            name=name,
            number=entry["number"],
            type=CardType(entry["type"]),
            stage=Stage(entry["stage"]),
            ops=entry["ops"],
            side=side,
            scoring_region=Region(entry["scoring_region"]) if entry["scoring_region"] else None,
            optional=entry["optional"],
            remove_on_event=entry["remove_on_event"],
            may_be_held=entry["may_be_held"],
            can_headline=entry["can_headline"],
            resolve_headline_first=entry["resolve_headline_first"],
            event_text=entry["event_text"],
            effect_spec=entry["effect_spec"],
        )
    return cards


CARDS: dict[str, Card] = _build_cards()

#: Cards in a stable order (by printed number) for observation vectors.
CARD_ORDER: tuple[str, ...] = tuple(
    c.name for c in sorted(CARDS.values(), key=lambda c: c.number)
)
CARD_SLOT: dict[str, int] = {name: i for i, name in enumerate(CARD_ORDER)}
NUM_CARDS = len(CARD_ORDER)


def deck_for_stage(stage: Stage, *, optional_cards: bool = False) -> list[str]:
    """The cards added to the draw pile when *stage* begins.

    The China Card is excluded: it starts in the USSR player's hands and is passed
    between players rather than drawn.
    """
    return sorted(
        (
            name
            for name, card in CARDS.items()
            if card.stage is stage
            and name != CHINA_CARD
            and (optional_cards or not card.optional)
        ),
        key=lambda n: CARDS[n].number,
    )


def card(name: str) -> Card:
    try:
        return CARDS[name]
    except KeyError:
        raise KeyError(f"no such card: {name!r}") from None


def country(name: str) -> Country:
    try:
        return COUNTRIES[name]
    except KeyError:
        raise KeyError(f"no such country: {name!r}") from None


__all__ = [
    "ADJACENT_TO_SUPERPOWER",
    "CARDS",
    "CARD_ORDER",
    "CARD_SLOT",
    "CHINA_CARD",
    "COUNTRIES",
    "COUNTRY_ORDER",
    "COUNTRY_SLOT",
    "Card",
    "Country",
    "NUM_CARDS",
    "NUM_COUNTRIES",
    "REGION_BATTLEGROUNDS",
    "REGION_COUNTRIES",
    "SUPERPOWER_SPACE",
    "card",
    "country",
    "deck_for_stage",
]

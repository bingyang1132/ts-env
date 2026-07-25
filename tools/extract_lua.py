"""Extract the Twilight Struggle map and card database from the shipped Lua files.

The game's Unity build keeps its data as plain Lua under
``TwilightStruggle_Data/StreamingAssets/Lua``. That data is authoritative -- country
stability values, battleground flags, adjacency, and every card's number, operations
value, associated side and event text -- so the environment reads from it rather than
from hand-transcribed tables.

Card *effects* in the Lua are only names of C# functions (e.g. ``"ScoreAsia"``). The
extractor preserves those names verbatim: they are the specification each event
implementation in ``twilight.events`` is written against, and the loader cross-checks
that every card either has a registered handler or is explicitly marked unimplemented.

Usage::

    python tools/extract_lua.py --game-dir "D:/ExcellentWorks/Twilight.Struggle"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from luaparse import parse_assignments, to_jsonable  # noqa: E402

LUA_SUBDIR = Path("TwilightStruggle_Data/StreamingAssets/Lua")

# Only "base" feeds the standard deck. The promo pack and the Turn Zero variant
# (which adds "Statecraft" cards and a pre-game crisis phase) are extracted so the
# data is available, but they are excluded from normal play.
CARD_FILES = {
    "base": "twilight_cards.lua",
    "promo": "twilight_promos.lua",
    "turnzero": "twilight_turnzero.lua",
}

# Regions as named in the Lua, and the super-region each one rolls up into.
# Finland and Austria carry region "Europe": they score for both halves.
SUB_TO_SUPER = {
    "Western Europe": "Europe",
    "Eastern Europe": "Europe",
    "Europe": "Europe",
    "Southeast Asia": "Asia",
    "Asia": "Asia",
    "Middle East": "Middle East",
    "Africa": "Africa",
    "Central America": "Central America",
    "South America": "South America",
}


def _read(path: Path) -> str:
    # The files are ASCII in practice; be permissive about a stray BOM.
    return path.read_text(encoding="utf-8-sig")


def extract_map(lua_dir: Path) -> dict:
    src = _read(lua_dir / "twilight_map.lua")
    countries: dict[str, dict] = {}

    for key, value in parse_assignments(src, "g_twilight_map"):
        raw = to_jsonable(value)
        region = raw.get("region")
        entry = {
            "name": raw["country_name"],
            "index": raw["country_index"],
            "region": region,
            "regions": _regions_for(region),
            "stability": raw.get("stability"),
            "battleground": bool(raw.get("battleground", False)),
            "superpower": bool(raw.get("superpower", False)),
            "chinese_civil_war": bool(raw.get("chinese_civil_war", False)),
            "adjacent": list(raw.get("adjacent_countries", [])),
        }
        if entry["name"] != key:
            raise ValueError(f"map key {key!r} disagrees with country_name {entry['name']!r}")
        countries[key] = entry

    _validate_map(countries)
    return countries


def _regions_for(region: str | None) -> list[str]:
    """All scoring regions a country belongs to, most specific first.

    A country in "Western Europe" also scores for "Europe"; the two countries
    whose region is literally "Europe" score for both halves.
    """
    if region is None:
        return []
    if region == "Europe":
        return ["Western Europe", "Eastern Europe", "Europe"]
    super_region = SUB_TO_SUPER[region]
    return [region] if super_region == region else [region, super_region]


def _validate_map(countries: dict[str, dict]) -> None:
    problems: list[str] = []

    indices: dict[int, str] = {}
    for name, c in countries.items():
        if c["index"] in indices:
            problems.append(f"duplicate country_index {c['index']}: {indices[c['index']]} / {name}")
        indices[c["index"]] = name

        for neighbour in c["adjacent"]:
            if neighbour not in countries:
                problems.append(f"{name} is adjacent to unknown country {neighbour!r}")
            elif name not in countries[neighbour]["adjacent"]:
                problems.append(f"adjacency not symmetric: {name} -> {neighbour}")

        placeable = not c["superpower"] and not c["chinese_civil_war"]
        if placeable and c["region"] is None:
            problems.append(f"{name} has no region")
        if placeable and c["stability"] is None:
            problems.append(f"{name} has no stability value")

    if problems:
        raise ValueError("map validation failed:\n  " + "\n  ".join(problems))


def extract_cards(lua_dir: Path) -> dict:
    cards: dict[str, dict] = {}

    for source, filename in CARD_FILES.items():
        src = _read(lua_dir / filename)
        for key, value in parse_assignments(src, "g_twilight_cards"):
            raw = to_jsonable(value)
            entry = {
                "name": raw["card_name"],
                "number": raw["card_number"],
                "type": raw["card_type"],
                "stage": raw.get("stage"),  # Turn Zero cards are outside the three stages
                "source": source,
                "ops": raw.get("operations_points", 0),
                "side": raw.get("event_owner", "Neutral"),
                "scoring_region": raw.get("scoring_region"),
                "optional": bool(raw.get("optional_card", False)),
                "remove_on_event": bool(raw.get("remove_if_used_as_event", False)),
                "may_be_held": raw.get("may_be_held", True),
                "can_headline": raw.get("can_headline_card", True),
                "resolve_headline_first": bool(raw.get("resolve_headline_first", False)),
                "pending_defcon": raw.get("pending_defcon_level"),
                "event_text": raw.get("event_text", ""),
                # Verbatim C# effect-function names -- the spec for each event.
                "effect_spec": _effect_spec(raw),
            }
            # Turn Zero "Statecraft" cards modify a pre-game crisis die roll.
            for extra in (
                "statecraft_modifier",
                "statecraft_cancel_opponent",
                "statecraft_return_to_hand",
                "discard_after_play",
            ):
                if extra in raw:
                    entry[extra] = raw[extra]
            if entry["name"] != key:
                raise ValueError(f"card key {key!r} disagrees with card_name {entry['name']!r}")
            if key in cards:
                raise ValueError(f"card {key!r} defined in more than one file")
            cards[key] = entry

    _validate_cards(cards)
    return cards


def _effect_spec(raw: dict) -> dict:
    """Collect the raw effect declarations without interpreting them."""
    spec: dict = {}
    for field in (
        "event_effect",
        "event_effects",
        "crisis_effect",
        "crisis_effects",
        "continuous_effects",
        "triggered_effects",
        "global_continuous_effects",
        "cardinplay_ability",
        "cardinplay_abilities",
    ):
        if field in raw:
            spec[field] = raw[field]
    return spec


def _validate_cards(cards: dict[str, dict]) -> None:
    problems: list[str] = []

    by_number: dict[tuple[str, int], list[str]] = {}
    for name, c in cards.items():
        by_number.setdefault((c["source"], c["number"]), []).append(name)
        if c["side"] not in ("USA", "USSR", "Neutral"):
            problems.append(f"{name} has unknown side {c['side']!r}")

        # The stage / operations invariants only hold for the standard deck.
        if c["source"] != "base":
            continue
        if c["type"] == "Scoring":
            if c["scoring_region"] is None:
                problems.append(f"scoring card {name} has no scoring_region")
            if c["may_be_held"]:
                problems.append(f"scoring card {name} is holdable")
        elif c["ops"] <= 0:
            problems.append(f"event card {name} has no operations value")
        if c["stage"] not in ("Early War", "Mid War", "Late War"):
            problems.append(f"{name} has unknown stage {c['stage']!r}")

    for (source, number), names in sorted(by_number.items()):
        if len(names) > 1:
            problems.append(f"{source} card number {number} used by {names}")

    if problems:
        raise ValueError("card validation failed:\n  " + "\n  ".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="root of the Twilight Struggle install (contains TwilightStruggle_Data)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "twilight" / "data",
        help="directory to write map.json and cards.json into",
    )
    args = parser.parse_args()

    lua_dir = args.game_dir / LUA_SUBDIR
    if not lua_dir.is_dir():
        parser.error(f"no Lua database under {lua_dir}")

    countries = extract_map(lua_dir)
    cards = extract_cards(lua_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (("map.json", countries), ("cards.json", cards)):
        path = args.out_dir / filename
        path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(payload)} entries)")

    placeable = [c for c in countries.values() if not c["superpower"] and not c["chinese_civil_war"]]
    battlegrounds = [c for c in placeable if c["battleground"]]
    print(f"  {len(placeable)} placeable countries, {len(battlegrounds)} battlegrounds")
    for stage in ("Early War", "Mid War", "Late War"):
        in_stage = [c for c in cards.values() if c["stage"] == stage and c["source"] == "base"]
        optional = sum(1 for c in in_stage if c["optional"])
        print(f"  {stage}: {len(in_stage)} cards ({optional} optional)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

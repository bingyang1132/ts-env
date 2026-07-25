"""Print a summary of the extracted database for eyeballing against the real board."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "twilight" / "data"

countries = json.loads((DATA / "map.json").read_text(encoding="utf-8"))
cards = json.loads((DATA / "cards.json").read_text(encoding="utf-8"))

print("=" * 70)
print("COUNTRIES")
print("=" * 70)
special = {n: c for n, c in countries.items() if c["superpower"] or c["chinese_civil_war"]}
print(f"total entries: {len(countries)}   special: {sorted(special)}")

by_region: dict[str, list[str]] = defaultdict(list)
for name, c in countries.items():
    if name in special:
        continue
    by_region[c["region"]].append(name)

for region in sorted(by_region):
    names = sorted(by_region[region])
    bg = sorted(n for n in names if countries[n]["battleground"])
    print(f"\n{region}: {len(names)} countries, {len(bg)} battleground")
    print(f"  battleground: {', '.join(bg) if bg else '-'}")
    print(f"  other:        {', '.join(n for n in names if n not in set(bg))}")

playable = [n for n in countries if n not in special]
print(f"\nplayable countries: {len(playable)}")
print(f"battlegrounds total: {sum(countries[n]['battleground'] for n in playable)}")

print("\nsuper-region rollups (per `regions` field):")
region_members: dict[str, set[str]] = defaultdict(set)
for name in playable:
    for r in countries[name]["regions"]:
        region_members[r].add(name)
for r in sorted(region_members):
    members = region_members[r]
    bgs = sum(countries[n]["battleground"] for n in members)
    print(f"  {r:<16} {len(members):>3} countries, {bgs:>2} battlegrounds")

print("\nstability distribution:", dict(sorted(Counter(
    countries[n]["stability"] for n in playable).items())))

adjacent_to_us = sorted(countries["USA"]["adjacent"])
adjacent_to_ussr = sorted(countries["USSR"]["adjacent"])
print(f"\nadjacent to USA:  {adjacent_to_us}")
print(f"adjacent to USSR: {adjacent_to_ussr}")

print()
print("=" * 70)
print("CARDS")
print("=" * 70)
base = {n: c for n, c in cards.items() if c["source"] == "base"}
print(f"base deck entries: {len(base)}")
numbers = sorted(c["number"] for c in base.values())
print(f"card numbers: {numbers[0]}..{numbers[-1]}  contiguous={numbers == list(range(numbers[0], numbers[-1] + 1))}")

for stage in ("Early War", "Mid War", "Late War"):
    in_stage = sorted((c["number"], n) for n, c in base.items() if c["stage"] == stage)
    print(f"\n{stage}: {len(in_stage)}")
    for num, name in in_stage:
        c = base[name]
        flags = []
        if c["optional"]:
            flags.append("OPTIONAL")
        if c["type"] == "Scoring":
            flags.append("scoring:" + c["scoring_region"])
        if c["remove_on_event"]:
            flags.append("remove*")
        if not c["may_be_held"]:
            flags.append("must-play")
        if not c["can_headline"]:
            flags.append("no-headline")
        print(f"  {num:>3} {c['side']:<7} {c['ops']}op  {name:<34} {' '.join(flags)}")

print("\noptional cards:", sorted(n for n, c in base.items() if c["optional"]))
print("scoring cards:", sorted((c["number"], n) for n, c in base.items() if c["type"] == "Scoring"))
print("\nnon-base sources:", dict(Counter(
    c["source"] for c in cards.values() if c["source"] != "base")))

# Every distinct effect-function name, with how many cards use it. This is the
# work list for the event implementations.
usage: Counter[str] = Counter()


def walk(node, out: Counter) -> None:
    if isinstance(node, list):
        if node and isinstance(node[0], str):
            out[node[0]] += 1
            return
        for item in node:
            walk(item, out)
    elif isinstance(node, dict):
        for value in node.values():
            walk(value, out)


for c in base.values():
    walk(c["effect_spec"], usage)

print(f"\ndistinct effect functions referenced by base cards: {len(usage)}")
for fn, n in usage.most_common():
    print(f"  {n:>3}  {fn}")

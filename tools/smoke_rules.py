"""Quick check that the data and rules layers agree with known board facts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twilight import data, rules  # noqa: E402
from twilight.enums import Region, Side, Stage  # noqa: E402
from twilight.state import GameState  # noqa: E402

failures: list[str] = []


def check(label: str, got, want) -> None:
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
    print(f"{'ok  ' if got == want else 'FAIL'} {label}: {got!r}")


print("--- data ---")
check("placeable countries", data.NUM_COUNTRIES, 84)
check("cards in deck data", data.NUM_CARDS, 110)
# Early War is cards 1-35 plus Defectors (#103) = 36, less the China Card which the
# USSR holds from the start rather than drawing.
check("early war deck (no optionals, no China Card)",
      len(data.deck_for_stage(Stage.EARLY_WAR)), 35)
check("Defectors is an Early War card", data.card("Defectors").stage, Stage.EARLY_WAR)
check("China Card not in any stage deck",
      any(data.CHINA_CARD in data.deck_for_stage(s) for s in Stage), False)
check("UK stability", data.country("UK").stability, 5)
check("Italy is battleground", data.country("Italy").battleground, True)
check("Austria in both Europes", set(data.country("Austria").regions),
      {Region.WESTERN_EUROPE, Region.EASTERN_EUROPE, Region.EUROPE})
check("Thailand regions", set(data.country("Thailand").regions),
      {Region.SOUTHEAST_ASIA, Region.ASIA})
check("Europe country count", len(data.REGION_COUNTRIES[Region.EUROPE]), 21)
check("Asia battlegrounds", len(data.REGION_BATTLEGROUNDS[Region.ASIA]), 6)

print("\n--- control ---")
s = GameState()
s.set_inf(Side.USSR, "Italy", 2)  # Italy stability 2
check("2 inf vs 0 in stability-2 controls", rules.controls(s, Side.USSR, "Italy"), True)
s.set_inf(Side.USA, "Italy", 1)
check("2 vs 1 in stability-2 does not control", rules.controls(s, Side.USSR, "Italy"), False)
check("controller is nobody", rules.controller(s, "Italy"), None)
s.set_inf(Side.USSR, "Italy", 3)
check("3 vs 1 in stability-2 controls", rules.controls(s, Side.USSR, "Italy"), True)

print("\n--- influence placement ---")
s2 = GameState()
check("can place next to homeland (Poland)", rules.can_place_influence(s2, Side.USSR, "Poland"), True)
check("cannot place in far country (Chile)", rules.can_place_influence(s2, Side.USSR, "Chile"), False)
check("cannot place into superpower", rules.can_place_influence(s2, Side.USSR, "USA"), False)
s2.set_inf(Side.USA, "Italy", 4)  # US controls Italy (stability 2)
check("cost into opponent-controlled is 2", rules.influence_cost(s2, Side.USSR, "Italy"), 2)
check("cost into neutral is 1", rules.influence_cost(s2, Side.USSR, "Poland"), 1)

print("\n--- DEFCON restrictions ---")
s3 = GameState()
s3.set_inf(Side.USA, "Italy", 1)
s3.set_inf(Side.USA, "Iran", 1)
s3.set_inf(Side.USA, "Angola", 1)
s3.defcon = 5
check("defcon 5 allows Europe coup", rules.can_coup(s3, Side.USSR, "Italy"), True)
s3.defcon = 4
check("defcon 4 blocks Europe coup", rules.can_coup(s3, Side.USSR, "Italy"), False)
check("defcon 4 allows Iran coup", rules.can_coup(s3, Side.USSR, "Iran"), True)
s3.defcon = 2
check("defcon 2 blocks Iran coup", rules.can_coup(s3, Side.USSR, "Iran"), False)
check("defcon 2 allows Africa coup", rules.can_coup(s3, Side.USSR, "Angola"), True)
check("cannot coup where opponent has nothing", rules.can_coup(s3, Side.USSR, "Sudan"), False)

print("\n--- coup arithmetic ---")
s4 = GameState()
s4.set_inf(Side.USA, "Iran", 3)  # Iran stability 2, battleground
r = rules.resolve_coup(s4, Side.USSR, "Iran", ops=4, roll=3)
check("total is roll+ops", r.total, 7)
check("required is 2x stability", r.required, 4)
check("removes surplus from opponent", r.removed, 3)
check("places remainder", r.placed, 0)
check("milops credited", s4.military_ops[Side.USSR], 4)
check("battleground coup drops defcon", s4.defcon, 4)

s5 = GameState()
s5.set_inf(Side.USA, "Panama", 1)  # stability 2
r5 = rules.resolve_coup(s5, Side.USSR, "Panama", ops=4, roll=5)
check("surplus beyond removal is placed", (r5.removed, r5.placed), (1, 4))

print("\n--- region scoring ---")
s6 = GameState()
# USSR controls one non-battleground in Africa -> presence only.
s6.set_inf(Side.USSR, "Sudan", 5)
check("africa presence for USSR", rules.region_vp(s6, Region.AFRICA, Side.USSR), 1)
check("africa nothing for USA", rules.region_vp(s6, Region.AFRICA, Side.USA), 0)
# Add a battleground: more countries and battlegrounds, one of each -> domination.
s6.set_inf(Side.USSR, "Nigeria", 5)
check("africa tier for USSR", rules.region_status(s6, Region.AFRICA).tier(Side.USSR), "domination")
check("africa domination + 1 bg bonus", rules.region_vp(s6, Region.AFRICA, Side.USSR), 4 + 1)

print("\n--- southeast asia scoring ---")
s7 = GameState()
s7.set_inf(Side.USSR, "Thailand", 5)
s7.set_inf(Side.USSR, "Vietnam", 5)
check("SE Asia counts Thailand double", rules.region_vp(s7, Region.SOUTHEAST_ASIA, Side.USSR), 3)

print("\n--- europe auto-victory ---")
s8 = GameState()
for name in data.REGION_COUNTRIES[Region.EUROPE]:
    s8.set_inf(Side.USSR, name, 9)
ussr_vp, usa_vp, auto = rules.score_region(s8, Region.EUROPE)
check("controlling Europe wins outright", auto, Side.USSR)

print()
if failures:
    print(f"{len(failures)} FAILURES:")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("all rules checks passed")

"""Pure rules functions: control, region scoring, and the three operations.

Nothing here yields decisions or mutates flow -- these are the primitives the engine
and the card events are built out of. Keeping them side-effect-light (they mutate the
board but never ask the player anything) makes them straightforward to unit test
against known positions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import (
    ADJACENT_TO_SUPERPOWER,
    COUNTRIES,
    COUNTRY_ORDER,
    COUNTRY_SLOT,
    REGION_BATTLEGROUNDS,
    REGION_COUNTRIES,
    SUPERPOWER_SPACE,
    country,
)
from .enums import (
    ADJACENCY_BONUS_REGIONS,
    AUTO_VICTORY,
    SCORING_VALUES,
    SOUTHEAST_ASIA_COUNTRY_VP,
    Region,
    Side,
)
from .state import GameState

# --------------------------------------------------------------------------- #
# Control
# --------------------------------------------------------------------------- #


#: Both sides in definition order (iterating the enum class itself is slow).
SIDES = (Side.USSR, Side.USA)


def controls(state: GameState, side: Side, name: str) -> bool:
    """Whether *side* controls *name*.

    A player controls a country when their influence there exceeds the opponent's by
    at least the country's stability value. With no influence present, nobody controls.
    """
    # Same as ``state.inf(side, name) - state.inf(side.opponent, name)``, inlined: this
    # is the engine's hottest function.
    slot = COUNTRY_SLOT[name]
    influence = state.influence
    mine = influence[side][slot]
    if mine == 0:
        return False
    return mine - influence[1 - side][slot] >= COUNTRIES[name].stability


def controller(state: GameState, name: str) -> Side | None:
    for side in SIDES:
        if controls(state, side, name):
            return side
    return None


def is_controlled_by_superpower(state: GameState, side: Side, name: str) -> bool:
    """Treat a superpower's own space as permanently controlled by its owner.

    Needed by influence-placement adjacency and realignment modifiers, which both
    count "countries you control" and must include the homeland.
    """
    if name == SUPERPOWER_SPACE[side]:
        return True
    if name in SUPERPOWER_SPACE.values():
        return False
    return controls(state, side, name)


def counts_as_battleground(
    state: GameState, name: str, region: Region | None = None, *, scoring: bool = True
) -> bool:
    """Whether *name* counts as a battleground right now.

    Formosan Resolution makes Taiwan a battleground for Asia scoring while the US
    controls it, and only for scoring -- never for any other purpose.
    """
    if COUNTRIES[name].battleground:
        return True
    if (
        scoring
        and name == "Taiwan"
        and region in (Region.ASIA, Region.SOUTHEAST_ASIA)
        and state.in_play("Formosan Resolution")
        and controls(state, Side.USA, "Taiwan")
    ):
        return True
    return False


# --------------------------------------------------------------------------- #
# Region scoring
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RegionSide:
    """One player's standing in a region."""

    countries: int
    battlegrounds: int
    non_battlegrounds: int
    adjacent_to_enemy_superpower: int


@dataclass(frozen=True, slots=True)
class RegionStatus:
    region: Region
    total_battlegrounds: int
    sides: tuple[RegionSide, RegionSide]  # indexed by Side

    def tier(self, side: Side) -> str:
        """``"control"``, ``"domination"``, ``"presence"`` or ``"none"`` for *side*."""
        me, them = self.sides[side], self.sides[side.opponent]
        if me.countries == 0:
            return "none"
        # Control: every battleground in the region, and more countries than the
        # opponent.
        if (
            self.total_battlegrounds > 0
            and me.battlegrounds == self.total_battlegrounds
            and me.countries > them.countries
        ):
            return "control"
        # Domination: more countries and more battlegrounds than the opponent, and at
        # least one battleground plus one non-battleground.
        if (
            me.countries > them.countries
            and me.battlegrounds > them.battlegrounds
            and me.battlegrounds >= 1
            and me.non_battlegrounds >= 1
        ):
            return "domination"
        return "presence"


#: Regions whose scoring card Shuttle Diplomacy affects.
SHUTTLE_DIPLOMACY_REGIONS = (Region.ASIA, Region.MIDDLE_EAST)


def shuttle_diplomacy_candidates(state: GameState, region: Region) -> list[str]:
    """The USSR battlegrounds Shuttle Diplomacy could remove from this scoring.

    Empty unless the card is in play and *region* is Asia or the Middle East (never
    Southeast Asia, final scoring, Summit or Kitchen Debates).
    """
    if region not in SHUTTLE_DIPLOMACY_REGIONS or not state.in_play("Shuttle Diplomacy"):
        return []
    return [
        n for n in REGION_COUNTRIES[region]
        if controls(state, Side.USSR, n) and counts_as_battleground(state, n, region)
    ]


def shuttle_diplomacy_excluded(state: GameState, region: Region) -> str | None:
    """The USSR battleground Shuttle Diplomacy would remove, for previews.

    "During the next scoring of the Middle East or Asia, subtract one Battleground
    country from USSR total." The country leaves the count entirely -- one fewer
    country, one fewer battleground, and no superpower-adjacency point for it (FAQ #73)
    -- it does not become a non-battleground. The US picks which when the scoring card
    resolves (:meth:`twilight.engine.Game.score_region`); the choice only matters for
    the adjacency point, so a preview assumes the US takes a country adjacent to it
    when there is one.
    """
    bgs = shuttle_diplomacy_candidates(state, region)
    if not bgs:
        return None
    adjacent = ADJACENT_TO_SUPERPOWER[Side.USA]
    return next((n for n in bgs if n in adjacent), bgs[0])


def region_status(
    state: GameState,
    region: Region,
    *,
    scoring: bool = False,
    shuttle: bool = False,
    shuttle_choice: str | None = None,
) -> RegionStatus:
    """Both players' standing in *region*.

    *scoring* counts Formosan Resolution's Taiwan as a battleground ("for scoring
    purposes only": scoring cards and final scoring). *shuttle* also applies Shuttle
    Diplomacy, which only a scoring card does; *shuttle_choice* is the battleground the
    US picked (default: the preview pick of :func:`shuttle_diplomacy_excluded`). Summit
    sees the board as it is.
    """
    scoring = scoring or shuttle
    excluded = None
    if shuttle:
        excluded = shuttle_choice or shuttle_diplomacy_excluded(state, region)
    # counts_as_battleground, inlined: a printed battleground, or Taiwan while Formosan
    # Resolution makes it one (scoring only, Asia / Southeast Asia, US-controlled).
    formosan = (
        scoring
        and region in _FORMOSAN_REGIONS
        and state.in_play("Formosan Resolution")
        and controls(state, Side.USA, "Taiwan")
    )
    table = _REGION_TABLE[region]
    influence = state.influence
    per_side: list[RegionSide] = []

    for side in SIDES:
        # controls(state, side, n), inlined.
        mine_row = influence[side]
        theirs_row = influence[1 - side]
        adjacent = ADJACENT_TO_SUPERPOWER[side.opponent]
        countries = battlegrounds = adjacent_count = 0
        for name, slot, stability, battleground in table:
            mine = mine_row[slot]
            if mine == 0 or mine - theirs_row[slot] < stability or name == excluded:
                continue
            countries += 1
            if battleground or (formosan and name == "Taiwan"):
                battlegrounds += 1
            if name in adjacent:
                adjacent_count += 1
        per_side.append(
            RegionSide(
                countries=countries,
                battlegrounds=battlegrounds,
                non_battlegrounds=countries - battlegrounds,
                adjacent_to_enemy_superpower=adjacent_count,
            )
        )

    total_bgs = _REGION_BG_COUNT[region]
    if formosan and _TAIWAN_EXTRA_BG[region]:
        total_bgs += 1
    return RegionStatus(region=region, total_battlegrounds=total_bgs, sides=tuple(per_side))


#: Per region, in REGION_COUNTRIES order: (name, slot, stability, printed battleground).
_REGION_TABLE: dict[Region, tuple[tuple[str, int, int, bool], ...]] = {
    region: tuple(
        (n, COUNTRY_SLOT[n], COUNTRIES[n].stability, COUNTRIES[n].battleground) for n in names
    )
    for region, names in REGION_COUNTRIES.items()
}
#: Printed battlegrounds per region.
_REGION_BG_COUNT: dict[Region, int] = {
    region: sum(1 for *_, bg in table if bg) for region, table in _REGION_TABLE.items()
}
#: Regions in which Formosan Resolution can make Taiwan a battleground for scoring.
_FORMOSAN_REGIONS = (Region.ASIA, Region.SOUTHEAST_ASIA)
#: Whether Formosan Resolution adds a battleground to the region's total (Taiwan is in
#: the region and not a printed battleground).
_TAIWAN_EXTRA_BG: dict[Region, bool] = {
    region: "Taiwan" in names and not COUNTRIES["Taiwan"].battleground
    for region, names in REGION_COUNTRIES.items()
}


def region_vp(
    state: GameState,
    region: Region,
    side: Side,
    *,
    shuttle: bool = False,
    shuttle_choice: str | None = None,
    status: RegionStatus | None = None,
) -> int:
    """VP *side* would score for *region* right now, before the opponent's total.

    *shuttle*: as the scoring card would, applying Shuttle Diplomacy (to
    *shuttle_choice*, if the US has picked one). *status*, if given, must be
    ``region_status(state, region, scoring=True, shuttle=shuttle,
    shuttle_choice=shuttle_choice)`` for the current state; it saves recomputing it.
    """
    if region not in SCORING_VALUES:
        raise ValueError(
            f"{region} has no scoring card; it is only scored as part of its parent region"
        )
    if region is Region.SOUTHEAST_ASIA:
        return sum(
            SOUTHEAST_ASIA_COUNTRY_VP.get(n, 1)
            for n in REGION_COUNTRIES[region]
            if controls(state, side, n)
        )

    if status is None:
        status = region_status(
            state, region, scoring=True, shuttle=shuttle, shuttle_choice=shuttle_choice
        )
    presence, domination, control = SCORING_VALUES[region]
    tier = status.tier(side)
    base = {"control": control, "domination": domination, "presence": presence, "none": 0}[tier]
    if base == AUTO_VICTORY:
        return AUTO_VICTORY

    me = status.sides[side]
    bonus = me.battlegrounds
    if region in ADJACENCY_BONUS_REGIONS:
        bonus += me.adjacent_to_enemy_superpower
    return base + bonus


def score_region(
    state: GameState,
    region: Region,
    *,
    shuttle: bool = False,
    shuttle_choice: str | None = None,
) -> tuple[int, int, Side | None]:
    """Score *region*: a scoring card (``shuttle=True``) or final scoring.

    *shuttle_choice* is the USSR battleground the US removed with Shuttle Diplomacy.

    Returns ``(ussr_vp, usa_vp, auto_winner)``. The VP difference is applied to the
    track; *auto_winner* is set only when a player controls Europe, which ends the
    game immediately.
    """
    totals = {
        side: region_vp(state, region, side, shuttle=shuttle, shuttle_choice=shuttle_choice)
        for side in Side
    }

    for side in Side:
        if totals[side] == AUTO_VICTORY:
            state.note(f"{side.label} controls Europe", side)
            return (0, 0, side)

    net = totals[Side.USSR] - totals[Side.USA]
    if net > 0:
        state.award_vp(Side.USSR, net)
    elif net < 0:
        state.award_vp(Side.USA, -net)

    if region is Region.SOUTHEAST_ASIA:
        status_note = ", ".join(f"{side.label} {totals[side]}" for side in Side)
    else:
        status = region_status(
            state, region, scoring=True, shuttle=shuttle, shuttle_choice=shuttle_choice
        )
        status_note = ", ".join(
            f"{side.label} {status.tier(side)} ({totals[side]})" for side in Side
        )
    state.note(f"scored {region}: {status_note} -> net {net:+d} VP to USSR")
    return (totals[Side.USSR], totals[Side.USA], None)


# --------------------------------------------------------------------------- #
# Influence placement
# --------------------------------------------------------------------------- #


def reachable_countries(state: GameState, side: Side) -> frozenset[str]:
    """Where *side* may place operations influence, as a snapshot.

    Legal targets are countries the player already occupies, and countries adjacent to
    one the player occupies or controls. A player may always place adjacent to their
    own superpower space regardless of presence (rule 6.1.4).

    This must be captured *once*, at the start of an action round: rule 6.1.1 measures
    adjacency against the markers in place when the round began, so influence placed
    during the round cannot be chained outward to reach further countries.
    """
    # Same insertion order as a loop over COUNTRIES, so the set is built identically.
    out: set[str] = set()
    row = state.influence[side]
    for name, slot, neighbours in _REACH_TABLE:
        if row[slot] > 0:
            out.add(name)
            out.update(neighbours)
    # Always reachable: neighbours of the player's own homeland.
    out.update(
        n for n in COUNTRIES[SUPERPOWER_SPACE[side]].adjacent if not COUNTRIES[n].superpower
    )
    return frozenset(out)


#: Non-superpower countries in COUNTRIES order: (name, slot, non-superpower neighbours).
_REACH_TABLE: tuple[tuple[str, int, tuple[str, ...]], ...] = tuple(
    (name, COUNTRY_SLOT[name], tuple(n for n in c.adjacent if not COUNTRIES[n].superpower))
    for name, c in COUNTRIES.items()
    if not c.superpower
)


def can_place_influence(
    state: GameState, side: Side, name: str, reachable: frozenset[str] | None = None
) -> bool:
    """Whether *side* may place operations influence into *name*.

    Pass *reachable* (from :func:`reachable_countries`) to honour the start-of-round
    snapshot; without it the check is recomputed live, which is only correct for the
    first point placed.
    """
    if name in SUPERPOWER_SPACE.values():
        return False
    if reachable is None:
        reachable = reachable_countries(state, side)
    return name in reachable


def influence_cost(state: GameState, side: Side, name: str) -> int:
    """Cost of placing one influence point: 2 into opponent-controlled countries."""
    return 2 if controls(state, side.opponent, name) else 1


def placeable_countries(
    state: GameState,
    side: Side,
    budget: int | None = None,
    reachable: frozenset[str] | None = None,
) -> list[str]:
    """Countries *side* may legally place influence into, in board order.

    Restricted to those affordable within *budget* operations points when given, and to
    those no card currently forbids.
    """
    if reachable is None:
        reachable = reachable_countries(state, side)
    banned = banned_placement_regions(state, side)
    return [
        name
        for name in COUNTRY_ORDER
        if name in reachable
        and (budget is None or influence_cost(state, side, name) <= budget)
        and not any(region in banned for region in COUNTRIES[name].regions)
    ]


def banned_placement_regions(state: GameState, side: Side) -> frozenset[Region]:
    """Regions *side* may not place operations influence into.

    Chernobyl names one region the opponent is shut out of for the rest of the turn.
    """
    effect = state.effects.get("Chernobyl")
    if effect is None or effect.owner is None or effect.owner is side:
        return frozenset()
    named = effect.data.get("region")
    return frozenset({Region(named)}) if named else frozenset()


# --------------------------------------------------------------------------- #
# DEFCON restrictions
# --------------------------------------------------------------------------- #

#: Regions progressively closed to coups and realignments as DEFCON degrades. Both
#: operations are restricted, not just coups.
DEFCON_RESTRICTED: dict[int, frozenset[Region]] = {
    5: frozenset(),
    4: frozenset({Region.EUROPE}),
    3: frozenset({Region.EUROPE, Region.ASIA}),
    2: frozenset({Region.EUROPE, Region.ASIA, Region.MIDDLE_EAST}),
    1: frozenset({Region.EUROPE, Region.ASIA, Region.MIDDLE_EAST}),
}


def restricted_regions(state: GameState) -> frozenset[Region]:
    return DEFCON_RESTRICTED[state.defcon]


def defcon_allows_operation(state: GameState, name: str) -> bool:
    """Whether DEFCON permits a coup or realignment in *name*."""
    blocked = restricted_regions(state)
    return not any(region in blocked for region in COUNTRIES[name].regions)


def can_coup(
    state: GameState, side: Side, name: str, *, ignore_defcon: bool = False
) -> bool:
    """Whether *side* may make a coup attempt in *name*.

    *ignore_defcon* is for an event's free coup in a named region, which ignores the
    DEFCON geographic restrictions (rule 6.3.5); every other restriction still holds.
    """
    if state.inf(side.opponent, name) == 0:
        return False
    if not ignore_defcon and not defcon_allows_operation(state, name):
        return False
    return _extra_operation_restrictions_allow(state, side, name, coup=True)


def can_realign(
    state: GameState, side: Side, name: str, *, ignore_defcon: bool = False
) -> bool:
    # The target must hold some opponent influence (FAQ 6.2.1: "at least one enemy
    # influence must be present"); the realigning side needs none of its own.
    if state.inf(side.opponent, name) == 0:
        return False
    if not ignore_defcon and not defcon_allows_operation(state, name):
        return False
    return _extra_operation_restrictions_allow(state, side, name, coup=False)


def _extra_operation_restrictions_allow(
    state: GameState, side: Side, name: str, *, coup: bool
) -> bool:
    """Card-imposed limits on where coups and realignments may happen."""
    regions = COUNTRIES[name].regions

    # The Reformer: USSR may not coup in Europe while it is in play.
    if coup and side is Side.USSR and state.in_play("The Reformer"):
        if Region.EUROPE in regions:
            return False

    # NATO: the USSR may not coup or realign in US-controlled European countries.
    # (Warsaw Pact Formed has no such clause; it only allows NATO to be played.)
    if side is Side.USSR and nato_protects(state, name):
        return False

    # US/Japan Mutual Defense Pact: the USSR may never coup or realign in Japan.
    if name == "Japan" and side is Side.USSR and state.in_play("US/Japan Mutual Defense Pact"):
        return False

    return True


def nato_protects(state: GameState, name: str) -> bool:
    """Whether NATO currently shields *name* from the USSR.

    NATO covers US-controlled countries in Europe, except France once De Gaulle Leads
    France has been played and West Germany while Willy Brandt is in effect.
    """
    if not state.in_play("NATO"):
        return False
    if Region.EUROPE not in COUNTRIES[name].regions:
        return False
    if not controls(state, Side.USA, name):
        return False
    if name == "France" and state.has_resolved("De Gaulle Leads France"):
        return False
    # Willy Brandt cancels NATO for West Germany only while it is in effect: Tear Down
    # This Wall cancels it and restores the protection.
    if name == "West Germany" and state.in_play("Willy Brandt"):
        return False
    return True


# --------------------------------------------------------------------------- #
# Coup
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CoupResult:
    country: str
    roll: int
    modifier: int
    total: int
    required: int
    success: bool
    removed: int
    placed: int


def coup_modifier(state: GameState, side: Side, name: str) -> int:
    """Card-driven adjustments to a coup's effective operations value."""
    modifier = 0
    regions = COUNTRIES[name].regions

    # Latin American Death Squads: +1 for the owner, -1 for the opponent, in Central
    # and South America.
    effect = state.effects.get("Latin American Death Squads")
    if effect is not None and (
        Region.CENTRAL_AMERICA in regions or Region.SOUTH_AMERICA in regions
    ):
        modifier += 1 if effect.owner is side else -1

    # SALT Negotiations: all coup attempts are at -1 while it is in effect.
    if state.in_play("Salt Negotiations"):
        modifier -= 1

    return modifier


def resolve_coup(
    state: GameState,
    side: Side,
    name: str,
    ops: int,
    *,
    free: bool = False,
    roll: int | None = None,
) -> CoupResult:
    """Carry out a coup attempt in *name* with *ops* operations points.

    Roll a die, add the operations value, and subtract twice the country's stability.
    Any surplus removes opponent influence first, then adds the player's own. A coup
    always costs military operations, and a coup in a battleground degrades DEFCON --
    both of which happen even when the attempt fails.

    *free* marks a coup granted by an event rather than paid for with a card. Free
    coups ignore the DEFCON geographic restrictions and do **not** count toward the
    required military operations, though they still degrade DEFCON in a battleground.
    """
    stability = country(name).stability
    modifier = coup_modifier(state, side, name)
    if roll is None:
        roll = state.rng.randint(1, 6)

    required = stability * 2
    total = roll + ops + modifier
    surplus = total - required

    removed = placed = 0
    if surplus > 0:
        removed = state.remove_inf(side.opponent, name, surplus)
        placed = surplus - removed
        if placed:
            state.add_inf(side, name, placed)

    if not free:
        state.military_ops[side] = min(5, state.military_ops[side] + ops)

    # Nuclear Subs exempts US coups made with operations points in battlegrounds from
    # degrading DEFCON -- including an event's "as if N Ops" operations, but not a free
    # coup (user ruling 2026-11-25).
    suppressed = (
        side is Side.USA and not free and state.in_play("Nuclear Subs", owner=Side.USA)
    )
    if COUNTRIES[name].battleground and not suppressed:
        state.change_defcon(-1)

    verb = "succeeds" if surplus > 0 else "fails"
    state.note(
        f"coup in {name} {verb}: roll {roll} + {ops} ops"
        + (f" {modifier:+d}" if modifier else "")
        + f" = {total} vs {required} required"
        + (f"; -{removed} {side.opponent.label}, +{placed} {side.label}" if surplus > 0 else ""),
        side,
    )

    return CoupResult(
        country=name,
        roll=roll,
        modifier=modifier,
        total=total,
        required=required,
        success=surplus > 0,
        removed=removed,
        placed=placed,
    )


# --------------------------------------------------------------------------- #
# Realignment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RealignResult:
    country: str
    rolls: tuple[int, int]
    modifiers: tuple[int, int]
    totals: tuple[int, int]
    winner: Side | None
    removed: int


def realignment_modifier(state: GameState, side: Side, name: str) -> int:
    """+1 per adjacent country controlled, +1 for more influence in the target."""
    modifier = sum(
        1
        for neighbour in COUNTRIES[name].adjacent
        if is_controlled_by_superpower(state, side, neighbour)
    )
    if state.inf(side, name) > state.inf(side.opponent, name):
        modifier += 1
    return modifier


def resolve_realignment(
    state: GameState,
    side: Side,
    name: str,
    *,
    rolls: tuple[int, int] | None = None,
) -> RealignResult:
    """Resolve one realignment roll in *name*.

    Both players roll a die and add their modifiers; the higher total removes the
    difference in influence from the loser. A tie changes nothing.
    """
    if rolls is None:
        rolls = (state.rng.randint(1, 6), state.rng.randint(1, 6))

    modifiers = [realignment_modifier(state, s, name) for s in Side]

    # Some events reduce the opponent's realignment rolls in a region.
    for s in Side:
        modifiers[s] -= _realignment_penalty(state, s, name)

    totals = [rolls[s] + modifiers[s] for s in Side]

    winner: Side | None = None
    removed = 0
    if totals[Side.USSR] != totals[Side.USA]:
        winner = Side.USSR if totals[Side.USSR] > totals[Side.USA] else Side.USA
        margin = abs(totals[Side.USSR] - totals[Side.USA])
        removed = state.remove_inf(winner.opponent, name, margin)

    detail = " vs ".join(
        f"{s.label} {rolls[s]}{modifiers[s]:+d}={totals[s]}" for s in Side
    )
    outcome = (
        f"{winner.opponent.label} loses {removed}" if winner is not None else "tie, no change"
    )
    state.note(f"realign {name}: {detail}; {outcome}", side)

    return RealignResult(
        country=name,
        rolls=(rolls[Side.USSR], rolls[Side.USA]),
        modifiers=(modifiers[Side.USSR], modifiers[Side.USA]),
        totals=(totals[Side.USSR], totals[Side.USA]),
        winner=winner,
        removed=removed,
    )


def _realignment_penalty(state: GameState, side: Side, name: str) -> int:
    """How much *side*'s realignment roll here is reduced by an opponent's card."""
    del name  # no card currently restricts realignment by country, only by side
    penalty = 0
    # Iran-Contra Scandal: US realignment rolls are at -1 while it is in play.
    if side is Side.USA and state.in_play("Iran-Contra Scandal"):
        penalty += 1
    return penalty


__all__ = [
    "CoupResult",
    "DEFCON_RESTRICTED",
    "RealignResult",
    "RegionSide",
    "RegionStatus",
    "can_coup",
    "can_place_influence",
    "can_realign",
    "controller",
    "controls",
    "counts_as_battleground",
    "coup_modifier",
    "defcon_allows_operation",
    "influence_cost",
    "is_controlled_by_superpower",
    "nato_protects",
    "placeable_countries",
    "reachable_countries",
    "realignment_modifier",
    "region_status",
    "region_vp",
    "resolve_coup",
    "resolve_realignment",
    "restricted_regions",
    "score_region",
    "shuttle_diplomacy_candidates",
    "shuttle_diplomacy_excluded",
]

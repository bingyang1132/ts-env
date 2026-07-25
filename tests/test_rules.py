"""Rules primitives, checked against positions with a known correct answer."""

from __future__ import annotations

import pytest

from twilight import data, rules
from twilight.enums import AUTO_VICTORY, Region, Side
from twilight.state import GameState


@pytest.fixture
def state() -> GameState:
    return GameState()


# --------------------------------------------------------------------------- #
# Board data
# --------------------------------------------------------------------------- #


def test_board_matches_the_printed_map():
    assert data.NUM_COUNTRIES == 84
    assert data.NUM_CARDS == 110
    # UK is the only stability-5 country.
    fives = [n for n in data.COUNTRY_ORDER if data.country(n).stability == 5]
    assert fives == ["UK"]
    # 29 battlegrounds, distributed as printed.
    assert sum(data.country(n).battleground for n in data.COUNTRY_ORDER) == 29
    expected = {
        Region.EUROPE: 5,
        Region.ASIA: 6,
        Region.MIDDLE_EAST: 6,
        Region.AFRICA: 5,
        Region.CENTRAL_AMERICA: 3,
        Region.SOUTH_AMERICA: 4,
    }
    for region, count in expected.items():
        assert len(data.REGION_BATTLEGROUNDS[region]) == count, region


def test_austria_and_finland_are_in_both_halves_of_europe():
    for name in ("Austria", "Finland"):
        regions = set(data.country(name).regions)
        assert regions == {
            Region.WESTERN_EUROPE,
            Region.EASTERN_EUROPE,
            Region.EUROPE,
        }


def test_adjacency_is_symmetric():
    for name in data.COUNTRIES:
        for neighbour in data.country(name).adjacent:
            assert name in data.country(neighbour).adjacent, f"{name} -> {neighbour}"


def test_superpower_adjacency():
    assert set(data.country("USA").adjacent) == {"Canada", "Cuba", "Japan", "Mexico"}
    assert set(data.country("USSR").adjacent) == {
        "Finland", "Poland", "Romania", "Afghanistan", "North Korea",
    }


# --------------------------------------------------------------------------- #
# Control
# --------------------------------------------------------------------------- #


def test_control_needs_stability_margin(state):
    # Italy has stability 2.
    state.set_inf(Side.USSR, "Italy", 2)
    assert rules.controls(state, Side.USSR, "Italy")

    state.set_inf(Side.USA, "Italy", 1)
    assert not rules.controls(state, Side.USSR, "Italy")
    assert rules.controller(state, "Italy") is None

    state.set_inf(Side.USSR, "Italy", 3)
    assert rules.controller(state, "Italy") is Side.USSR


def test_nobody_controls_an_empty_country(state):
    assert rules.controller(state, "Italy") is None


def test_superpower_space_counts_as_controlled_by_its_owner(state):
    assert rules.is_controlled_by_superpower(state, Side.USSR, "USSR")
    assert not rules.is_controlled_by_superpower(state, Side.USSR, "USA")


# --------------------------------------------------------------------------- #
# Influence placement
# --------------------------------------------------------------------------- #


def test_can_always_place_next_to_own_superpower(state):
    for name in ("Poland", "Finland", "Romania", "Afghanistan", "North Korea"):
        assert rules.can_place_influence(state, Side.USSR, name), name
    assert not rules.can_place_influence(state, Side.USSR, "Chile")


def test_cannot_place_into_a_superpower_space(state):
    assert not rules.can_place_influence(state, Side.USSR, "USA")
    assert not rules.can_place_influence(state, Side.USA, "USSR")


def test_placement_costs_double_into_enemy_controlled(state):
    state.set_inf(Side.USA, "Italy", 4)
    assert rules.influence_cost(state, Side.USSR, "Italy") == 2
    assert rules.influence_cost(state, Side.USA, "Italy") == 1
    assert rules.influence_cost(state, Side.USSR, "Poland") == 1


def test_reachability_is_a_snapshot_not_a_chain(state):
    """Influence placed this round must not extend reach further out."""
    state.set_inf(Side.USSR, "Poland", 1)
    reachable = rules.reachable_countries(state, Side.USSR)
    assert "East Germany" in reachable          # adjacent to Poland
    assert "West Germany" not in reachable      # two steps away

    # Placing in East Germany does not open West Germany within the same snapshot.
    state.add_inf(Side.USSR, "East Germany", 1)
    assert "West Germany" not in reachable
    # A fresh snapshot, i.e. a later action round, does see it.
    assert "West Germany" in rules.reachable_countries(state, Side.USSR)


# --------------------------------------------------------------------------- #
# DEFCON restrictions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "defcon,country,allowed",
    [
        (5, "Italy", True),
        (4, "Italy", False),      # Europe closed
        (4, "Iran", True),
        (3, "Iran", True),
        (3, "Japan", False),      # Asia closed
        (3, "Thailand", False),   # Southeast Asia counts as Asia
        (2, "Iran", False),       # Middle East closed
        (2, "Angola", True),      # Africa never restricted
        (2, "Panama", True),
    ],
)
def test_defcon_closes_regions_progressively(state, defcon, country, allowed):
    state.defcon = defcon
    state.set_inf(Side.USA, country, 1)
    assert rules.can_coup(state, Side.USSR, country) is allowed
    # The restriction applies to realignment too, not only coups.
    assert rules.can_realign(state, Side.USSR, country) is allowed


def test_coup_requires_enemy_influence(state):
    assert not rules.can_coup(state, Side.USSR, "Chile")
    state.set_inf(Side.USA, "Chile", 1)
    assert rules.can_coup(state, Side.USSR, "Chile")


# --------------------------------------------------------------------------- #
# Coup
# --------------------------------------------------------------------------- #


def test_coup_arithmetic_and_side_effects(state):
    state.set_inf(Side.USA, "Iran", 3)  # stability 2, battleground
    result = rules.resolve_coup(state, Side.USSR, "Iran", ops=4, roll=3)

    assert result.total == 7 and result.required == 4
    assert result.success
    assert (result.removed, result.placed) == (3, 0)
    assert state.inf(Side.USA, "Iran") == 0
    assert state.military_ops[Side.USSR] == 4
    assert state.defcon == 4  # battleground coup degrades DEFCON


def test_coup_surplus_beyond_removal_is_placed(state):
    state.set_inf(Side.USA, "Panama", 1)  # stability 2
    result = rules.resolve_coup(state, Side.USSR, "Panama", ops=4, roll=5)
    assert (result.removed, result.placed) == (1, 4)
    assert state.inf(Side.USSR, "Panama") == 4


def test_coup_fails_when_not_strictly_greater(state):
    state.set_inf(Side.USA, "Iran", 3)  # needs > 4
    result = rules.resolve_coup(state, Side.USSR, "Iran", ops=1, roll=3)
    assert result.total == 4 and not result.success
    assert state.inf(Side.USA, "Iran") == 3
    # Military ops and DEFCON still move on a failure.
    assert state.military_ops[Side.USSR] == 1
    assert state.defcon == 4


def test_free_coup_does_not_count_military_operations(state):
    state.set_inf(Side.USA, "Angola", 2)
    rules.resolve_coup(state, Side.USSR, "Angola", ops=3, roll=4, free=True)
    assert state.military_ops[Side.USSR] == 0


def test_non_battleground_coup_leaves_defcon_alone(state):
    state.set_inf(Side.USA, "Sudan", 2)
    rules.resolve_coup(state, Side.USSR, "Sudan", ops=3, roll=4)
    assert state.defcon == 5


# --------------------------------------------------------------------------- #
# Realignment
# --------------------------------------------------------------------------- #


def test_realignment_removes_the_difference_in_rolls(state):
    state.set_inf(Side.USSR, "Chile", 5)
    state.set_inf(Side.USA, "Chile", 5)
    # Equal influence, no adjacency: modifiers are 0 for both, so the margin is 4.
    result = rules.resolve_realignment(state, Side.USSR, "Chile", rolls=(6, 2))
    assert result.winner is Side.USSR
    assert result.removed == 4
    assert state.inf(Side.USA, "Chile") == 1


def test_realignment_removal_is_capped_by_what_is_present(state):
    state.set_inf(Side.USSR, "Chile", 3)
    state.set_inf(Side.USA, "Chile", 3)
    # Margin of 4 against only 3 influence removes 3, not 4.
    result = rules.resolve_realignment(state, Side.USSR, "Chile", rolls=(6, 2))
    assert result.removed == 3
    assert state.inf(Side.USA, "Chile") == 0


def test_realignment_can_backfire_on_the_attacker(state):
    """A losing realignment costs the attacker influence, not the defender."""
    state.set_inf(Side.USSR, "Chile", 5)
    state.set_inf(Side.USA, "Chile", 5)
    result = rules.resolve_realignment(state, Side.USSR, "Chile", rolls=(1, 5))
    assert result.winner is Side.USA
    assert result.removed == 4
    assert state.inf(Side.USSR, "Chile") == 1
    assert state.inf(Side.USA, "Chile") == 5


def test_realignment_tie_changes_nothing(state):
    state.set_inf(Side.USSR, "Chile", 2)
    state.set_inf(Side.USA, "Chile", 2)
    result = rules.resolve_realignment(state, Side.USSR, "Chile", rolls=(4, 4))
    assert result.winner is None and result.removed == 0
    assert state.inf(Side.USSR, "Chile") == 2
    assert state.inf(Side.USA, "Chile") == 2


def test_realignment_modifiers(state):
    # More influence in the target is worth +1.
    state.set_inf(Side.USSR, "Poland", 3)
    state.set_inf(Side.USA, "Poland", 1)
    # Poland is adjacent to the USSR homeland, worth another +1.
    assert rules.realignment_modifier(state, Side.USSR, "Poland") == 2
    assert rules.realignment_modifier(state, Side.USA, "Poland") == 0


def test_superpower_adjacency_counts_once(state):
    """The homeland bonus must not be double-counted."""
    # Cuba is adjacent to the USA and nothing else the US controls.
    modifier = rules.realignment_modifier(state, Side.USA, "Cuba")
    assert modifier == 1


# --------------------------------------------------------------------------- #
# Region scoring
# --------------------------------------------------------------------------- #


def test_presence_domination_and_control_tiers(state):
    # One non-battleground in Africa: presence only.
    state.set_inf(Side.USSR, "Sudan", 5)
    status = rules.region_status(state, Region.AFRICA)
    assert status.tier(Side.USSR) == "presence"
    assert rules.region_vp(state, Region.AFRICA, Side.USSR) == 1

    # Add a battleground: now more countries and battlegrounds, with one of each.
    state.set_inf(Side.USSR, "Nigeria", 5)
    status = rules.region_status(state, Region.AFRICA)
    assert status.tier(Side.USSR) == "domination"
    assert rules.region_vp(state, Region.AFRICA, Side.USSR) == 4 + 1

    # All five battlegrounds and more countries: control.
    for name in data.REGION_BATTLEGROUNDS[Region.AFRICA]:
        state.set_inf(Side.USSR, name, 9)
    status = rules.region_status(state, Region.AFRICA)
    assert status.tier(Side.USSR) == "control"
    assert rules.region_vp(state, Region.AFRICA, Side.USSR) == 6 + 5


def test_domination_requires_a_non_battleground(state):
    """Battlegrounds alone give presence, never domination."""
    for name in data.REGION_BATTLEGROUNDS[Region.AFRICA]:
        state.set_inf(Side.USSR, name, 9)
    status = rules.region_status(state, Region.AFRICA)
    # Every battleground but no plain country, and more countries than the opponent:
    # that satisfies Control, which outranks Domination.
    assert status.tier(Side.USSR) == "control"

    # With only some battlegrounds and no non-battleground, it drops to presence.
    state.clear_inf(Side.USSR, "Algeria")
    assert rules.region_status(state, Region.AFRICA).tier(Side.USSR) == "presence"


def test_superpower_adjacency_bonus_only_in_the_three_regions(state):
    # Cuba is a Central American battleground adjacent to the USA.
    state.set_inf(Side.USSR, "Cuba", 5)
    state.set_inf(Side.USSR, "Nicaragua", 5)
    status = rules.region_status(state, Region.CENTRAL_AMERICA)
    assert status.tier(Side.USSR) == "domination"
    # 3 domination + 1 battleground + 1 adjacent to the USA.
    assert rules.region_vp(state, Region.CENTRAL_AMERICA, Side.USSR) == 3 + 1 + 1

    # Middle East prints no adjacency clause, and has no adjacent country anyway.
    state.set_inf(Side.USSR, "Iraq", 9)
    state.set_inf(Side.USSR, "Jordan", 9)
    status = rules.region_status(state, Region.MIDDLE_EAST)
    assert rules.region_vp(state, Region.MIDDLE_EAST, Side.USSR) == 5 + 1


def test_southeast_asia_is_scored_per_country_with_thailand_double(state):
    state.set_inf(Side.USSR, "Vietnam", 5)
    assert rules.region_vp(state, Region.SOUTHEAST_ASIA, Side.USSR) == 1
    state.set_inf(Side.USSR, "Thailand", 5)
    assert rules.region_vp(state, Region.SOUTHEAST_ASIA, Side.USSR) == 3


def test_controlling_europe_is_an_automatic_victory(state):
    for name in data.REGION_COUNTRIES[Region.EUROPE]:
        state.set_inf(Side.USSR, name, 9)
    assert rules.region_vp(state, Region.EUROPE, Side.USSR) == AUTO_VICTORY
    _, _, auto = rules.score_region(state, Region.EUROPE)
    assert auto is Side.USSR


def test_scoring_applies_only_the_net_difference(state):
    state.set_inf(Side.USSR, "Sudan", 5)      # USSR presence in Africa: 1
    state.set_inf(Side.USA, "Nigeria", 5)     # US presence + 1 battleground: 2
    before = state.vp
    rules.score_region(state, Region.AFRICA)
    assert state.vp == before - 1             # net 1 VP to the US


def test_sub_regions_have_no_scoring_card(state):
    for region in (Region.WESTERN_EUROPE, Region.EASTERN_EUROPE):
        with pytest.raises(ValueError, match="no scoring card"):
            rules.region_vp(state, region, Side.USSR)

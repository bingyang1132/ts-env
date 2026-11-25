"""DEFCON restrictions and loss attribution (R 8.1, V:B9) and required military
operations (R 8.2, V:B8). File name keeps the test_mech_ prefix: see test_mech_*.py."""

from __future__ import annotations

import pytest

from _table import USA, USSR, Table
from twilight import rules
from twilight.enums import WinReason


class TestDefconRestrictions:
    @pytest.mark.parametrize(
        "defcon,blocked,open_",
        [
            (5, [], ["Italy", "India", "Iraq"]),
            (4, ["Italy"], ["India", "Iraq"]),
            (3, ["Italy", "India"], ["Iraq"]),
            (2, ["Italy", "India", "Iraq"], ["Angola"]),
        ],
    )
    def test_regions_closed_to_coups_and_realignments(self, defcon, blocked, open_):
        """R 8.1.5: DEFCON 4 "No Coup or Realignment rolls are permitted in Europe";
        DEFCON 3 "... in Europe or Asia"; DEFCON 2 "... in Europe, Asia, or the Middle
        East". FAQ §8.1.5: realignments are restricted too."""
        t = Table(turn=2, defcon=defcon)
        names = ["Italy", "India", "Iraq", "Angola"]
        t.inf(usa={n: 1 for n in names}, ussr={n: 1 for n in names})
        for n in blocked:
            assert not rules.can_coup(t.s, USSR, n), (defcon, n)
            assert not rules.can_realign(t.s, USSR, n), (defcon, n)
        for n in open_:
            assert rules.can_coup(t.s, USSR, n), (defcon, n)

    def test_defcon_improves_one_per_turn_capped_at_five(self):
        """R 8.1.6 "Improve the DEFCON status by 1"; R 8.1.2 never above 5."""
        t = Table(turn=2, defcon=3)
        t.turn_end()
        assert t.s.defcon == 4
        t2 = Table(turn=2, defcon=5)
        t2.turn_end()
        assert t2.s.defcon == 5


class TestDefconLoss:
    def test_own_battleground_coup_to_1_loses(self):
        """R 8.1.3: the phasing player who takes DEFCON to 1 loses. USSR coups Angola at
        DEFCON 2."""
        t = Table(turn=2, defcon=2)
        t.inf(usa={"Angola": 1})
        t.dice(6)
        t.play(USSR, "Decolonization", "coup")
        assert t.s.is_over and t.s.winner is USA and t.s.win_reason is WinReason.DEFCON

    def test_opponent_coup_granted_by_triggered_event_loses_for_the_phasing_player(self):
        """V:B9 / FAQ #62 (same pattern as Lone Gunman): the USSR plays CIA Created for
        ops, the US event gives the US 1 op, the US coups the battleground Angola at
        DEFCON 2 -> DEFCON 1 during the USSR action round: the USSR (phasing) loses."""
        t = Table(turn=2, defcon=2)
        t.inf(ussr={"Angola": 1, "Poland": 1})
        t.dice(1)
        t.play(USSR, "CIA Created", "influence", "yes")
        if t.d is not None and t.d.type.value == "card_use":
            t.answer("use:coup")
        if t.d is not None and "country:Angola" in t.keys():
            t.answer("country:Angola")
        assert t.s.defcon == 1
        assert t.s.is_over and t.s.winner is USA, "the phasing USSR is responsible"


class TestMilitaryOperations:
    def test_shortfall_gives_vp_per_point(self):
        """R 8.2.1: "If fewer Military Operations are carried out, the opponent gains 1 VP
        per unplayed Military Operations point." EXAMPLE: US spent 2 at DEFCON 4 -> USSR
        +2 (the USSR met the requirement)."""
        t = Table(turn=2, defcon=4)
        t.s.military_ops = [4, 2]
        t.turn_end()
        assert t.s.vp == 2

    def test_both_short_nets_out(self):
        """R 8.2.1 applied to both players, R 10.2.3 net difference: DEFCON 5, USSR 2
        (short 3 -> US +3), US 0 (short 5 -> USSR +5): net USSR +2."""
        t = Table(turn=2, defcon=5)
        t.s.military_ops = [2, 0]
        t.turn_end()
        assert t.s.vp == 2

    def test_requirement_is_defcon_at_the_check(self):
        """R 8.2.1 "equal to the current DEFCON status"; the check (phase E) precedes the
        next turn's improvement. DEFCON 2, both at 2: no VP."""
        t = Table(turn=2, defcon=2)
        t.s.military_ops = [2, 2]
        t.turn_end()
        assert t.s.vp == 0 and t.s.military_ops == [0, 0]

    def test_realignment_is_not_military_operations(self):
        """R 8.2.2: "Realignment Rolls are not considered Military Operations.\""""
        t = Table(turn=2)
        t.inf(usa={"Angola": 2})
        t.dice(3, 3)
        t.play(USSR, "Decolonization", "realign", "country:Angola")
        t.finish_with()
        assert t.s.military_ops[USSR] == 0

    def test_containment_raises_coup_credit(self):
        """R 7.4.2 "for all purposes" / FAQ §7.4 (military ops credit): a 3-op US card
        under Containment coups for 4 and credits 4."""
        t = Table(turn=2)
        t.effect("Containment", USA, expires="end_of_turn")
        t.inf(ussr={"Angola": 1})
        t.dice(1)
        t.play(USA, "East European Unrest", "coup")
        assert t.s.military_ops[USA] == 4

    def test_military_ops_reset_each_turn(self):
        """R 4.5 E: "Each player then resets his Military Operations markers back to
        zero.\""""
        t = Table(turn=2, defcon=5)
        t.s.military_ops = [5, 5]
        t.turn_end()
        assert t.s.military_ops == [0, 0]

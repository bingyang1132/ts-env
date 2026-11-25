"""Operations: influence placement, coups, realignment, opponent events (R 5-6)."""

from __future__ import annotations

from _table import USA, USSR, Table
from twilight import rules
from twilight.decisions import DecisionType


class TestInfluencePlacement:
    def test_breaking_control_makes_further_points_cheaper(self):
        """R 6.1.2 EXAMPLE: "The US player has 2 Influence markers in Turkey ... the US
        controls Turkey. The USSR player uses a 4 Operations point card ... the first
        marker costs 2 Operations points. However, after placement of the first USSR
        influence marker ... a second or third Soviet influence marker would only cost 1
        operation point per marker." 4 ops -> 3 USSR influence in Turkey."""
        t = Table(turn=2)
        t.inf(usa={"Turkey": 2}, ussr={"Bulgaria": 1})
        t.play(USSR, "Red Scare/Purge", "influence")
        t.answer("country:Turkey", "country:Turkey", "country:Turkey")
        assert t.done and t.s.inf(USSR, "Turkey") == 3

    def test_adjacency_is_measured_at_the_start_of_the_round(self):
        """R 6.1.1 EXAMPLE: with markers in Panama "he cannot place Influence markers in
        Costa Rica and then Nicaragua.\""""
        t = Table(turn=2)
        t.inf(usa={"Panama": 1})
        t.play(USA, "Containment", "influence")
        t.answer("country:Costa Rica")
        assert "country:Nicaragua" not in t.keys()
        assert "country:Colombia" in t.keys()

    def test_superpower_neighbours_always_reachable(self):
        """R 6.1.4: "Influence markers may always be placed in any country that is
        adjacent (connected) to the phasing player's superpower space.\""""
        t = Table(turn=2)
        t.play(USSR, "Socialist Governments", "influence")
        for n in ("Finland", "Poland", "Romania", "Afghanistan", "North Korea"):
            assert f"country:{n}" in t.keys()


class TestCoup:
    def test_rulebook_example(self):
        """R 6.3.3 EXAMPLE: US 3-op coup in Mexico (USSR 2): roll 4 + 3 = 7 - 2x2 = 3:
        remove 2 USSR, place 1 US; military ops 3 (R 6.3.3); Mexico is a battleground so
        DEFCON degrades (R 6.3.4)."""
        t = Table(turn=2, defcon=5)
        t.inf(ussr={"Mexico": 2})
        t.dice(4)
        t.play(USA, "Containment", "coup")
        assert t.s.inf(USSR, "Mexico") == 0 and t.s.inf(USA, "Mexico") == 1
        assert t.s.military_ops[USA] == 3 and t.s.defcon == 4

    def test_needs_opponent_influence(self):
        """R 6.3.1: "your opponent must have Influence markers in the target country for
        a Coup to be attempted.\""""
        t = Table(turn=2)
        t.inf(usa={"Mexico": 2})
        assert not rules.can_coup(t.s, USA, "Mexico")
        assert not rules.can_coup(t.s, USA, "Angola")

    def test_failed_battleground_coup_still_degrades(self):
        """R 6.3.4: "Any Coup attempt in a Battleground country degrades the DEFCON
        status one level"; R 6.3.3 military ops are spent regardless."""
        t = Table(turn=2, defcon=5)
        t.inf(usa={"Angola": 3})
        t.dice(1)
        t.play(USSR, "Nasser", "coup")  # 1 + 1 = 2, not > 2 x stability 1
        assert t.s.inf(USA, "Angola") == 3 and t.s.defcon == 4
        assert t.s.military_ops[USSR] == 1


class TestRealignment:
    def test_requires_opponent_influence(self):
        """FAQ §6.2.1: "at least one enemy influence must be present within a nation in
        order to attempt a realignment." A country holding only the realigning player's
        own influence is not a legal target."""
        t = Table(turn=2, defcon=5)
        t.inf(ussr={"Angola": 2})
        assert not rules.can_realign(t.s, USSR, "Angola")
        assert rules.can_realign(t.s, USA, "Angola")

    def test_modifiers_rulebook_example(self):
        """R 6.2.2 EXAMPLE: US realigns North Korea (USSR 3): the US has no modifiers;
        "The USSR player has +1 because North Korea is adjacent to the USSR and +1 for
        having more Influence". US 5 vs USSR 2+2 = 4: the USSR loses 1."""
        t = Table(turn=2, defcon=5)
        t.inf(ussr={"North Korea": 3})
        t.dice(2, 5)  # resolve_realignment rolls USSR first, then US
        result = rules.resolve_realignment(t.s, USA, "North Korea", rolls=(2, 5))
        assert result.modifiers == (2, 0)
        assert t.s.inf(USSR, "North Korea") == 2

    def test_never_adds_influence_and_ties_do_nothing(self):
        """R 6.2.2 "Ties are considered a draw"; R 6.2.3 "No Influence is ever added to a
        country as a result of a Realignment roll.\""""
        t = Table(turn=2)
        t.inf(usa={"Angola": 1})
        rules.resolve_realignment(t.s, USSR, "Angola", rolls=(6, 1))
        assert t.s.inf(USA, "Angola") == 0 and t.s.inf(USSR, "Angola") == 0

    def test_same_country_may_be_targeted_repeatedly(self):
        """R 6.2.1: "Countries may be targeted for Realignment more than once per Action
        Round.\""""
        t = Table(turn=2, defcon=5)
        t.inf(usa={"Angola": 5}, ussr={"Zaire": 1})
        t.dice(1, 1, 1, 1)
        t.play(USSR, "Decolonization", "realign", "country:Angola")
        assert t.d is not None and "country:Angola" in t.keys()


class TestOpponentEvents:
    def test_phasing_player_chooses_event_timing(self):
        """R 5.2: "The phasing player always decides whether the event is to take place
        before or after the Operations are conducted.\""""
        t = Table(turn=2)
        t.inf(usa={"Mexico": 1})
        t.play(USA, "Fidel", "influence")
        assert t.d.type is DecisionType.CONFIRM and t.d.player is USA

    def test_event_after_operations(self):
        """R 5.2: choosing "after": the US places into Cuba first (US 1 -> 3), then Fidel
        removes all of it."""
        t = Table(turn=2)
        t.inf(usa={"Cuba": 1})
        t.play(USA, "Fidel", "influence", "no", "country:Cuba", "country:Cuba")
        t.finish_with()
        assert t.s.inf(USA, "Cuba") == 0 and t.s.inf(USSR, "Cuba") == 3

    def test_modifiers_aggregate(self):
        """R 7.4: "These modifiers should be applied in aggregate". Containment (+1) and
        Red Scare against the US (-1) cancel out."""
        t = Table(turn=2)
        t.effect("Containment", USA, expires="end_of_turn")
        t.effect("Red Scare/Purge", USSR, expires="end_of_turn")
        assert t.g.effective_ops("Duck and Cover", USA) == 3

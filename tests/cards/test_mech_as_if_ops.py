"""Events worded "Conduct Operations as if they played an N Ops card" (V:B8).

The Deluxe rule that governs them (7.4.3, cited by FAQ #26/#57/#62) is not in any
available source, so the expectation here is the coordinator's chosen reading: such
operations are a normal N-op play -- the player's own ops modifiers apply (R 7.4.2
Example 3: Containment makes CIA Created worth 2), a coup counts toward required
military operations (R 6.3.3 / 8.2.3), and the DEFCON geography restrictions apply (they
are not "free" coups, R 6.3.5). Every test carries ``unverified_rule``.
"""

from __future__ import annotations

import pytest

from _table import USA, USSR, Table
from twilight.decisions import DecisionType

pytestmark = pytest.mark.unverified_rule


def count_placements(t: Table, country: str) -> int:
    if t.d is not None and t.d.type is DecisionType.CARD_USE:
        t.answer("use:influence")
    n = 0
    while t.d is not None and t.d.type is DecisionType.PLACE_INFLUENCE:
        t.answer(f"country:{country}")
        n += 1
    return n


class TestCIACreatedAsIf:
    def test_containment_makes_it_two(self):
        """R 7.4.2 EXAMPLE 3: "If the US player played 'Containment' earlier in the turn,
        he could play 'CIA Created' subsequently and use 2 Ops.\""""
        t = Table(turn=2)
        t.effect("Containment", USA, expires="end_of_turn")
        t.play(USA, "CIA Created", "event")
        assert count_placements(t, "Mexico") == 2

    def test_coup_counts_military_ops(self):
        """R 6.3.3 / 8.2.3: a coup with the event's 1 op credits 1 military op."""
        t = Table(turn=2)
        t.inf(ussr={"Angola": 1})
        t.dice(1)
        t.play(USA, "CIA Created", "event")
        if t.d is not None and t.d.type is DecisionType.CARD_USE:
            t.answer("use:coup")
        assert t.s.military_ops[USA] == 1

    def test_defcon_restricts_the_coup(self):
        """Not a free coup (R 6.3.5): at DEFCON 4 no coup in Europe."""
        t = Table(turn=2, defcon=4)
        t.inf(ussr={"Italy": 1, "Angola": 1})
        t.dice(6)
        t.play(USA, "CIA Created", "event")
        if t.d is not None and t.d.type is DecisionType.CARD_USE:
            t.answer("use:coup")
        if t.d is not None:
            assert "country:Italy" not in t.keys()
            t.answer("country:Angola")
        assert t.s.inf(USSR, "Italy") == 1 and t.s.inf(USSR, "Angola") == 0


class TestLoneGunmanAsIf:
    def test_brezhnev_makes_it_two(self):
        """Same reading as R 7.4.2 Ex.3 for the USSR: Brezhnev Doctrine +1."""
        t = Table(turn=5)
        t.effect("Brezhnev Doctrine", USSR, expires="end_of_turn")
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Lone Gunman", "event")
        assert count_placements(t, "Poland") == 2

    def test_coup_counts_military_ops(self):
        t = Table(turn=5)
        t.inf(usa={"Zimbabwe": 1})
        t.dice(1)
        t.play(USSR, "Lone Gunman", "event")
        if t.d is not None and t.d.type is DecisionType.CARD_USE:
            t.answer("use:coup")
        assert t.s.military_ops[USSR] == 1


class TestABMTreatyAsIf:
    def test_coup_counts_four_military_ops(self):
        t = Table(turn=5, defcon=5)
        t.inf(usa={"Zimbabwe": 1})
        t.dice(1)
        t.play(USSR, "ABM Treaty", "event")
        if t.d is not None and t.d.type is DecisionType.CARD_USE:
            t.answer("use:coup")
        assert t.s.military_ops[USSR] == 4

    def test_red_scare_reduces_it(self):
        """FAQ #31: Deluxe reworded event ops ("as if ... card") so that modifiers apply;
        US Red Scare against the USSR -> 3."""
        t = Table(turn=5)
        t.effect("Red Scare/Purge", USA, expires="end_of_turn")
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "ABM Treaty", "event")
        assert count_placements(t, "Poland") == 3


class TestOlympicBoycottAsIf:
    def test_red_scare_reduces_it(self):
        """FAQ #31: "Only the 4 OPs granted by Olympics Boycott in the previous Edition
        were unaffected, but has now been reworded (7.4.3)" -> affected in Deluxe."""
        t = Table(turn=2, defcon=5)
        t.effect("Red Scare/Purge", USA, expires="end_of_turn")
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Olympic Games", "event", "option:1")
        assert count_placements(t, "Poland") == 3


class TestGlasnostAsIf:
    def test_red_scare_reduces_it(self):
        t = Table(turn=8)
        t.effect("The Reformer", USSR)
        t.effect("Red Scare/Purge", USA, expires="end_of_turn")
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Glasnost", "event")
        assert count_placements(t, "Poland") == 3

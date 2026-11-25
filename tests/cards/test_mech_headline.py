"""Headline phase: order, opponent events, Defectors, Man in Earth Orbit (R 4.5 C)."""

from __future__ import annotations

from _table import USA, USSR, Table
from twilight import observe
from twilight.decisions import DecisionType
from twilight.enums import WinReason


def headline(t: Table, ussr_card: str, usa_card: str) -> Table:
    t.headline_phase()
    picks = {USSR: ussr_card, USA: usa_card}
    while t.d is not None and t.d.type is DecisionType.HEADLINE:
        t.answer(f"card:{picks[t.d.player]}")
    return t


class TestHeadlineOrder:
    def test_higher_ops_resolves_first(self):
        """R 4.5 C: "The card with the higher Headline Value takes effect first." USSR
        COMECON (3) resolves before US Independent Reds (2): the US then matches the
        USSR's freshly added influence in Hungary (1 -> US 1)."""
        t = Table(turn=2)
        t.hand(USSR, "COMECON", "Fidel")
        t.hand(USA, "Independent Reds", "Duck and Cover")
        headline(t, "COMECON", "Independent Reds")
        # COMECON first: 1 USSR in four EE countries.
        t.answer("country:Hungary", "country:Romania", "country:Bulgaria", "country:Yugoslavia")
        t.finish_with(lambda d: "country:Hungary" if "country:Hungary" in d.legal_keys else d.legal_keys[0])
        assert t.s.inf(USSR, "Hungary") == 1 and t.s.inf(USA, "Hungary") == 1

    def test_tie_goes_to_the_us(self):
        """R 4.5 C: "In the event of a tie, the Headline Event played by the US player
        goes into effect first." Both 1-op: US Truman Doctrine clears Italy (USSR 1,
        uncontrolled) before USSR Nasser."""
        t = Table(turn=2)
        t.inf(ussr={"Italy": 1})
        t.hand(USSR, "Nasser", "Fidel")
        t.hand(USA, "Truman Doctrine", "Duck and Cover")
        headline(t, "Nasser", "Truman Doctrine")
        idx_truman = next(i for i, e in enumerate(t.s.log) if "Truman Doctrine" in e.text and "resolves" in e.text)
        idx_nasser = next(i for i, e in enumerate(t.s.log) if "Nasser" in e.text and "resolves" in e.text)
        assert idx_truman < idx_nasser

    def test_both_scoring_us_first(self):
        """R 4.5 C: "If both players select a scoring card as their Headline Cards, the US
        player's scoring card takes effect first.\""""
        t = Table(turn=2)
        t.hand(USSR, "Asia Scoring", "Fidel")
        t.hand(USA, "Europe Scoring", "Duck and Cover")
        headline(t, "Asia Scoring", "Europe Scoring")
        i_eu = next(i for i, e in enumerate(t.s.log) if "scored Europe" in e.text)
        i_as = next(i for i, e in enumerate(t.s.log) if "scored Asia" in e.text)
        assert i_eu < i_as

    def test_opponent_event_is_implemented_by_the_opponent(self):
        """R 4.5 C NOTE: "If playing an opponent's event during the Headline phase, your
        opponent implements the event text as if they had played the card themselves."
        The US headlines Fidel: the USSR gets Cuba."""
        t = Table(turn=2)
        t.inf(usa={"Cuba": 1})
        t.hand(USSR, "Nasser", "Allende")
        t.hand(USA, "Fidel", "Duck and Cover")
        headline(t, "Nasser", "Fidel")
        t.finish_with()
        assert t.s.inf(USSR, "Cuba") == 3 and t.s.inf(USA, "Cuba") == 0
        assert t.gone_for_good("Fidel")

    def test_headliner_carries_defcon_blame(self):
        """R 4.5 C NOTE: "the player of the headline card would be considered the phasing
        player for purposes of the DEFCON status". USSR headlines Duck and Cover at
        DEFCON 2: the US implements it, DEFCON 1, the USSR loses."""
        t = Table(turn=2, defcon=2)
        t.hand(USSR, "Duck and Cover", "Fidel")
        t.hand(USA, "CIA Created", "Nasser")
        headline(t, "Duck and Cover", "CIA Created")
        assert t.s.is_over and t.s.winner is USA and t.s.win_reason is WinReason.DEFCON

    def test_no_operations_from_headlines(self):
        """R 4.5 C: "neither player receives operations points from cards played during
        the headline phase" -- a headlined card with an unplayable event is just
        discarded (NATO without prerequisite)."""
        t = Table(turn=2)
        t.hand(USSR, "Nasser", "Fidel")
        t.hand(USA, "NATO", "Duck and Cover")
        headline(t, "Nasser", "NATO")
        t.finish_with()
        assert not t.s.in_play("NATO") and t.in_cycle("NATO")
        assert all(t.s.inf(USA, n) == 0 for n in ("Canada", "Mexico"))


class TestDefectors:
    def test_cancels_ussr_headline_even_with_higher_ops(self):
        """PD #103: "Play in Headline Phase to cancel USSR Headline event, including
        Scoring Card. Cancelled card returns to the Discard Pile." FAQ #103: cancelled
        even if the USSR headline has the higher ops."""
        t = Table(turn=2)
        t.inf(usa={"Cuba": 1})
        t.hand(USSR, "Fidel", "Nasser")
        t.hand(USA, "Defectors", "Duck and Cover")
        t.s.hands[USSR].remove("Fidel")
        t.hand(USSR, "COMECON")
        headline(t, "COMECON", "Defectors")
        assert t.done
        assert all(t.s.inf(USSR, n) == 0 for n in ("Hungary", "Romania", "Poland"))
        assert t.in_cycle("COMECON"), "cancelled starred card goes to the discard pile"
        assert t.in_cycle("Defectors"), "Defectors is not starred (errata)"

    def test_cancels_a_scoring_card(self):
        """PD #103 "including Scoring Card"."""
        t = Table(turn=2)
        t.inf(ussr={"Iraq": 3})
        t.hand(USSR, "Middle East Scoring", "Nasser")
        t.hand(USA, "Defectors", "Duck and Cover")
        headline(t, "Middle East Scoring", "Defectors")
        assert t.s.vp == 0 and t.in_cycle("Middle East Scoring")

    def test_ussr_headlining_defectors_does_nothing(self):
        """FAQ #103: "What happens if the Soviet player plays Defectors in the headline
        phase? A. Nothing.\""""
        t = Table(turn=2)
        t.inf(usa={"Cuba": 1})
        t.hand(USSR, "Defectors", "Nasser")
        t.hand(USA, "Truman Doctrine", "Duck and Cover")
        headline(t, "Defectors", "Truman Doctrine")
        t.finish_with()
        assert t.s.vp == 0
        assert t.in_cycle("Defectors")


class TestManInEarthOrbit:
    def test_opponent_chooses_first(self):
        """R 6.4.4: "Upon reaching space 4 (Man in Earth Orbit), the opposing player must
        select and reveal his or her Headline Event before the player with a 'Man in
        Earth Orbit' makes his/her Headline Event selection.\""""
        t = Table(turn=2)
        t.s.space_race = [4, 2]
        t.hand(USSR, "Fidel", "Nasser")
        t.hand(USA, "Duck and Cover", "CIA Created")
        t.headline_phase()
        assert t.d.type is DecisionType.HEADLINE and t.d.player is USA

    def test_holder_sees_the_opponent_headline(self):
        """R 6.4.4 "select and *reveal*" (V:B4): when the holder (USSR) is asked for its
        headline, the US choice must be visible to it -- in the decision, or in its
        observation (log)."""
        t = Table(turn=2)
        t.s.space_race = [4, 2]
        t.hand(USSR, "Fidel", "Nasser")
        t.hand(USA, "Duck and Cover", "CIA Created")
        t.headline_phase()
        t.answer("card:Duck and Cover")
        assert t.d.type is DecisionType.HEADLINE and t.d.player is USSR
        obs = observe(t.s, USSR, t.d)
        visible = [t.d.prompt, repr(t.d.context), *obs.log_tail]
        assert any("Duck and Cover" in v for v in visible)

    def test_ability_lost_when_opponent_reaches_box_4(self):
        """R 6.4.4: "The special effect is immediately cancelled when the second player
        reaches that box." Both at 4: normal order (USSR picks first)."""
        t = Table(turn=2)
        t.s.space_race = [4, 4]
        t.hand(USSR, "Fidel", "Nasser")
        t.hand(USA, "Duck and Cover", "CIA Created")
        t.headline_phase()
        assert t.d.player is USSR


class TestHeadlineMisc:
    def test_must_headline_a_card(self):
        """R 4.5 C: "Players must create a Headline event" -- no pass option."""
        t = Table(turn=2)
        t.hand(USSR, "Fidel", "Nasser")
        t.hand(USA, "Duck and Cover", "CIA Created")
        t.headline_phase()
        assert "pass" not in t.keys()

    def test_trap_does_not_affect_headline(self):
        """FAQ #42/#44: "Remember that neither card affects headline phase." The trapped
        US still headlines normally."""
        t = Table(turn=5)
        t.effect("Quagmire", USA)
        t.hand(USSR, "Fidel", "Nasser")
        t.hand(USA, "Duck and Cover", "CIA Created")
        t.headline_phase()
        t.answer("card:Fidel")
        assert t.d.type is DecisionType.HEADLINE and t.d.player is USA

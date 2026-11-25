"""Defectors (#103) and the optional cards #104-#110.

Card text quoted from the official digital edition text (``PD``, the Deluxe card faces)
as collected in ``scratchpad/rules/cards.md`` (``TS`` = twilightstrategy.com transcription
from the same file); rule numbers are the Deluxe rulebook (``R``); ``FAQ`` is the
designer-authorised FAQ v5 (2010); ``V:Bn`` points at an entry in ``verdicts.md``.
Expectations follow those sources, not the current implementation. ``Table`` builds
games with ``optional_cards=True``.
"""

from __future__ import annotations

import pytest

from _table import USA, USSR, Table
from twilight import rules
from twilight.decisions import DecisionType


def _use_keys(t: Table) -> tuple[str, ...]:
    assert t.d is not None and t.d.type is DecisionType.CARD_USE, t.d
    return t.keys()


# --------------------------------------------------------------------------- #
# #103 Defectors
# --------------------------------------------------------------------------- #


class TestC103Defectors:
    def test_headline_cancels_ussr_scoring_card(self):
        """PD #103: "Play in Headline Phase to cancel USSR Headline event, including
        Scoring Card. Cancelled card returns to the Discard Pile.\""""
        t = Table(turn=2)
        t.inf(ussr={"North Korea": 3, "Afghanistan": 2})
        t.hand(USSR, "Asia Scoring")
        t.hand(USA, "Defectors")
        t.headline_phase()
        t.answer("card:Asia Scoring", "card:Defectors")
        assert t.done
        assert t.s.vp == 0, "Asia was not scored"
        assert t.in_cycle("Asia Scoring")
        assert t.in_cycle("Defectors"), "not starred (FAQ errata 1)"

    def test_cancels_even_a_higher_ops_headline(self):
        """FAQ #103: "If the USSR Headline event has a higher Ops number than Defectors,
        does it still get canceled? A: Yes." Socialist Governments (3 ops) is cancelled
        and goes to the discard pile."""
        t = Table(turn=2)
        t.inf(usa={"Italy": 3, "France": 2})
        t.hand(USSR, "Socialist Governments")
        t.hand(USA, "Defectors")
        t.headline_phase()
        t.answer("card:Socialist Governments", "card:Defectors")
        assert t.done
        assert t.s.inf(USA, "Italy") == 3 and t.s.inf(USA, "France") == 2
        assert t.in_cycle("Socialist Governments")

    def test_ussr_headlining_defectors_does_nothing(self):
        """FAQ #103: "What happens if the Soviet player plays Defectors in the headline
        phase? A. Nothing.\""""
        t = Table(turn=2)
        t.hand(USSR, "Defectors")
        t.hand(USA, "Panama Canal Returned")
        t.headline_phase()
        t.answer("card:Defectors", "card:Panama Canal Returned")
        assert t.done
        assert t.s.vp == 0
        assert t.s.inf(USA, "Panama") == 1, "the US headline still resolves"
        assert t.in_cycle("Defectors")

    def test_ussr_action_round_play_gives_us_a_vp(self):
        """PD #103: "If Defectors played by USSR during Soviet action round, US gains 1
        VP".\""""
        t = Table(turn=2)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Defectors", "influence", "yes", "country:Poland", "country:Poland")
        assert t.done
        assert t.s.vp == -1
        assert t.in_cycle("Defectors")

    def test_space_race_discard_gives_nothing(self):
        """PD #103: "(unless played on the Space Race)"; FAQ #103 / R 6.4.5."""
        t = Table(turn=2)
        t.dice(6)
        t.play(USSR, "Defectors", "space")
        assert t.done and t.s.vp == 0 and t.s.space_race[USSR] == 0


# --------------------------------------------------------------------------- #
# #104 The Cambridge Five
# --------------------------------------------------------------------------- #


class TestC104CambridgeFive:
    def test_adds_one_in_a_region_named_by_a_revealed_scoring_card(self):
        """PD #104: "The US player exposes all scoring cards in their hand. The USSR
        player may then add 1 Influence in any single region named on one of those
        scoring cards (USSR choice).\""""
        t = Table(turn=2)
        t.hand(USA, "Asia Scoring", "Duck and Cover")
        t.play(USSR, "The Cambridge Five", "event")
        assert t.d is not None and t.d.type is DecisionType.PLACE_INFLUENCE
        assert "country:Japan" in t.keys()
        assert "country:Iraq" not in t.keys(), "Middle East not named"
        t.answer("country:Japan")
        assert t.done
        assert t.s.inf(USSR, "Japan") == 1
        assert sum(t.s.influence[USSR]) == 1
        assert t.in_cycle("The Cambridge Five"), "not starred"

    def test_not_playable_as_event_in_late_war(self):
        """PD #104: "Cannot be played as an event in Late War." (turn 8+)"""
        t = Table(turn=8)
        t.hand(USA, "Asia Scoring")
        t.inf(ussr={"Poland": 1})
        t.hand(USSR, "The Cambridge Five")
        t.action_round(USSR).answer("card:The Cambridge Five")
        assert "use:event" not in _use_keys(t)

    def test_no_scoring_cards_no_influence(self):
        """PD #104: only regions named on revealed scoring cards; none held -> nothing."""
        t = Table(turn=2)
        t.hand(USA, "Duck and Cover")
        t.play(USSR, "The Cambridge Five", "event")
        assert t.done and sum(t.s.influence[USSR]) == 0

    def test_us_ops_play_triggers_it(self):
        """R 5.2: the US playing it for ops lets the USSR implement the event."""
        t = Table(turn=2)
        t.hand(USA, "Europe Scoring")
        t.play(USA, "The Cambridge Five", "influence", "yes")
        assert t.d is not None and t.d.player is USSR
        t.answer("country:Italy")
        t.finish_with(lambda d: "country:Mexico" if "country:Mexico" in d.legal_keys else d.legal_keys[0])
        assert t.s.inf(USSR, "Italy") == 1


# --------------------------------------------------------------------------- #
# #105 Special Relationship
# --------------------------------------------------------------------------- #


class TestC105SpecialRelationship:
    def test_without_nato_one_influence_adjacent_to_uk(self):
        """PD #105: "If UK is US controlled but NATO is not in effect, US adds 1
        Influence to any country adjacent to the UK.\""""
        t = Table(turn=2)
        t.inf(usa={"UK": 5})
        t.play(USA, "Special Relationship", "event")
        assert set(t.keys()) == {"country:Canada", "country:Norway", "country:Benelux", "country:France"}
        t.answer("country:France")
        assert t.done
        assert t.s.inf(USA, "France") == 1 and t.s.vp == 0
        assert t.in_cycle("Special Relationship"), "not starred"

    def test_with_nato_two_in_western_europe_and_two_vp(self):
        """PD #105: "If UK is US controlled and NATO is in effect, US adds 2 Influence to
        any Western European country and gains 2 VPs.\""""
        t = Table(turn=2)
        t.effect("NATO", USA)
        t.inf(usa={"UK": 5})
        t.play(USA, "Special Relationship", "event")
        assert "country:Poland" not in t.keys()
        t.answer("country:Italy")
        assert t.done
        assert t.s.inf(USA, "Italy") == 2 and t.s.vp == -2

    def test_requires_us_control_of_uk(self):
        """PD #105: both clauses require "UK is US controlled"."""
        t = Table(turn=2)
        t.inf(usa={"UK": 5}, ussr={"UK": 1})
        assert not rules.controls(t.s, USA, "UK")
        t.hand(USA, "Special Relationship")
        t.action_round(USA).answer("card:Special Relationship")
        assert "use:event" not in _use_keys(t)


# --------------------------------------------------------------------------- #
# #106 NORAD
# --------------------------------------------------------------------------- #


def _norad_turn(canada: int) -> Table:
    """Turn 2 at DEFCON 3 with NORAD in play. Headlines: Fidel / Panama Canal Returned
    (neither asks anything). The USSR's first action round plays Duck and Cover for
    ops, the US event first: DEFCON 3 -> 2 during that action round."""
    t = Table(turn=2, defcon=3, china_face_up=False)
    t.effect("NORAD", USA)
    t.inf(usa={"Canada": canada, "Japan": 1}, ussr={"Poland": 1})
    t.hand(USSR, "Fidel", "Duck and Cover", "Nasser")
    t.hand(USA, "Panama Canal Returned", "CIA Created")
    t.play_turn()
    t.answer("card:Fidel", "card:Panama Canal Returned")
    assert t.d is not None and t.d.type is DecisionType.PLAY_CARD and t.d.player is USSR
    t.answer("card:Duck and Cover", "use:influence", "yes")
    assert t.s.defcon == 2
    t.answer("country:Poland", "country:Poland", "country:Poland")
    return t


class TestC106NORAD:
    def test_bonus_influence_when_defcon_moves_to_2(self):
        """PD #106: "If the US controls Canada, the US may add 1 Influence to any country
        already containing US Influence at the conclusion of any Action Round in which
        the DEFCON marker moves to the '2' box.\""""
        t = _norad_turn(canada=4)
        assert t.d is not None and t.d.player is USA
        assert t.d.type is DecisionType.PLACE_INFLUENCE
        with_us = {f"country:{n}" for n in ("Canada", "Japan", "Panama", "Costa Rica", "Venezuela")}
        assert set(k for k in t.keys() if k != "pass") <= with_us
        t.answer("country:Japan")
        assert t.s.inf(USA, "Japan") == 2

    def test_needs_us_control_of_canada(self):
        """PD #106: "If the US controls Canada" -- Canada 1 is not control (stab 4)."""
        t = _norad_turn(canada=1)
        assert t.d is not None and t.d.type is DecisionType.PLAY_CARD and t.d.player is USA

    def test_cancelled_by_quagmire(self):
        """PD #106: "This event cancelled by 'Quagmire'." NORAD is starred, so it then
        leaves the game."""
        t = Table(turn=4)
        t.effect("NORAD", USA)
        t.play(USSR, "Quagmire", "event")
        assert t.done
        assert not t.s.in_play("NORAD")
        assert t.gone_for_good("NORAD")

    def test_event_puts_it_in_play(self):
        """PD #106 is a lasting event (asterisked: never back in the draw cycle)."""
        t = Table(turn=2)
        t.play(USA, "NORAD", "event")
        assert t.done and t.s.in_play("NORAD")
        assert t.gone_for_good("NORAD")


# --------------------------------------------------------------------------- #
# #107 Che
# --------------------------------------------------------------------------- #


class TestC107Che:
    def test_second_coup_after_removing_us_influence(self):
        """PD #107: "USSR may immediately make a Coup attempt using this card's Operations
        value against a non-battleground country in Central America, South America, or
        Africa. If the Coup removes any US Influence, USSR may make a second Coup attempt
        against a different target under the same restrictions." Cameroon (stab 1):
        6+3-2 = 7 removes the US point; then Zimbabwe is offered but not Cameroon or
        battleground Nigeria."""
        t = Table(turn=5)
        t.inf(usa={"Cameroon": 1, "Zimbabwe": 1, "Nigeria": 1, "Italy": 1})
        t.dice(6, 1)
        t.play(USSR, "Che", "event")
        assert "country:Nigeria" not in t.keys(), "battleground"
        assert "country:Italy" not in t.keys(), "Europe"
        t.answer("country:Cameroon")
        assert t.s.inf(USA, "Cameroon") == 0
        assert t.d is not None, "a second coup is offered"
        assert "country:Cameroon" not in t.keys() and "country:Zimbabwe" in t.keys()
        assert "country:Nigeria" not in t.keys()
        t.answer("country:Zimbabwe")
        assert t.done
        assert t.in_cycle("Che"), "not starred"

    def test_no_second_coup_if_nothing_removed(self):
        """PD #107: second attempt only "If the Coup removes any US Influence". Honduras
        (stab 2): 1+3-4 = 0 -> fail."""
        t = Table(turn=5)
        t.inf(usa={"Honduras": 1, "Zimbabwe": 1})
        t.dice(1)
        t.play(USSR, "Che", "event", "country:Honduras")
        assert t.done
        assert t.s.inf(USA, "Honduras") == 1 and t.s.inf(USA, "Zimbabwe") == 1

    @pytest.mark.skip(
        reason="V:B8 table: Che's wording does not say 'free'; whether its coups count "
        "toward required military operations is marked 未能核实 in verdicts.md."
    )
    def test_military_ops_credit(self):
        """V:B8 #107: unverified."""


# --------------------------------------------------------------------------- #
# #108 Our Man In Tehran
# --------------------------------------------------------------------------- #


class TestC108OurManInTehran:
    def test_draw_five_discard_any_return_rest(self):
        """PD #108: "If the US controls at least one Middle East country, the US player
        draws the top 5 cards from the draw pile. They may reveal and then discard any or
        all of these drawn cards without triggering the Event. Any remaining drawn cards
        are returned to the draw deck, and it is reshuffled.\""""
        t = Table(turn=5, defcon=4)
        t.inf(usa={"Israel": 4})
        deck = ["Duck and Cover", "Fidel", "Nasser", "Allende", "Summit", "Junta"]
        t.deck(*deck)
        t.play(USA, "Our Man In Tehran", "event")
        assert t.d is not None and t.d.player is USA
        offered = {k[5:] for k in t.keys() if k.startswith("card:")}
        assert offered == set(deck[:5]), "the top five cards"
        t.answer("card:Duck and Cover", "card:Fidel", "pass")
        assert t.done
        assert t.in_cycle("Duck and Cover") and t.in_cycle("Fidel")
        assert t.s.defcon == 4, "Duck and Cover's event did not occur"
        assert sorted(t.s.deck) == sorted(["Nasser", "Allende", "Summit", "Junta"])
        assert t.s.hands[USA] == []
        assert t.gone_for_good("Our Man In Tehran")

    def test_requires_us_controlled_middle_east_country(self):
        """PD #108: "If the US controls at least one Middle East country"."""
        t = Table(turn=5)
        t.inf(usa={"Israel": 1})
        t.deck("Fidel")
        t.hand(USA, "Our Man In Tehran")
        t.action_round(USA).answer("card:Our Man In Tehran")
        assert "use:event" not in _use_keys(t)

    def test_ussr_ops_play_without_condition_discards_it(self):
        """R 5.2 bullet 2: condition unmet -> the event does not occur and the asterisked
        card goes to the discard pile."""
        t = Table(turn=5)
        t.inf(ussr={"Poland": 1})
        t.deck("Fidel", "Nasser")
        t.play(USSR, "Our Man In Tehran", "influence")
        t.finish_with(lambda d: "country:Poland" if "country:Poland" in d.legal_keys else d.legal_keys[0])
        assert t.in_cycle("Our Man In Tehran")
        assert len(t.s.deck) == 2


# --------------------------------------------------------------------------- #
# #109 Yuri and Samantha
# --------------------------------------------------------------------------- #


class TestC109YuriAndSamantha:
    def test_ussr_scores_per_us_coup_attempt(self):
        """PD #109: "USSR receives 1 VP for each US coup attempt made for the remainder of
        the current turn." A failed attempt still counts."""
        t = Table(turn=8)
        t.play(USSR, "Yuri and Samantha", "event")
        assert t.s.in_play("Yuri and Samantha")
        t.inf(ussr={"Morocco": 2})
        t.dice(1)
        t.play(USA, "Duck and Cover", "coup")  # Morocco is the only target
        assert t.done
        assert t.s.inf(USSR, "Morocco") == 2, "1+3 vs 6: the coup failed"
        assert t.s.vp == 1

    def test_event_granted_us_coup_also_scores(self):
        """PD #109: "each US coup attempt" -- including the free coup of Tear Down This
        Wall (DEFCON 5, Poland)."""
        t = Table(turn=8, defcon=5)
        t.effect("Yuri and Samantha", USSR, expires="end_of_turn")
        t.inf(ussr={"Poland": 2})
        t.dice(1)
        t.play(USA, "Tear Down This Wall", "event")
        coup = next(a.key for a in t.d.options if "coup" in a.label)
        t.answer(coup, "country:Poland")
        assert t.s.vp == 1

    def test_ussr_coups_do_not_score_and_effect_ends_with_turn(self):
        """PD #109: only US coups; "for the remainder of the current turn"."""
        t = Table(turn=8)
        t.effect("Yuri and Samantha", USSR, expires="end_of_turn")
        t.inf(usa={"Angola": 1})
        t.dice(6)
        t.play(USSR, "Fidel", "coup")
        assert t.s.vp == 0
        t.turn_end()
        assert not t.s.in_play("Yuri and Samantha")
        assert t.gone_for_good("Yuri and Samantha")


# --------------------------------------------------------------------------- #
# #110 AWACS Sale to Saudis
# --------------------------------------------------------------------------- #


class TestC110AWACS:
    def test_two_in_saudi_arabia(self):
        """PD #110: "US receives 2 Influence in Saudi Arabia.\""""
        t = Table(turn=8)
        t.play(USA, "AWACS Sale to Saudis", "event")
        assert t.done and t.s.inf(USA, "Saudi Arabia") == 2
        assert t.gone_for_good("AWACS Sale to Saudis")

    def test_muslim_revolution_no_longer_playable(self):
        """PD #110: "Muslim Revolution may no longer be played as an event." The US
        playing Muslim Revolution for ops afterwards triggers nothing (R 5.2)."""
        t = Table(turn=8)
        t.play(USA, "AWACS Sale to Saudis", "event")
        t.finish_with()
        t.inf(ussr={"Poland": 1}, usa={"Iran": 2, "Egypt": 1})
        t.hand(USSR, "Muslim Revolution")
        t.action_round(USSR).answer("card:Muslim Revolution")
        assert "use:event" not in _use_keys(t)
        # play_card has already taken Muslim Revolution out of the USSR hand.
        t.play(USA, "Muslim Revolution", "influence")
        t.finish_with(lambda d: "country:Mexico" if "country:Mexico" in d.legal_keys else d.legal_keys[0])
        assert t.s.inf(USA, "Iran") == 2 and t.s.inf(USA, "Egypt") == 1

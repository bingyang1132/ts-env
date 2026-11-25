"""The seven scoring cards (#1 #2 #3 #37 #38 #79 #81) and The China Card (#6).

Values are the printed scoring cards (PD event_text, e.g. #1 "Presence: 3 / Domination: 7
/ Control: 9 / +1 per controlled Battleground Country in Region / +1 per Country
controlled that is adjacent to enemy superpower") and R 10.1. Each test builds a region
with a known answer and plays the card in an action round.
"""

from __future__ import annotations

import pytest

from _table import USA, USSR, Table
from twilight.decisions import DecisionType
from twilight.enums import WinReason


def score(card: str, *, ussr=None, usa=None, vp: int = 0, turn: int = 5) -> Table:
    t = Table(turn=turn, vp=vp)
    t.inf(ussr=ussr, usa=usa)
    t.play(USSR, card)
    return t


# --------------------------------------------------------------------------- #
# #1 Asia Scoring
# --------------------------------------------------------------------------- #


class TestC001AsiaScoring:
    def test_presence_domination_and_bonuses(self):
        """PD #1: "Presence: 3 / Domination: 7 / Control: 9 / +1 per controlled
        Battleground Country in Region / +1 per Country controlled that is adjacent to
        enemy superpower". USSR controls North Korea, India (BGs) and Burma (non-BG,
        SE Asia counts as Asia, R 10.3.2) -> domination 7 + 2 BG = 9. US controls Japan
        (BG, adjacent to the USSR? no -- Japan is adjacent to the USA, so no bonus for
        the US) -> presence 3 + 1 = 4. Net USSR +5."""
        t = score(
            "Asia Scoring",
            ussr={"North Korea": 3, "India": 3, "Burma": 2},
            usa={"Japan": 4},
        )
        assert t.done and t.s.vp == 5
        assert t.in_cycle("Asia Scoring"), "not starred (R 7.2)"

    def test_adjacency_bonus(self):
        """PD #1 "+1 per Country controlled that is adjacent to enemy superpower": the
        US controlling North Korea (BG) and Afghanistan (non-BG), both adjacent to the
        USSR, dominates: 7 + 1 BG + 2 adjacency = 10."""
        t = score("Asia Scoring", usa={"North Korea": 3, "Afghanistan": 2})
        assert t.s.vp == -10

    def test_control(self):
        """PD #1 Control 9 (R 10.1.1: all battlegrounds and more countries). The US
        controls all six Asian battlegrounds: 9 + 6 + 1 (North Korea adjacent to the
        USSR) = 16."""
        bgs = {"Japan": 4, "India": 3, "North Korea": 3, "South Korea": 3, "Pakistan": 2, "Thailand": 2}
        t = score("Asia Scoring", usa=bgs)
        assert t.s.vp == -16

    def test_domination_needs_a_non_battleground(self):
        """R 10.1.1: "A superpower must Control at least one non-Battleground and one
        Battleground country in a Region in order to achieve Domination." Two BGs only
        and more countries than the US: presence (3) + 2 = 5, not domination."""
        t = score("Asia Scoring", ussr={"North Korea": 3, "India": 3})
        assert t.s.vp == 5


# --------------------------------------------------------------------------- #
# #2 Europe Scoring
# --------------------------------------------------------------------------- #


class TestC002EuropeScoring:
    def test_domination_and_adjacency(self):
        """PD #2: "Presence: 3 / Domination: 7 / Control: VICTORY / +1 per controlled
        Battleground Country / +1 per Country controlled that is adjacent to enemy
        superpower". US dominates: West Germany, France (BG) + Finland (adjacent to
        USSR, non-BG) -> 7 + 2 + 1 = 10; USSR presence Poland (BG): 3 + 1 = 4. Net US 6."""
        t = score(
            "Europe Scoring",
            usa={"West Germany": 4, "France": 3, "Finland": 4},
            ussr={"Poland": 3},
        )
        assert t.s.vp == -6 and not t.s.is_over

    def test_control_wins_immediately(self):
        """R 10.3.1: "If either side Controls Europe, that side wins when the Europe
        Scoring card is played." USSR controls all 5 European BGs and more countries."""
        t = score(
            "Europe Scoring",
            ussr={"East Germany": 3, "Poland": 3, "West Germany": 4, "France": 3, "Italy": 2},
        )
        assert t.s.is_over and t.s.winner is USSR
        assert t.s.win_reason is WinReason.EUROPE_CONTROL
        assert t.done


# --------------------------------------------------------------------------- #
# #3 Middle East Scoring
# --------------------------------------------------------------------------- #


class TestC003MiddleEastScoring:
    def test_domination_vs_presence(self):
        """PD #3: "Presence: 3 / Domination: 5 / Control: 7 / +1 per controlled
        Battleground Country in Region" (no adjacency clause). USSR: Iraq, Libya (BG),
        Syria -> 5 + 2 = 7; US: Israel -> 3 + 1 = 4. Net USSR +3."""
        t = score(
            "Middle East Scoring",
            ussr={"Iraq": 3, "Libya": 2, "Syria": 2},
            usa={"Israel": 4},
        )
        assert t.s.vp == 3

    def test_control(self):
        """PD #3 Control 7 + 6 battlegrounds = 13."""
        bgs = {"Israel": 4, "Iraq": 3, "Saudi Arabia": 3, "Egypt": 2, "Iran": 2, "Libya": 2}
        t = score("Middle East Scoring", ussr=bgs)
        assert t.s.vp == 13

    def test_more_battlegrounds_but_equal_countries_is_presence(self):
        """R 10.1.1 Domination needs more countries *and* more BGs. USSR: Iraq, Libya
        (2 BG); US: Jordan, Lebanon (2 non-BG). Equal countries -> both presence:
        USSR 3+2=5, US 3. Net USSR +2."""
        t = score(
            "Middle East Scoring",
            ussr={"Iraq": 3, "Libya": 2},
            usa={"Jordan": 2, "Lebanon": 1},
        )
        assert t.s.vp == 2


# --------------------------------------------------------------------------- #
# #37 Central America Scoring
# --------------------------------------------------------------------------- #


class TestC037CentralAmericaScoring:
    def test_rulebook_example(self):
        """R 10.1.2 EXAMPLE (with FAQ example errata): "The USSR controls Cuba, Haiti and
        the Dominican Republic. The United States controls Guatemala, and has 1 point of
        influence in Panama. The USSR player would therefore get points for Dominating
        Central America (3 VPs) + 1 VP for control of a battleground country (Cuba). +1
        VP for Cuba's being adjacent to your opponent's home nation for a total of 5
        VPs. The United States would receive 1 VP for presence ... move the VP point
        track a net 4 spaces toward Soviet victory.\""""
        t = score(
            "Central America Scoring",
            ussr={"Cuba": 3, "Haiti": 1, "Dominican Republic": 1},
            usa={"Guatemala": 1, "Panama": 1},
        )
        assert t.s.vp == 4

    def test_control(self):
        """PD #37: Presence 1 / Domination 3 / Control 5, +1 per BG, +1 adjacent to the
        enemy superpower. US controls Mexico, Cuba, Panama: 5 + 3 = 8 (Cuba/Mexico are
        adjacent to the USA, not to the USSR, so no US bonus)."""
        t = score("Central America Scoring", usa={"Mexico": 2, "Cuba": 3, "Panama": 2})
        assert t.s.vp == -8


# --------------------------------------------------------------------------- #
# #38 Southeast Asia Scoring
# --------------------------------------------------------------------------- #


class TestC038SoutheastAsiaScoring:
    def test_per_country_thailand_double_and_removed(self):
        """PD #38: 1 VP per controlled country in SE Asia, Thailand 2 VP; the only
        scoring card removed after play (R 7.2). USSR Thailand + Vietnam = 3, US
        Philippines = 1. Net USSR +2."""
        t = score(
            "Southeast Asia Scoring",
            ussr={"Thailand": 2, "Vietnam": 1},
            usa={"Philippines": 2},
        )
        assert t.s.vp == 2
        assert t.gone_for_good("Southeast Asia Scoring")


# --------------------------------------------------------------------------- #
# #79 Africa Scoring
# --------------------------------------------------------------------------- #


class TestC079AfricaScoring:
    def test_control(self):
        """PD #79: Presence 1 / Domination 4 / Control 6, +1 per BG. US controls all five
        African BGs (Algeria, South Africa, Angola, Nigeria, Zaire): 6 + 5 = 11."""
        bgs = {"Algeria": 2, "South Africa": 3, "Angola": 1, "Nigeria": 1, "Zaire": 1}
        t = score("Africa Scoring", usa=bgs)
        assert t.s.vp == -11

    def test_domination(self):
        """USSR: Angola, Zaire (BG), Kenya -> 4 + 2 = 6; US: Nigeria -> 1 + 1 = 2."""
        t = score(
            "Africa Scoring",
            ussr={"Angola": 1, "Zaire": 1, "Kenya": 2},
            usa={"Nigeria": 1},
        )
        assert t.s.vp == 4


# --------------------------------------------------------------------------- #
# #81 South America Scoring
# --------------------------------------------------------------------------- #


class TestC081SouthAmericaScoring:
    def test_presence_domination(self):
        """PD #81: Presence 2 / Domination 5 / Control 6, +1 per BG. USSR: Chile,
        Argentina (BG), Bolivia -> 5 + 2 = 7; US: Brazil -> 2 + 1 = 3. Net USSR +4."""
        t = score(
            "South America Scoring",
            ussr={"Chile": 3, "Argentina": 2, "Bolivia": 2},
            usa={"Brazil": 2},
        )
        assert t.s.vp == 4

    def test_control(self):
        """PD #81 Control 6 + 4 battlegrounds = 10."""
        t = score(
            "South America Scoring",
            usa={"Chile": 3, "Argentina": 2, "Brazil": 2, "Venezuela": 2},
        )
        assert t.s.vp == -10


# --------------------------------------------------------------------------- #
# Common to all scoring cards
# --------------------------------------------------------------------------- #


class TestScoringCardsCommon:
    @pytest.mark.parametrize(
        "card",
        [
            "Asia Scoring",
            "Europe Scoring",
            "Middle East Scoring",
            "Central America Scoring",
            "Southeast Asia Scoring",
            "Africa Scoring",
            "South America Scoring",
        ],
    )
    def test_scoring_card_offers_no_ops_or_space_use(self, card):
        """R 4.5 D / 10.1: a scoring card is played to score; it has no Operations value
        and cannot be used for operations or the space race."""
        t = Table(turn=5)
        t.inf(ussr={"Poland": 1})
        t.hand(USSR, card)
        t.action_round(USSR).answer(f"card:{card}")
        assert t.d is None or t.d.type is not DecisionType.CARD_USE

    def test_held_scoring_card_at_turn_end_loses(self):
        """R 4.5 D: "Scoring cards may never be held." R 11.1.2 / FAQ: a player found
        holding one "loses immediately"."""
        t = Table(turn=5)
        t.hand(USA, "Africa Scoring")
        t.turn_end()
        assert t.s.is_over and t.s.winner is USSR
        assert t.s.win_reason is WinReason.HELD_SCORING_CARD

    def test_both_holding_is_a_us_win(self):
        """FAQ §8.5: "The designers are American and have declared the US Wins if both
        players are discovered to have Held a Scoring Card.\""""
        t = Table(turn=5)
        t.hand(USA, "Africa Scoring")
        t.hand(USSR, "Asia Scoring")
        t.turn_end()
        assert t.s.is_over and t.s.winner is USA

    def test_scoring_card_can_be_headlined_and_resolves_after(self):
        """R 4.5 C: "Scoring cards may be played during the Headline Phase. However, they
        are considered to have a Headline Value of zero (0) and always take effect
        second." The US 2-op Independent Reds resolves before the USSR Europe Scoring:
        US influence matches the USSR in Hungary, breaking USSR control, so the USSR
        has no presence when Europe is scored."""
        t = Table(turn=2)
        t.inf(ussr={"Hungary": 3})
        t.hand(USSR, "Europe Scoring")
        t.hand(USA, "Independent Reds")
        t.headline_phase()
        t.answer("card:Europe Scoring", "card:Independent Reds")
        t.finish_with()
        assert t.s.inf(USA, "Hungary") == 3
        assert t.s.vp == 0

    def test_net_difference_only(self):
        """R 10.2.3 / 10.1.2: "the net difference between the two scores is marked on
        the Victory Point Track". Equal standing scores nothing."""
        t = score("Middle East Scoring", ussr={"Iraq": 3}, usa={"Iran": 2})
        assert t.s.vp == 0


# --------------------------------------------------------------------------- #
# #6 The China Card
# --------------------------------------------------------------------------- #


class TestC006TheChinaCard:
    def test_passes_face_down_to_the_opponent(self):
        """PD #6 "Pass to opponent after play." R 9.3: "When 'The China Card' is
        played, it is immediately handed to your opponent face down. It may not be played
        again by your opponent this turn. At the end of the turn, it is flipped face up.\""""
        t = Table(turn=2, china=USSR)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "The China Card", "influence")
        t.finish_with(lambda d: "country:Poland" if "country:Poland" in d.legal_keys else d.legal_keys[0])
        assert t.s.china_card_owner is USA and not t.s.china_card_face_up
        assert "The China Card" not in t.s.playable_hand(USA)
        t.turn_end()
        assert t.s.china_card_face_up

    def test_plus_one_when_all_in_asia(self):
        """PD #6: "+1 Operations value when all points are used in Asia." R 9.6
        (Southeast Asia included). 5 influence placed in Asia."""
        t = Table(turn=2, china=USSR)
        t.inf(ussr={"North Korea": 3})
        t.play(USSR, "The China Card", "influence")
        for _ in range(4):
            t.answer("country:South Korea")
        assert t.d is not None, "the bonus fifth point must be offered"
        t.answer("country:South Korea")
        assert t.done and t.s.inf(USSR, "South Korea") == 5

    def test_no_bonus_when_a_point_leaves_asia(self):
        """R 9.6: "all of the Operations Points on the card must be spent in Asia"."""
        t = Table(turn=2, china=USSR)
        t.inf(ussr={"North Korea": 3, "Poland": 1})
        t.play(USSR, "The China Card", "influence")
        t.answer("country:Poland", "country:South Korea", "country:South Korea", "country:South Korea")
        assert t.done

    def test_coup_in_asia_gets_five(self):
        """FAQ #6: "The bonus certainly applies if the coup is done in Asia." Roll 1 + 5
        ops vs Afghanistan stability 2 x 2 = 4: margin 2."""
        t = Table(turn=2, defcon=5, china=USSR)
        t.inf(usa={"Afghanistan": 1})
        t.dice(1)
        t.play(USSR, "The China Card", "coup")
        assert t.s.inf(USA, "Afghanistan") == 0 and t.s.inf(USSR, "Afghanistan") == 1
        assert t.s.military_ops[USSR] == 5

    def test_vietnam_revolts_and_red_scare_aggregate(self):
        """R 7.4 EXAMPLE: China Card all in SE Asia = 5, Vietnam Revolts +1 = 6, Red
        Scare/Purge -1 = 5 operations points."""
        t = Table(turn=2, china=USSR)
        t.effect("Vietnam Revolts", USSR, expires="end_of_turn")
        t.effect("Red Scare/Purge", USA, expires="end_of_turn")
        t.inf(ussr={"Vietnam": 2})
        t.play(USSR, "The China Card", "influence")
        placed = 0
        while t.d is not None and t.d.type is DecisionType.PLACE_INFLUENCE:
            t.answer("country:Laos/Cambodia" if "country:Laos/Cambodia" in t.keys() else "country:Vietnam")
            placed += 1
        assert placed == 5

    def test_cannot_be_headlined(self):
        """R 4.5 C / 9.5: "'The China Card' may not be played during the Headline
        Phase.\""""
        t = Table(turn=2, china=USSR)
        t.hand(USSR, "Fidel")
        t.hand(USA, "Duck and Cover")
        t.headline_phase()
        assert t.d.type is DecisionType.HEADLINE
        assert "card:The China Card" not in t.keys()

    def test_end_of_turn_10_vp(self):
        """PD #6: "+1 VP for the player holding this card at the end of Turn 10.\""""
        t = Table(turn=10, china=USA, china_face_up=False)
        t.turn_end()
        assert t.s.vp == -1

    def test_event_transfer_is_face_up(self):
        """R 9.4: "If 'The China Card' is passed as a result of an Event, the card is
        passed face up and may be played by the new owning player during the same turn"
        (Cultural Revolution)."""
        t = Table(turn=5, china=USA, china_face_up=True)
        t.play(USSR, "Cultural Revolution", "event")
        assert t.s.china_card_owner is USSR and t.s.china_card_face_up
        assert "The China Card" in t.s.playable_hand(USSR)

    def test_may_not_be_played_if_it_prevents_a_scoring_card(self):
        """R 9.5: "'The China Card' may not be played ... if it prevents the play of a
        Scoring card". Last USSR action round with a scoring card in hand: the China
        Card must not be offered."""
        t = Table(turn=2, action_round=6, china=USSR)
        t.inf(ussr={"Poland": 1})
        t.hand(USSR, "Asia Scoring")
        t.s.action_round = 6
        t.action_round(USSR)
        assert t.d.type is DecisionType.PLAY_CARD
        assert "card:The China Card" not in t.keys()

"""Mid War cards #59-#78 and #80 (#79 / #81 scoring cards live with the scoring tests).

Card text quoted from ``scratchpad/rules/cards.md``: ``PD`` is the official digital
(Deluxe) card face, ``TS`` the twilightstrategy.com transcription. Rule numbers are the
Deluxe rulebook (``R``); ``FAQ`` is the designer-authorised FAQ v5 (2010); ``V:Bn`` is
an entry in ``verdicts.md``. Expectations follow those sources, not the implementation.
"""

from __future__ import annotations

import pytest

from _table import USA, USSR, Table
from twilight import rules, spacerace
from twilight.data import CHINA_CARD
from twilight.decisions import DecisionType
from twilight.enums import WinReason


def _use_if_asked(t: Table, use: str) -> None:
    if t.d is not None and t.d.type is DecisionType.CARD_USE:
        t.answer(f"use:{use}")


# --------------------------------------------------------------------------- #
# #59 Flower Power
# --------------------------------------------------------------------------- #


class TestC059FlowerPower:
    def test_is_a_permanent_effect(self):
        """PD #59 (asterisked, underlined in R 7.3 example): stays in play once evented."""
        t = Table(turn=4)
        t.play(USSR, "Flower Power", "event")
        assert t.done
        assert t.s.in_play("Flower Power")
        assert t.gone_for_good("Flower Power")

    def test_us_war_card_event_scores_two(self):
        """PD #59: "USSR gains 2 VP for every subsequently US played 'war card' (played
        as an Event or Operations)". US plays Brush War for its event (fails)."""
        t = Table(turn=4)
        t.effect("Flower Power", USSR)
        t.dice(1)
        t.play(USA, "Brush War", "event", "country:Panama")
        t.finish_with()
        assert t.s.vp == 2

    def test_us_war_card_for_operations_scores_two(self):
        """PD #59: "(played as an Event or Operations)". US plays the USSR's Korean War
        for ops (the USSR war event also fires and fails on a 1)."""
        t = Table(turn=4)
        t.effect("Flower Power", USSR)
        t.dice(1)
        t.play(USA, "Korean War", "influence", "yes")
        t.finish_with()
        assert t.s.vp == 2

    def test_space_race_is_exempt(self):
        """PD #59: "unless played on the Space Race"."""
        t = Table(turn=4)
        t.effect("Flower Power", USSR)
        t.dice(6)
        t.play(USA, "Brush War", "space")
        assert t.done
        assert t.s.vp == 0

    def test_ussr_war_cards_do_not_score(self):
        """PD #59: only "US played" war cards count."""
        t = Table(turn=4)
        t.effect("Flower Power", USSR)
        t.dice(1)
        t.play(USSR, "Brush War", "event", "country:Panama")
        t.finish_with()
        assert t.s.vp == 0

    def test_us_headlined_war_card_scores_two(self):
        """PD #59 counts war cards "played as an Event"; a headline is an event play
        (R 4.5C: "These cards are called 'Headline cards' and their Events take place in
        this phase"). FAQ #59: "all war cards played for OPS or the EVENT will be subject
        to the 2 VP penalty"; only the Space Race is exempt."""
        t = Table(turn=4)
        t.effect("Flower Power", USSR)
        t.hand(USSR, "Fidel")
        t.hand(USA, "Indo-Pakistani War")
        t.dice(1)
        t.headline_phase()
        t.answer("card:Fidel", "card:Indo-Pakistani War")
        t.finish_with(lambda d: "country:India" if "country:India" in d.legal_keys else d.legal_keys[0])
        assert t.s.vp == 2

    def test_an_evil_empire_cancels_it(self):
        """PD #59: "This event cancelled by 'An Evil Empire'." PD #97: "Cancels/Prevents
        Flower Power. US gains 1 VP." After it, US war cards cost nothing."""
        t = Table(turn=8)
        t.effect("Flower Power", USSR)
        t.play(USA, "An Evil Empire", "event")
        t.finish_with()
        assert not t.s.in_play("Flower Power")
        assert t.s.vp == -1
        t.dice(1)
        t.play(USA, "Brush War", "event", "country:Panama")
        t.finish_with()
        assert t.s.vp == -1

    def test_an_evil_empire_prevents_it(self):
        """PD #97 "Prevents Flower Power": played first, Flower Power can no longer be an
        event."""
        t = Table(turn=8)
        t.resolved("An Evil Empire")
        t.hand(USSR, "Flower Power")
        t.action_round(USSR).answer("card:Flower Power")
        assert "use:event" not in t.keys()


# --------------------------------------------------------------------------- #
# #60 U2 Incident
# --------------------------------------------------------------------------- #


class TestC060U2Incident:
    def test_one_vp_and_leaves_with_the_turn(self):
        """PD #60: "USSR gains 1VP." Asterisked; its rider lasts "this turn"."""
        t = Table(turn=4)
        t.play(USSR, "U2 Incident", "event")
        assert t.done and t.s.vp == 1
        t.turn_end()
        assert not t.s.in_play("U2 Incident")
        assert t.gone_for_good("U2 Incident")

    def test_us_un_intervention_later_gives_another_vp(self):
        """PD #60: "If UN Intervention played later this turn as an Event, either by US
        or USSR, gain 1 additional VP." US plays UN Intervention with Fidel."""
        t = Table(turn=4)
        t.effect("U2 Incident", USSR, expires="end_of_turn")
        t.hand(USA, "Fidel")
        t.play(USA, "UN Intervention", "event")
        if t.d is not None and t.d.type is DecisionType.CHOOSE_CARD:
            t.answer("card:Fidel")
        t.finish_with()
        assert t.s.vp == 1

    def test_ussr_un_intervention_later_gives_another_vp(self):
        """PD #60: "either by US or USSR"."""
        t = Table(turn=4)
        t.effect("U2 Incident", USSR, expires="end_of_turn")
        t.hand(USSR, "Duck and Cover")
        t.play(USSR, "UN Intervention", "event")
        if t.d is not None and t.d.type is DecisionType.CHOOSE_CARD:
            t.answer("card:Duck and Cover")
        t.finish_with()
        assert t.s.vp == 1
        assert t.s.defcon == 5, "the cancelled Duck and Cover must not fire"

    def test_un_intervention_for_operations_does_not_count(self):
        """PD #60: only UN Intervention "played ... as an Event"."""
        t = Table(turn=4)
        t.effect("U2 Incident", USSR, expires="end_of_turn")
        t.play(USA, "UN Intervention", "influence")
        t.finish_with()
        assert t.s.vp == 0


# --------------------------------------------------------------------------- #
# #61 OPEC
# --------------------------------------------------------------------------- #


class TestC061OPEC:
    def test_one_vp_per_controlled_oil_country(self):
        """PD #61: "USSR gains 1VP for each of the following countries he controls: Egypt,
        Iran, Libya, Saudi Arabia, Iraq, Gulf States, and Venezuela." Not starred."""
        t = Table(turn=4)
        t.inf(ussr={"Egypt": 2, "Iran": 2, "Venezuela": 2, "Syria": 2, "Iraq": 1})
        t.play(USSR, "OPEC", "event")
        assert t.done and t.s.vp == 3, "Syria is not listed; Iraq is not controlled"
        assert t.in_cycle("OPEC")

    def test_us_ops_play_triggers_it(self):
        """R 5.2: the US playing the USSR card for ops still triggers the event."""
        t = Table(turn=4)
        t.inf(ussr={"Libya": 2})
        t.play(USA, "OPEC", "influence", "yes")
        t.finish_with()
        assert t.s.vp == 1

    def test_north_sea_oil_prevents_it(self):
        """PD #61: "Unplayable as an event if 'North Sea Oil' is in effect." R 5.2: the
        US playing it for ops then gets no event."""
        t = Table(turn=8)
        t.effect("North Sea Oil", USA)
        t.inf(ussr={"Libya": 2})
        t.hand(USSR, "OPEC")
        t.action_round(USSR).answer("card:OPEC")
        assert "use:event" not in t.keys()
        t = Table(turn=8)
        t.effect("North Sea Oil", USA)
        t.inf(ussr={"Libya": 2})
        t.play(USA, "OPEC", "influence")
        t.finish_with()
        assert t.s.vp == 0
        assert t.in_cycle("OPEC")


# --------------------------------------------------------------------------- #
# #62 Lone Gunman
# --------------------------------------------------------------------------- #


class TestC062LoneGunman:
    def test_one_op_of_operations_and_removed(self):
        """PD #62: "US player reveals his hand. Then the USSR may Conduct Operations as if
        they played a 1 Op card." Asterisked."""
        t = Table(turn=4)
        t.inf(ussr={"Poland": 1})
        t.hand(USA, "Duck and Cover")
        t.play(USSR, "Lone Gunman", "event")
        _use_if_asked(t, "influence")
        t.answer("country:Poland")
        assert t.done
        assert t.s.inf(USSR, "Poland") == 2
        assert t.gone_for_good("Lone Gunman")

    def test_respects_defcon_restrictions(self):
        """V:B13: "as if they played an N Ops card" operations are not free coups and
        are restricted by DEFCON (R 8.1.5). At DEFCON 2 no coup in the Middle East."""
        t = Table(turn=4, defcon=2)
        t.inf(ussr={"Poland": 1}, usa={"Iran": 1})
        t.play(USSR, "Lone Gunman", "event")
        assert t.d is None or "use:coup" not in t.keys()

    def test_us_ops_play_coup_to_defcon_1_loses_for_the_us(self):
        """FAQ #62: "If initiated by the US player, and the Soviet player uses the 1 OP to
        conduct a coup in a battleground country (thereby degrading to DEFCON 1), who
        loses? A. As the phasing player is the US player (he played the card) the US
        player loses." (V:B9.) Panama is a Central American battleground."""
        t = Table(turn=4, defcon=2)
        t.inf(usa={"Panama": 1})
        t.dice(1)
        t.play(USA, "Lone Gunman", "influence", "yes")
        _use_if_asked(t, "coup")
        if t.d is not None and t.d.type is DecisionType.COUP_TARGET:
            t.answer("country:Panama")
        assert t.s.defcon == 1
        assert t.s.is_over and t.s.winner is USSR and t.s.win_reason is WinReason.DEFCON

    # Lone Gunman military ops / Brezhnev: see test_mech_as_if_ops.py (unverified_rule).


# --------------------------------------------------------------------------- #
# #63 Colonial Rear Guards
# --------------------------------------------------------------------------- #


class TestC063ColonialRearGuards:
    def test_four_different_african_or_se_asian_countries(self):
        """PD #63: "Add 1 US Influence in each of four different African and/or Southeast
        Asian countries." Not starred."""
        t = Table(turn=4)
        t.play(USA, "Colonial Rear Guards", "event", "country:Angola")
        assert "country:Angola" not in t.keys(), "four *different* countries"
        assert "country:India" not in t.keys(), "India is not Southeast Asia"
        assert "country:Iran" not in t.keys()
        t.answer("country:Zaire", "country:Vietnam", "country:Thailand")
        assert t.done
        for n in ("Angola", "Zaire", "Vietnam", "Thailand"):
            assert t.s.inf(USA, n) == 1
        assert t.in_cycle("Colonial Rear Guards")

    def test_ussr_ops_play_triggers_it(self):
        """R 5.2: played by the USSR for ops, the US implements the event."""
        t = Table(turn=4)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Colonial Rear Guards", "influence", "yes")
        assert t.d is not None and t.d.player is USA
        t.answer("country:Angola", "country:Zaire", "country:Vietnam", "country:Thailand")
        t.finish_with()
        assert sum(t.s.inf(USA, n) for n in ("Angola", "Zaire", "Vietnam", "Thailand")) == 4


# --------------------------------------------------------------------------- #
# #64 Panama Canal Returned
# --------------------------------------------------------------------------- #


class TestC064PanamaCanalReturned:
    def test_one_each_in_three_countries(self):
        """PD #64: "Add 1 US Influence in Panama, Costa Rica, and Venezuela." Starred."""
        t = Table(turn=4)
        t.inf(usa={"Panama": 1})
        t.play(USA, "Panama Canal Returned", "event")
        assert t.done
        assert (t.s.inf(USA, "Panama"), t.s.inf(USA, "Costa Rica"), t.s.inf(USA, "Venezuela")) == (2, 1, 1)
        assert t.gone_for_good("Panama Canal Returned")


# --------------------------------------------------------------------------- #
# #65 Camp David Accords
# --------------------------------------------------------------------------- #


class TestC065CampDavidAccords:
    def test_vp_influence_and_permanent(self):
        """PD #65: "US gains 1 VP. US receives 1 Influence in Israel, Jordan and Egypt.
        Arab-Israeli War event no longer playable." Starred, lasting effect."""
        t = Table(turn=4)
        t.play(USA, "Camp David Accords", "event")
        assert t.done and t.s.vp == -1
        for n in ("Israel", "Jordan", "Egypt"):
            assert t.s.inf(USA, n) == 1
        assert t.gone_for_good("Camp David Accords")
        t.hand(USSR, "Arab-Israeli War")
        t.action_round(USSR).answer("card:Arab-Israeli War")
        assert "use:event" not in t.keys()

    def test_ussr_ops_play_triggers_it(self):
        """R 5.2: the USSR playing it for ops gives the US the event."""
        t = Table(turn=4)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Camp David Accords", "influence", "yes")
        t.finish_with()
        assert t.s.vp == -1 and t.s.inf(USA, "Israel") == 1
        assert t.gone_for_good("Camp David Accords")


# --------------------------------------------------------------------------- #
# #66 Puppet Governments
# --------------------------------------------------------------------------- #


class TestC066PuppetGovernments:
    def test_three_empty_countries_one_each(self):
        """PD #66: "US may add 1 Influence in three countries that currently contain no
        Influence from either power." Starred."""
        t = Table(turn=4)
        t.inf(ussr={"Angola": 1}, usa={"Zaire": 1})
        t.play(USA, "Puppet Governments", "event")
        assert "country:Angola" not in t.keys() and "country:Zaire" not in t.keys()
        t.answer("country:Kenya")
        assert "country:Kenya" not in t.keys()
        t.answer("country:Somalia", "country:Chile")
        assert t.done
        for n in ("Kenya", "Somalia", "Chile"):
            assert t.s.inf(USA, n) == 1
        assert t.gone_for_good("Puppet Governments")


# --------------------------------------------------------------------------- #
# #67 Grain Sales to Soviets
# --------------------------------------------------------------------------- #


class TestC067GrainSales:
    def test_play_the_stolen_card(self):
        """PD #67: "Randomly choose one card from USSR hand. Play it or return it." The
        US plays the stolen Duck and Cover for its (US) event. Not starred."""
        t = Table(turn=4, defcon=5)
        t.hand(USSR, "Duck and Cover")
        t.play(USA, "Grain Sales to Soviets", "event")
        assert t.d is not None and t.d.player is USA
        t.answer("yes")
        _use_if_asked(t, "event")
        assert t.done
        assert t.s.defcon == 4 and t.s.vp == -1
        assert t.s.hands[USSR] == []
        assert t.in_cycle("Duck and Cover") and t.in_cycle("Grain Sales to Soviets")

    def test_return_then_normal_operations_count_military_ops(self):
        """PD #67: "If Soviet player has no cards, or returned, use this card to conduct
        Operations normally." V:B8: normal use of this 2-ops card, so a coup counts 2
        military operations (R 6.3.3)."""
        t = Table(turn=4)
        t.hand(USSR, "Fidel")
        t.inf(ussr={"Angola": 1})
        t.dice(6)
        t.play(USA, "Grain Sales to Soviets", "event", "no")
        assert t.s.hands[USSR] == ["Fidel"]
        _use_if_asked(t, "coup")
        if t.d is not None and t.d.type is DecisionType.COUP_TARGET:
            t.answer("country:Angola")
        assert t.done
        assert t.s.military_ops[USA] == 2
        assert t.s.inf(USA, "Angola") >= 1

    def test_empty_ussr_hand_gives_operations(self):
        """PD #67: "If Soviet player has no cards ... use this card to conduct Operations
        normally." FAQ #67: "Yes, the US player would still get to conduct operations"."""
        t = Table(turn=4)
        t.play(USA, "Grain Sales to Soviets", "event")
        _use_if_asked(t, "influence")
        t.answer("country:Mexico", "country:Mexico")
        assert t.done and t.s.inf(USA, "Mexico") == 2

    def test_stolen_card_may_go_to_the_space_race(self):
        """FAQ #67: "If Grain Sales is played ... can the card received be played on the
        Space Race? A. Yes, if the Operations Value qualifies." Its event does not occur
        (R 6.4.5)."""
        t = Table(turn=4)
        t.hand(USSR, "Arms Race")
        t.dice(1)
        t.play(USA, "Grain Sales to Soviets", "event", "yes")
        assert t.d is not None and "use:space" in t.keys()
        t.answer("use:space")
        assert t.done
        assert t.s.space_race[USA] == 1 and t.s.vp == -2
        assert t.in_cycle("Arms Race")

    def test_ussr_ops_play_how_i_learned_to_defcon_1_loses_for_the_ussr(self):
        """FAQ #67: "The USSR plays Grain Sales to Soviets and US player chooses How I
        Learned to Stop Worrying. Can the US player set the DEFCON at 1? The USSR would be
        the phasing player, right? ... A Correct." (V:B9.)"""
        t = Table(turn=4, defcon=3)
        t.inf(ussr={"Poland": 1})
        t.hand(USSR, "How I Learned to Stop Worrying")
        t.play(USSR, "Grain Sales to Soviets", "influence", "yes")
        assert t.d is not None and t.d.player is USA
        t.answer("yes")
        _use_if_asked(t, "event")
        t.answer("option:0")
        assert t.s.defcon == 1
        assert t.s.is_over and t.s.winner is USA and t.s.win_reason is WinReason.DEFCON

    def test_ussr_ops_modifiers_do_not_transfer(self):
        """R 7.4.1 EXAMPLE: "If the US player steals one of his cards, for example by
        playing 'Grain Sales to the Soviets', the US player does not benefit from a +1
        Operations value modifier on that card." Brezhnev in play for the USSR; the
        stolen Arms Race is worth 3 to the US."""
        t = Table(turn=4)
        t.effect("Brezhnev Doctrine", USSR, expires="end_of_turn")
        t.hand(USSR, "Arms Race")
        t.play(USA, "Grain Sales to Soviets", "event", "yes")
        assert t.d is not None and t.d.type is DecisionType.CARD_USE
        assert t.d.context.get("ops") == 3


# --------------------------------------------------------------------------- #
# #68 John Paul II Elected Pope
# --------------------------------------------------------------------------- #


class TestC068JohnPaulII:
    def test_poland_swing_and_removed(self):
        """PD #68: "Remove 2 USSR Influence in Poland and then add 1 US Influence in
        Poland." Starred."""
        t = Table(turn=4)
        t.inf(ussr={"Poland": 3})
        t.play(USA, "John Paul II Elected Pope", "event")
        assert t.done
        assert t.s.inf(USSR, "Poland") == 1 and t.s.inf(USA, "Poland") == 1
        assert t.gone_for_good("John Paul II Elected Pope")

    def test_allows_solidarity(self):
        """PD #68: "Allows play of 'Solidarity'." PD #101: "Playable as an event only if
        John Paul II Elected Pope is in effect"."""
        t = Table(turn=8)
        t.hand(USA, "Solidarity")
        t.action_round(USA).answer("card:Solidarity")
        assert "use:event" not in t.keys(), "before John Paul II"
        t = Table(turn=8)
        t.inf(ussr={"Poland": 3})
        t.play(USA, "John Paul II Elected Pope", "event")
        t.hand(USA, "Solidarity")
        t.action_round(USA).answer("card:Solidarity")
        assert "use:event" in t.keys()

    def test_ussr_ops_play_triggers_it(self):
        """R 5.2: played by the USSR for ops, the US event still happens."""
        t = Table(turn=4)
        t.inf(ussr={"Poland": 2})
        t.play(USSR, "John Paul II Elected Pope", "influence", "yes")
        t.finish_with()
        assert t.s.inf(USA, "Poland") == 1
        assert t.gone_for_good("John Paul II Elected Pope")


# --------------------------------------------------------------------------- #
# #69 Latin American Death Squads
# --------------------------------------------------------------------------- #


class TestC069LatinAmericanDeathSquads:
    def test_player_plus_one_in_latin_america(self):
        """PD #69: "All of the player's Coup attempts in Central and South America are +1
        for the remainder of the turn". Panama (stab 2): roll 2 + 2 ops + 1 = 5 > 4."""
        t = Table(turn=4)
        t.effect("Latin American Death Squads", USSR, expires="end_of_turn")
        t.inf(usa={"Panama": 1})
        t.dice(2)
        t.play(USSR, "Decolonization", "coup")
        assert t.done
        assert t.s.inf(USA, "Panama") == 0

    def test_opponent_minus_one_in_latin_america(self):
        """PD #69: "while all opponent's Coup attempts are -1". US: 3 + 2 - 1 = 4, not
        > 4, fails."""
        t = Table(turn=4)
        t.effect("Latin American Death Squads", USSR, expires="end_of_turn")
        t.inf(ussr={"Panama": 1})
        t.dice(3)
        t.play(USA, "Colonial Rear Guards", "coup")
        assert t.done
        assert t.s.inf(USSR, "Panama") == 1

    def test_no_effect_outside_central_and_south_america(self):
        """FAQ #69: "Only coups in Central and South America are affected." US coup in
        Iran (stab 2): 3 + 2 = 5 > 4 succeeds."""
        t = Table(turn=4)
        t.effect("Latin American Death Squads", USSR, expires="end_of_turn")
        t.inf(ussr={"Iran": 1})
        t.dice(3)
        t.play(USA, "Colonial Rear Guards", "coup")
        assert t.done
        assert t.s.inf(USSR, "Iran") == 0

    def test_lasts_the_turn_and_is_not_starred(self):
        """PD #69: "for the remainder of the turn"; not asterisked (discarded)."""
        t = Table(turn=4)
        t.play(USA, "Latin American Death Squads", "event")
        assert t.s.in_play("Latin American Death Squads")
        assert rules.coup_modifier(t.s, USA, "Panama") == 1
        assert rules.coup_modifier(t.s, USSR, "Brazil") == -1
        t.turn_end()
        assert not t.s.in_play("Latin American Death Squads")
        assert t.in_cycle("Latin American Death Squads")


# --------------------------------------------------------------------------- #
# #70 OAS Founded
# --------------------------------------------------------------------------- #


class TestC070OASFounded:
    def test_two_influence_in_the_americas(self):
        """PD #70: "Add 2 US Influence in Central America and/or South America." No
        per-country limit. Starred."""
        t = Table(turn=4)
        t.play(USA, "OAS Founded", "event", "country:Panama")
        assert "country:Panama" in t.keys(), "both points may go to one country"
        assert "country:Poland" not in t.keys()
        t.answer("country:Panama")
        assert t.done and t.s.inf(USA, "Panama") == 2
        assert t.gone_for_good("OAS Founded")


# --------------------------------------------------------------------------- #
# #71 Nixon Plays The China Card
# --------------------------------------------------------------------------- #


class TestC071Nixon:
    def test_takes_the_china_card_face_down(self):
        """PD #71: "Otherwise, US player receives 'The China Card' now, face down and
        unavailable for immediate play." Starred."""
        t = Table(turn=4, china=USSR)
        t.play(USA, "Nixon Plays The China Card", "event")
        assert t.done
        assert t.s.china_card_owner is USA and not t.s.china_card_face_up
        assert CHINA_CARD not in t.s.playable_hand(USA)
        assert t.gone_for_good("Nixon Plays The China Card")

    def test_two_vp_if_already_held(self):
        """PD #71: "If US has 'The China Card', gain 2 VP.\""""
        t = Table(turn=4, china=USA)
        t.play(USA, "Nixon Plays The China Card", "event")
        assert t.s.vp == -2 and t.s.china_card_owner is USA

    def test_china_card_flips_up_at_turn_end(self):
        """R 9.3: "At the end of the turn, it is flipped face up"."""
        t = Table(turn=4, china=USSR)
        t.play(USA, "Nixon Plays The China Card", "event")
        t.turn_end()
        assert t.s.china_card_face_up


# --------------------------------------------------------------------------- #
# #72 Sadat Expels Soviets
# --------------------------------------------------------------------------- #


class TestC072Sadat:
    def test_clears_egypt(self):
        """PD #72: "Remove all USSR Influence in Egypt and add one US Influence."
        Starred."""
        t = Table(turn=4)
        t.inf(ussr={"Egypt": 3})
        t.play(USA, "Sadat Expels Soviets", "event")
        assert t.s.inf(USSR, "Egypt") == 0 and t.s.inf(USA, "Egypt") == 1
        assert t.gone_for_good("Sadat Expels Soviets")


# --------------------------------------------------------------------------- #
# #73 Shuttle Diplomacy
# --------------------------------------------------------------------------- #


class TestC073ShuttleDiplomacy:
    def test_stays_in_front_of_the_us_player(self):
        """PD #73: "Play in front of US player.\""""
        t = Table(turn=4)
        t.play(USA, "Shuttle Diplomacy", "event")
        assert t.done
        assert t.where("Shuttle Diplomacy") == "in_play"

    def test_removed_battleground_is_gone_entirely(self):
        """FAQ #73: "I control North Korea and that's all in Asia ... I do still have
        presence in the region right? A. No, the card temporarily 'removes' the country
        entirely from the equation." (V:B2.) USSR scores 0 for Asia."""
        t = Table(turn=4)
        t.effect("Shuttle Diplomacy", USA)
        t.inf(ussr={"North Korea": 3})
        t.play(USSR, "Asia Scoring")
        assert t.done
        assert t.s.vp == 0

    def test_country_total_drops_too(self):
        """V:B2 point 2: the battleground is struck from the USSR's country count, so it
        cannot be re-used as a non-battleground. USSR controls Iraq, Iran, Saudi Arabia;
        US controls Israel and Jordan. With one USSR battleground removed: USSR 2
        countries (2 BG) vs US 2 countries -> both Presence: USSR 3+2=5, US 3+1=4, net
        USSR +1 (not Domination's 7)."""
        t = Table(turn=4)
        t.effect("Shuttle Diplomacy", USA)
        t.inf(ussr={"Iraq": 3, "Iran": 2, "Saudi Arabia": 3}, usa={"Israel": 4, "Jordan": 2})
        t.play(USSR, "Middle East Scoring")
        assert t.done
        assert t.s.vp == 1

    def test_us_chooses_and_adjacency_bonus_goes_with_it(self):
        """FAQ #73: "Who chooses the country that doesn't count? This is an issue only if
        Japan is Soviet controlled, as it would also affect the extra point for being
        adjacent to the US. A. ... the US player chooses." (V:B2.) USSR controls Japan
        and North Korea; the US strikes Japan: USSR keeps North Korea only -> Presence
        3 + 1 BG = 4."""
        t = Table(turn=4)
        t.effect("Shuttle Diplomacy", USA)
        t.inf(ussr={"Japan": 4, "North Korea": 3})
        t.play(USSR, "Asia Scoring")
        t.finish_with(lambda d: "country:Japan" if "country:Japan" in d.legal_keys else d.legal_keys[0])
        assert t.s.vp == 4

    def test_discarded_after_the_scoring(self):
        """PD #73: "then put this card in the discard pile" (not removed)."""
        t = Table(turn=4)
        t.effect("Shuttle Diplomacy", USA)
        t.inf(ussr={"Iraq": 3})
        t.play(USSR, "Middle East Scoring")
        assert not t.s.in_play("Shuttle Diplomacy")
        assert t.in_cycle("Shuttle Diplomacy")

    def test_spent_even_if_the_ussr_has_no_battleground(self):
        """PD #73: "During the next scoring of the Middle East or Asia" -- the card is used
        up by that scoring whatever the USSR holds (V:B2 point 4)."""
        t = Table(turn=4)
        t.effect("Shuttle Diplomacy", USA)
        t.inf(usa={"Israel": 4})
        t.play(USA, "Middle East Scoring")
        assert not t.s.in_play("Shuttle Diplomacy")

    def test_southeast_asia_scoring_does_not_use_it(self):
        """FAQ #73: "Does this card affect the Southeast Asia scoring card? A. Not at all.\""""
        t = Table(turn=4)
        t.effect("Shuttle Diplomacy", USA)
        t.inf(ussr={"Thailand": 2})
        t.play(USSR, "Southeast Asia Scoring")
        assert t.done
        assert t.s.vp == 2
        assert t.s.in_play("Shuttle Diplomacy")

    def test_does_not_count_in_final_scoring(self):
        """PD #73: "Does not count for Final Scoring at the end of Turn 10." (V:B2.) USSR
        controls North Korea only: Presence 3 + 1 BG = 4."""
        t = Table(turn=10)
        t.effect("Shuttle Diplomacy", USA)
        t.inf(ussr={"North Korea": 3})
        t.final_scoring()
        assert t.s.vp == 4


# --------------------------------------------------------------------------- #
# #74 The Voice Of America
# --------------------------------------------------------------------------- #


class TestC074VoiceOfAmerica:
    def test_four_outside_europe_max_two_each(self):
        """PD #74: "Remove 4 USSR Influence from non-European countries. No more than 2
        may be removed from any one country." Not starred."""
        t = Table(turn=4)
        t.inf(ussr={"Angola": 3, "Cuba": 2, "Iraq": 2, "Poland": 2})
        t.play(USA, "The Voice Of America", "event")
        assert "country:Poland" not in t.keys()
        t.answer("country:Angola", "country:Angola")
        assert "country:Angola" not in t.keys()
        t.answer("country:Cuba", "country:Iraq")
        assert t.done
        assert (t.s.inf(USSR, "Angola"), t.s.inf(USSR, "Cuba"), t.s.inf(USSR, "Iraq")) == (1, 1, 1)
        assert t.s.inf(USSR, "Poland") == 2
        assert t.in_cycle("The Voice Of America")


# --------------------------------------------------------------------------- #
# #75 Liberation Theology
# --------------------------------------------------------------------------- #


class TestC075LiberationTheology:
    def test_three_in_central_america_max_two_each(self):
        """PD #75: "Add 3 USSR Influence in Central America, no more than 2 per country."
        Not starred."""
        t = Table(turn=4)
        t.play(USSR, "Liberation Theology", "event", "country:Cuba")
        assert "country:Brazil" not in t.keys(), "South America is not Central America"
        t.answer("country:Cuba")
        assert "country:Cuba" not in t.keys()
        t.answer("country:Nicaragua")
        assert t.done
        assert t.s.inf(USSR, "Cuba") == 2 and t.s.inf(USSR, "Nicaragua") == 1
        assert t.in_cycle("Liberation Theology")


# --------------------------------------------------------------------------- #
# #76 Ussuri River Skirmish
# --------------------------------------------------------------------------- #


class TestC076UssuriRiver:
    def test_claims_the_china_card_face_up(self):
        """PD #76: "If the USSR has 'The China Card', claim it face up and available for
        play." R 9.4: passed by an event, it may be played the same turn. Starred."""
        t = Table(turn=4, china=USSR)
        t.play(USA, "Ussuri River Skirmish", "event")
        assert t.done
        assert t.s.china_card_owner is USA and t.s.china_card_face_up
        assert CHINA_CARD in t.s.playable_hand(USA)
        assert t.gone_for_good("Ussuri River Skirmish")

    def test_four_in_asia_max_two_each_if_already_held(self):
        """PD #76: "If the US already has 'The China Card', add 4 US Influence in Asia,
        no more than 2 per country." Southeast Asia is part of Asia."""
        t = Table(turn=4, china=USA)
        t.play(USA, "Ussuri River Skirmish", "event", "country:Japan", "country:Japan")
        assert "country:Japan" not in t.keys()
        assert "country:Iran" not in t.keys()
        t.answer("country:Thailand", "country:Thailand")
        assert t.done
        assert t.s.inf(USA, "Japan") == 2 and t.s.inf(USA, "Thailand") == 2


# --------------------------------------------------------------------------- #
# #77 Ask Not What Your Country Can Do For You...
# --------------------------------------------------------------------------- #


class TestC077AskNot:
    def test_discard_any_then_draw_as_many(self):
        """PD #77: "US player may discard up to entire hand (including Scoring cards) and
        draw replacements from the deck. The number of cards discarded must be decided
        prior to drawing any replacements." Discarded events do not occur (R 5.4).
        Starred."""
        t = Table(turn=4)
        t.inf(usa={"Cuba": 2})
        t.hand(USA, "Duck and Cover", "Fidel", "Asia Scoring")
        t.deck("Arms Race", "Summit", "Junta")
        t.play(USA, "Ask Not What Your Country Can Do For You...", "event")
        t.answer("card:Fidel", "card:Asia Scoring", "pass")
        assert t.done
        assert sorted(t.s.hands[USA]) == ["Arms Race", "Duck and Cover", "Summit"]
        assert t.in_cycle("Fidel") and t.in_cycle("Asia Scoring")
        assert t.s.inf(USA, "Cuba") == 2 and t.s.vp == 0
        assert t.s.deck == ["Junta"]
        assert t.gone_for_good("Ask Not What Your Country Can Do For You...")

    def test_may_discard_nothing(self):
        """PD #77: "up to" -- zero is allowed; the hand is unchanged."""
        t = Table(turn=4)
        t.hand(USA, "Duck and Cover")
        t.deck("Arms Race")
        t.play(USA, "Ask Not What Your Country Can Do For You...", "event", "pass")
        assert t.done
        assert t.s.hands[USA] == ["Duck and Cover"]


# --------------------------------------------------------------------------- #
# #78 Alliance for Progress
# --------------------------------------------------------------------------- #


class TestC078AllianceForProgress:
    def test_one_vp_per_us_battleground_in_the_americas(self):
        """PD #78: "US gains 1 VP for each US controlled Battleground country in Central
        America and South America." Panama, Chile (BG) count; Colombia (not BG) and
        Iran (not the Americas) do not. Starred."""
        t = Table(turn=4)
        t.inf(usa={"Panama": 2, "Chile": 3, "Colombia": 1, "Iran": 2})
        t.play(USA, "Alliance for Progress", "event")
        assert t.done and t.s.vp == -2
        assert t.gone_for_good("Alliance for Progress")

    def test_no_effect_event_is_still_played_and_removed(self):
        """R 5.2 EXAMPLE 3: "The USSR Player plays 'Alliance for Progress;' however, the
        US Player does not control a Battleground country in either South or Central
        America. Nevertheless, the Event is considered played, and the card would be
        removed from the game after the USSR player's round.\""""
        t = Table(turn=4)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Alliance for Progress", "influence")
        t.finish_with()
        assert t.s.vp == 0
        assert t.gone_for_good("Alliance for Progress")


# --------------------------------------------------------------------------- #
# #80 One Small Step...
# --------------------------------------------------------------------------- #


class TestC080OneSmallStep:
    def test_requires_being_behind(self):
        """PD #80: "If you are behind on the Space Race Track, play this card ...".
        Level with the opponent, the event is not available."""
        t = Table(turn=4)
        t.s.space_race = [1, 1]
        t.hand(USSR, "One Small Step...")
        t.action_round(USSR).answer("card:One Small Step...")
        assert "use:event" not in t.keys()

    def test_two_boxes_vp_of_the_second_only(self):
        """PD #80: "move your marker two boxes forward ..., gaining the VP value of the
        second box only." USSR at 3, US at 4: USSR jumps to Lunar Orbit (5), first there
        -> 3 VP; nothing for box 4. Not starred."""
        t = Table(turn=4)
        t.s.space_race = [3, 4]
        t.play(USSR, "One Small Step...", "event")
        assert t.done
        assert t.s.space_race[USSR] == 5 and t.s.vp == 3
        assert t.in_cycle("One Small Step...")

    def test_skipped_box_pays_nothing_and_landing_ability_is_gained(self):
        """PD #80 "VP value of the second box only"; FAQ #18/#80: "if the second box is a
        benefit box and you are the first player to that box you gain the benefit." USSR
        at 0, US at 1: USSR lands on Animal in Space (box 2), no VP from Earth
        Satellite, and gains the two-attempts ability (R 6.4.4)."""
        t = Table(turn=4)
        t.s.space_race = [0, 1]
        t.play(USSR, "One Small Step...", "event")
        assert t.s.space_race[USSR] == 2 and t.s.vp == 0
        assert spacerace.attempts_allowed(t.s.space_race, USSR) == 2

    def test_second_arrival_gets_second_place_vp(self):
        """PD #80 + R 6.4.3 (right-hand number to the second player). USSR at 3, US at 5:
        USSR lands on Lunar Orbit second -> 1 VP."""
        t = Table(turn=4)
        t.s.space_race = [3, 5]
        t.play(USSR, "One Small Step...", "event")
        assert t.s.space_race[USSR] == 5 and t.s.vp == 1

    def test_does_not_use_the_turns_space_attempt(self):
        """FAQ #18 (by analogy, "Events affecting the Space Race do not count as a space
        race attempt"): the player may still make a normal attempt this turn."""
        t = Table(turn=4)
        t.s.space_race = [0, 1]
        t.play(USSR, "One Small Step...", "event")
        assert t.s.space_attempts[USSR] == 0

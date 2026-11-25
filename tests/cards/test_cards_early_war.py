"""Early War cards #4-#35 (scoring cards and the China Card have their own files).

Card text quoted from the official digital edition text (``PD``, the Deluxe card faces)
as collected in ``scratchpad/rules/cards.md``; rule numbers are the Deluxe rulebook
(``R``); ``FAQ`` is the designer-authorised FAQ v5 (2010); ``V:Bn`` points at an entry
in ``verdicts.md``. Expectations follow those sources, not the current implementation.
"""

from __future__ import annotations

import pytest

from _table import USA, USSR, Table
from twilight import rules
from twilight.decisions import DecisionType
from twilight.enums import Region, WinReason


# --------------------------------------------------------------------------- #
# #4 Duck and Cover
# --------------------------------------------------------------------------- #


class TestC004DuckAndCover:
    def test_degrades_then_scores_five_minus_defcon(self):
        """PD #4: "Degrade DEFCON one level. Then US player earns VPs equal to 5 minus
        current DEFCON level." DEFCON 4 -> 3, US +2."""
        t = Table(turn=2, defcon=4)
        t.play(USA, "Duck and Cover", "event")
        assert t.done
        assert t.s.defcon == 3
        assert t.s.vp == -2
        assert t.in_cycle("Duck and Cover"), "not starred: goes to the discard pile"

    def test_at_defcon_2_the_phasing_us_player_loses(self):
        """R 8.1.3: "The phasing player is responsible for the status marker moving to
        DEFCON 1, and loses the game." US plays it at DEFCON 2 -> US loses, no VP."""
        t = Table(turn=2, defcon=2)
        t.play(USA, "Duck and Cover", "event")
        assert t.s.is_over and t.s.winner is USSR and t.s.win_reason is WinReason.DEFCON
        assert t.done, "no decision may follow the end of the game"

    def test_ussr_playing_it_for_ops_at_defcon_2_loses(self):
        """R 5.2 + 8.1.3: the USSR plays the US card for operations, the US event
        occurs, DEFCON reaches 1 during the USSR action round -> the USSR (phasing)
        loses."""
        t = Table(turn=2, defcon=2)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Duck and Cover", "influence", "yes")
        assert t.s.is_over and t.s.winner is USA and t.s.win_reason is WinReason.DEFCON
        # That no operations decision follows is checked in test_mech_endgame (B11).


# --------------------------------------------------------------------------- #
# #5 Five-Year Plan
# --------------------------------------------------------------------------- #


class TestC005FiveYearPlan:
    def test_random_us_event_card_fires(self):
        """PD #5: "USSR player must randomly discard one card. If the card is a US
        associated Event, the Event occurs immediately." The only USSR card is Duck and
        Cover: DEFCON 5 -> 4 and the US gains 1 VP."""
        t = Table(turn=2, defcon=5)
        t.hand(USSR, "Duck and Cover")
        t.play(USA, "Five-Year Plan", "event")
        assert t.done
        assert t.s.defcon == 4 and t.s.vp == -1
        assert "Duck and Cover" not in t.s.hands[USSR]
        assert t.in_cycle("Duck and Cover")
        assert t.in_cycle("Five-Year Plan")

    def test_random_starred_us_event_fires_and_is_removed(self):
        """PD #5 + R 4.3: the discarded US event occurs; an asterisked event that
        occurs is removed from the game. Truman Doctrine strips Poland (uncontrolled)."""
        t = Table(turn=2)
        t.inf(ussr={"Italy": 1})
        t.hand(USSR, "Truman Doctrine")
        t.play(USA, "Five-Year Plan", "event")
        t.finish_with()
        assert t.s.inf(USSR, "Italy") == 0
        assert t.gone_for_good("Truman Doctrine")

    def test_ussr_or_neutral_event_is_discarded_without_effect(self):
        """PD #5: "If the card is a USSR associated Event or an Event applicable to both
        players, then the card must be discarded without triggering the Event."
        (R 5.4: forced discards never trigger.)"""
        for card in ("Fidel", "Nuclear Test Ban"):
            t = Table(turn=2, defcon=3)
            t.inf(usa={"Cuba": 2})
            t.hand(USSR, card)
            t.play(USA, "Five-Year Plan", "event")
            assert t.done, card
            assert t.s.inf(USA, "Cuba") == 2 and t.s.defcon == 3 and t.s.vp == 0, card
            assert t.in_cycle(card), f"{card} must be discarded, not removed"

    def test_the_china_card_is_never_the_random_discard(self):
        """R 9.5: the China Card "may not be played ... as a discard required by an
        Event". With an empty hand apart from the China Card nothing is discarded."""
        t = Table(turn=2, china=USSR)
        t.play(USA, "Five-Year Plan", "event")
        assert t.done
        assert t.s.china_card_owner is USSR and t.s.china_card_face_up

    def test_triggered_us_event_at_defcon_2_loses_for_the_phasing_ussr(self):
        """V:B9 table: Five-Year Plan played by the USSR for ops, the US event fires and
        the randomly discarded Duck and Cover drops DEFCON to 1 -> the phasing player
        (USSR) loses (R 8.1.3)."""
        t = Table(turn=2, defcon=2)
        t.inf(ussr={"Poland": 1})
        t.hand(USSR, "Duck and Cover")
        t.play(USSR, "Five-Year Plan", "influence", "yes")
        assert t.s.is_over and t.s.winner is USA and t.s.win_reason is WinReason.DEFCON


# --------------------------------------------------------------------------- #
# #7 Socialist Governments
# --------------------------------------------------------------------------- #


class TestC007SocialistGovernments:
    def test_removes_three_with_at_most_two_per_country(self):
        """PD #7: "Remove US Influence in Western Europe by a total of 3 Influence
        points, removing no more than 2 per country.\""""
        t = Table(turn=2)
        t.inf(usa={"Italy": 3, "France": 1, "Benelux": 1, "Poland": 2})
        t.play(USSR, "Socialist Governments", "event", "country:Italy", "country:Italy")
        assert "country:Italy" not in t.keys(), "at most 2 from one country"
        assert "country:Poland" not in t.keys(), "Poland is Eastern Europe"
        t.answer("country:France")
        assert t.done
        assert t.s.inf(USA, "Italy") == 1 and t.s.inf(USA, "France") == 0
        assert t.in_cycle("Socialist Governments")

    def test_iron_lady_prevents_the_event(self):
        """PD #7: "Unplayable as an event if 'The Iron Lady' is in effect." R 5.2: the
        US playing it for ops then triggers nothing and it is discarded."""
        t = Table(turn=8)
        t.effect("The Iron Lady", USA)
        t.inf(usa={"Italy": 3}, ussr={"Poland": 1})
        t.hand(USSR, "Socialist Governments")
        t.action_round(USSR).answer("card:Socialist Governments")
        if t.d is not None and t.d.type is DecisionType.CARD_USE:
            assert "use:event" not in t.keys()


# --------------------------------------------------------------------------- #
# #8 Fidel
# --------------------------------------------------------------------------- #


class TestC008Fidel:
    def test_clears_us_and_gives_ussr_control(self):
        """PD #8: "Remove all US Influence in Cuba. USSR gains sufficient Influence in
        Cuba for Control." Cuba stability 3."""
        t = Table(turn=2)
        t.inf(usa={"Cuba": 2})
        t.play(USSR, "Fidel", "event")
        assert t.done
        assert t.s.inf(USA, "Cuba") == 0 and t.s.inf(USSR, "Cuba") == 3
        assert rules.controls(t.s, USSR, "Cuba")
        assert t.gone_for_good("Fidel")

    def test_us_playing_it_for_ops_triggers_and_removes_it(self):
        """R 5.2: "If a player plays a card as an Operation, and the card's Event is
        associated only with his opponent, the Event still occurs (and the card, if it
        has an asterisk after the Event title, is removed)."""
        t = Table(turn=2)
        t.inf(usa={"Cuba": 1})
        t.play(USA, "Fidel", "influence", "no", "country:Mexico", "country:Mexico")
        t.finish_with()
        assert t.s.inf(USA, "Cuba") == 0 and t.s.inf(USSR, "Cuba") == 3
        assert t.s.inf(USA, "Mexico") == 2
        assert t.gone_for_good("Fidel")


# --------------------------------------------------------------------------- #
# #9 Vietnam Revolts
# --------------------------------------------------------------------------- #


class TestC009VietnamRevolts:
    def test_adds_two_in_vietnam(self):
        """PD #9: "Add 2 USSR Influence in Vietnam.\""""
        t = Table(turn=2)
        t.play(USSR, "Vietnam Revolts", "event")
        assert t.done
        assert t.s.inf(USSR, "Vietnam") == 2
        assert t.where("Vietnam Revolts") in ("removed", "in_play")

    def test_plus_one_op_when_all_points_go_to_southeast_asia(self):
        """PD #9: "For the remainder of the turn, the Soviet player may add 1 Operations
        point to any card that uses all points in Southeast Asia." (V: B6.) A 2-op card
        placed entirely in SE Asia places 3 influence."""
        t = Table(turn=2)
        t.effect("Vietnam Revolts", USSR, expires="end_of_turn")
        t.inf(ussr={"Vietnam": 2})
        t.hand(USSR, "Decolonization")
        t.play(USSR, "Decolonization", "influence")
        t.answer("country:Laos/Cambodia", "country:Laos/Cambodia")
        assert t.d is not None, "a third (bonus) point should be offered in SE Asia"
        t.answer("country:Thailand")
        assert t.done
        assert t.s.inf(USSR, "Laos/Cambodia") + t.s.inf(USSR, "Thailand") == 3

    def test_bonus_is_ussr_only(self):
        """PD #9: the bonus belongs to "the Soviet player" (R 7.4.1). The US placing 2
        ops in SE Asia gets exactly 2."""
        t = Table(turn=2)
        t.effect("Vietnam Revolts", USSR, expires="end_of_turn")
        t.inf(usa={"Thailand": 1})
        t.play(USA, "Colonial Rear Guards", "influence", "country:Thailand", "country:Thailand")
        assert t.done
        assert t.s.inf(USA, "Thailand") == 3

    def test_effect_ends_with_the_turn(self):
        """PD #9: "For the remainder of the turn" -- gone after the turn end."""
        t = Table(turn=2)
        t.effect("Vietnam Revolts", USSR, expires="end_of_turn")
        t.turn_end()
        assert not t.s.in_play("Vietnam Revolts")
        assert t.gone_for_good("Vietnam Revolts")


# --------------------------------------------------------------------------- #
# #10 Blockade
# --------------------------------------------------------------------------- #


class TestC010Blockade:
    def test_us_may_discard_a_three_op_card(self):
        """PD #10: "Unless US Player immediately discards a '3' or more value Operations
        card, eliminate all US Influence in West Germany." R 5.4: the discarded card's
        event does not occur."""
        t = Table(turn=2, defcon=4)
        t.inf(usa={"West Germany": 3})
        t.hand(USA, "Duck and Cover", "CIA Created")
        t.play(USSR, "Blockade", "event")
        assert set(t.keys()) == {"card:Duck and Cover", "pass"}
        t.answer("card:Duck and Cover")
        assert t.done
        assert t.s.inf(USA, "West Germany") == 3
        assert t.in_cycle("Duck and Cover") and t.s.defcon == 4
        assert t.gone_for_good("Blockade")

    def test_without_discard_west_germany_is_cleared(self):
        """PD #10 as above; declining (or lacking a 3+ card) eliminates US influence."""
        t = Table(turn=2)
        t.inf(usa={"West Germany": 4})
        t.hand(USA, "CIA Created")
        t.play(USSR, "Blockade", "event")
        assert t.done, "with no 3+ ops card there is nothing to ask"
        assert t.s.inf(USA, "West Germany") == 0

    def test_the_china_card_cannot_be_discarded(self):
        """R 9.5: the China Card may not be played "as a discard required by an Event"."""
        t = Table(turn=2, china=USA)
        t.inf(usa={"West Germany": 4})
        t.play(USSR, "Blockade", "event")
        assert t.done
        assert t.s.inf(USA, "West Germany") == 0
        assert t.s.china_card_owner is USA

    def test_modified_ops_count(self):
        """R 7.4.2: ops modifiers apply "for all purposes"; R 5.3 any higher card
        satisfies. Under Containment a printed 2-op US card is worth 3."""
        t = Table(turn=2)
        t.effect("Containment", USA, expires="end_of_turn")
        t.inf(usa={"West Germany": 2})
        t.hand(USA, "Independent Reds")
        t.play(USSR, "Blockade", "event")
        assert "card:Independent Reds" in t.keys()


# --------------------------------------------------------------------------- #
# #11 Korean War
# --------------------------------------------------------------------------- #


class TestC011KoreanWar:
    def test_victory_on_four(self):
        """PD #11: "Roll one die and subtract 1 for every US Controlled country adjacent
        to South Korea. USSR Victory on modified die roll 4-6. USSR add 2 to Military Ops
        Track. Effects of Victory: USSR gains 2 VP and replaces all US Influence in South
        Korea with USSR Influence." Second edition: no DEFCON change (FAQ #11)."""
        t = Table(turn=2, defcon=4)
        t.inf(usa={"South Korea": 2}, ussr={"South Korea": 1})
        t.dice(4)
        t.play(USSR, "Korean War", "event")
        assert t.done
        assert t.s.inf(USA, "South Korea") == 0 and t.s.inf(USSR, "South Korea") == 3
        assert t.s.vp == 2 and t.s.military_ops[USSR] == 2 and t.s.defcon == 4
        assert t.gone_for_good("Korean War")

    def test_adjacent_us_control_subtracts(self):
        """PD #11: -1 per US-controlled neighbour (Japan). 4 - 1 = 3: no victory, but
        the 2 military ops are still credited (R 7.6)."""
        t = Table(turn=2)
        t.inf(usa={"South Korea": 2, "Japan": 4})
        t.dice(4)
        t.play(USSR, "Korean War", "event")
        assert t.s.inf(USA, "South Korea") == 2 and t.s.vp == 0
        assert t.s.military_ops[USSR] == 2

    def test_us_playing_it_for_ops_credits_ussr_military_ops(self):
        """R 8.2.4: "If a player uses a card for Operations points, and thereby triggers
        a War Event associated with his opponent, his opponent's Military marker is
        moved on the Military Operations track as directed by the Event text.\""""
        t = Table(turn=2)
        t.inf(usa={"South Korea": 1})
        t.dice(1)
        t.play(USA, "Korean War", "influence", "yes", "country:Mexico", "country:Mexico")
        t.finish_with()
        assert t.s.military_ops[USSR] == 2 and t.s.military_ops[USA] == 0


# --------------------------------------------------------------------------- #
# #12 Romanian Abdication
# --------------------------------------------------------------------------- #


class TestC012RomanianAbdication:
    def test_clears_us_and_gives_control(self):
        """PD #12: "Remove all US Influence in Romania. USSR gains sufficient Influence
        in Romania for Control." (stability 3)"""
        t = Table(turn=2)
        t.inf(usa={"Romania": 1}, ussr={"Romania": 1})
        t.play(USSR, "Romanian Abdication", "event")
        assert t.s.inf(USA, "Romania") == 0 and t.s.inf(USSR, "Romania") == 3
        assert t.gone_for_good("Romanian Abdication")


# --------------------------------------------------------------------------- #
# #13 Arab-Israeli War
# --------------------------------------------------------------------------- #


class TestC013ArabIsraeliWar:
    def test_israel_us_control_also_subtracts(self):
        """PD #13: "Roll one die and subtract 1 for US Control of Israel and for
        US-controlled country adjacent to Israel. USSR Victory on modified die roll 4-6."
        US controls Israel and Jordan: 5 - 2 = 3, no victory; 2 military ops anyway."""
        t = Table(turn=2)
        t.inf(usa={"Israel": 4, "Jordan": 2})
        t.dice(5)
        t.play(USSR, "Arab-Israeli War", "event")
        assert t.s.inf(USA, "Israel") == 4 and t.s.vp == 0
        assert t.s.military_ops[USSR] == 2
        assert t.in_cycle("Arab-Israeli War"), "not starred"

    def test_victory_replaces_influence(self):
        """PD #13: "Effects of Victory: USSR gains 2 VP and replaces all US Influence in
        Israel with USSR Influence.\""""
        t = Table(turn=2)
        t.inf(usa={"Israel": 3})
        t.dice(6)
        t.play(USSR, "Arab-Israeli War", "event")
        assert t.s.inf(USA, "Israel") == 0 and t.s.inf(USSR, "Israel") == 3 and t.s.vp == 2

    def test_camp_david_prevents_the_event(self):
        """PD #65: Camp David Accords: "Arab-Israeli War event no longer playable." R 5.2
        Example 2: the US playing it for ops conducts the ops, no event, and the card is
        not eliminated."""
        t = Table(turn=5)
        t.effect("Camp David Accords", USA)
        t.inf(usa={"Israel": 1})
        t.play(USA, "Arab-Israeli War", "influence", "country:Mexico", "country:Mexico")
        t.finish_with()
        assert t.s.military_ops[USSR] == 0
        assert t.in_cycle("Arab-Israeli War")


# --------------------------------------------------------------------------- #
# #14 COMECON
# --------------------------------------------------------------------------- #


class TestC014Comecon:
    def test_four_different_non_us_controlled_countries(self):
        """PD #14: "Add 1 USSR Influence in each of four non-US Controlled countries in
        Eastern Europe.\""""
        t = Table(turn=2)
        t.inf(usa={"Poland": 4})
        t.play(USSR, "COMECON", "event")
        assert "country:Poland" not in t.keys()
        assert "country:Italy" not in t.keys()
        t.answer("country:Hungary")
        assert "country:Hungary" not in t.keys(), "one per country"
        t.answer("country:Romania", "country:Bulgaria", "country:Austria")
        assert t.done
        for n in ("Hungary", "Romania", "Bulgaria", "Austria"):
            assert t.s.inf(USSR, n) == 1
        assert t.gone_for_good("COMECON")


# --------------------------------------------------------------------------- #
# #15 Nasser
# --------------------------------------------------------------------------- #


class TestC015Nasser:
    def test_half_rounded_up(self):
        """PD #15: "Add 2 USSR Influence in Egypt. Remove half (rounded up) of the US
        Influence in Egypt." US 3 -> loses 2 -> 1."""
        t = Table(turn=2)
        t.inf(usa={"Egypt": 3})
        t.play(USSR, "Nasser", "event")
        assert t.s.inf(USSR, "Egypt") == 2 and t.s.inf(USA, "Egypt") == 1
        assert t.gone_for_good("Nasser")


# --------------------------------------------------------------------------- #
# #16 Warsaw Pact Formed
# --------------------------------------------------------------------------- #


class TestC016WarsawPactFormed:
    def test_remove_option_clears_four_countries(self):
        """PD #16: "Remove all US Influence from four countries in Eastern Europe ...\""""
        t = Table(turn=2)
        t.inf(usa={"Poland": 2, "Hungary": 1, "Romania": 1, "Bulgaria": 3, "Yugoslavia": 1})
        t.play(USSR, "Warsaw Pact Formed", "event", "option:0")
        t.answer("country:Poland", "country:Hungary", "country:Romania", "country:Bulgaria")
        assert t.done
        assert t.s.inf(USA, "Yugoslavia") == 1
        assert sum(t.s.inf(USA, n) for n in ("Poland", "Hungary", "Romania", "Bulgaria")) == 0

    def test_add_option_five_at_most_two_each(self):
        """PD #16: "... or add 5 USSR Influence in Eastern Europe, adding no more than 2
        per country.\""""
        t = Table(turn=2)
        t.play(USSR, "Warsaw Pact Formed", "event", "option:1")
        t.answer("country:Poland", "country:Poland")
        assert "country:Poland" not in t.keys()
        t.answer("country:Hungary", "country:Hungary", "country:Austria")
        assert t.done
        assert t.s.inf(USSR, "Poland") == 2 and t.s.inf(USSR, "Austria") == 1

    def test_allows_nato(self):
        """PD #16: "Allow play of NATO.\""""
        t = Table(turn=2)
        t.play(USSR, "Warsaw Pact Formed", "event", "option:1")
        t.finish_with()
        t.hand(USA, "NATO")
        t.action_round(USA).answer("card:NATO")
        assert "use:event" in t.keys()

    def test_gives_no_coup_protection(self):
        """V:B1: Warsaw Pact has no coup/realignment restriction; only NATO protects.
        After it, the US may still coup a USSR-controlled Eastern European country at
        DEFCON 5."""
        t = Table(turn=2, defcon=5)
        t.effect("Warsaw Pact Formed", USSR)
        t.inf(ussr={"Poland": 4})
        assert rules.can_coup(t.s, USA, "Poland")
        assert rules.can_realign(t.s, USA, "Poland")

    def test_is_removed_after_its_event(self):
        """PD #16 is asterisked (R 4.3): out of the draw cycle after the event."""
        t = Table(turn=2)
        t.play(USSR, "Warsaw Pact Formed", "event", "option:1")
        t.finish_with()
        assert t.gone_for_good("Warsaw Pact Formed")


# --------------------------------------------------------------------------- #
# #17 De Gaulle Leads France
# --------------------------------------------------------------------------- #


class TestC017DeGaulle:
    def test_influence_change(self):
        """PD #17: "Remove 2 US Influence in France, add 1 USSR Influence.\""""
        t = Table(turn=2)
        t.inf(usa={"France": 3})
        t.play(USSR, "De Gaulle Leads France", "event")
        assert t.s.inf(USA, "France") == 1 and t.s.inf(USSR, "France") == 1
        assert t.gone_for_good("De Gaulle Leads France")

    def test_cancels_nato_for_france_only(self):
        """PD #17: "Cancels effects of NATO for France." US-controlled France loses NATO
        protection; US-controlled Italy keeps it (DEFCON 5)."""
        t = Table(turn=3, defcon=5)
        t.effect("NATO", USA)
        t.resolved("Marshall Plan")
        t.inf(usa={"France": 6, "Italy": 3}, ussr={"Italy": 1})
        t.play(USSR, "De Gaulle Leads France", "event")
        t.finish_with()
        assert rules.controls(t.s, USA, "France")
        assert rules.can_coup(t.s, USSR, "France")
        assert not rules.can_coup(t.s, USSR, "Italy")


# --------------------------------------------------------------------------- #
# #18 Captured Nazi Scientist
# --------------------------------------------------------------------------- #


class TestC018CapturedNaziScientist:
    def test_advances_one_box_with_vp(self):
        """PD #18: "Advance player's Space Race marker one box." First to Earth
        Satellite scores 2 (R 6.4.3)."""
        t = Table(turn=2)
        t.play(USA, "Captured Nazi Scientist", "event")
        assert t.s.space_race[USA] == 1 and t.s.vp == -2
        assert t.gone_for_good("Captured Nazi Scientist")

    def test_does_not_use_up_the_space_attempt(self):
        """FAQ #18: "Does this count as a space race attempt? A. No.\""""
        t = Table(turn=2)
        t.play(USSR, "Captured Nazi Scientist", "event")
        assert t.s.space_attempts[USSR] == 0
        t.hand(USSR, "Duck and Cover")
        t.action_round(USSR).answer("card:Duck and Cover")
        assert "use:space" in t.keys()


# --------------------------------------------------------------------------- #
# #19 Truman Doctrine
# --------------------------------------------------------------------------- #


class TestC019TrumanDoctrine:
    def test_only_uncontrolled_european_countries(self):
        """PD #19: "Remove all USSR Influence markers in one uncontrolled country in
        Europe." FAQ #19: "a European country must not be controlled by either power"."""
        t = Table(turn=2)
        t.inf(
            ussr={"Poland": 3, "Italy": 2, "France": 1, "Iraq": 1},
            usa={"Italy": 1, "France": 5},
        )
        t.play(USA, "Truman Doctrine", "event")
        # Poland USSR-controlled, France US-controlled, Iraq not Europe: only Italy.
        assert t.done
        assert t.s.inf(USSR, "Italy") == 0 and t.s.inf(USSR, "Poland") == 3
        assert t.s.inf(USSR, "France") == 1
        assert t.gone_for_good("Truman Doctrine")


# --------------------------------------------------------------------------- #
# #20 Olympic Games
# --------------------------------------------------------------------------- #


class TestC020OlympicGames:
    def test_participate_sponsor_adds_two(self):
        """PD #20: "each player rolls one die, with the sponsor adding 2 to his roll.
        High roll gains 2 VP." Sponsor USSR 3+2=5 vs US 4 -> USSR +2."""
        t = Table(turn=2)
        t.dice(3, 4)
        t.play(USSR, "Olympic Games", "event")
        assert t.d.player is USA, "the opponent decides"
        t.answer("option:0")
        assert t.done and t.s.vp == 2
        assert t.in_cycle("Olympic Games")

    def test_ties_are_rerolled(self):
        """PD #20: "Reroll ties". 2+2 vs 4 ties; then 1+2 vs 6 -> opponent +2."""
        t = Table(turn=2)
        t.dice(2, 4, 1, 6)
        t.play(USSR, "Olympic Games", "event", "option:0")
        assert t.done and t.s.vp == -2

    def test_boycott_degrades_and_gives_four_ops(self):
        """PD #20: "If Opponent boycotts, degrade DEFCON one level and the Sponsor may
        Conduct Operations as if they played a 4 Ops card.\""""
        t = Table(turn=2, defcon=4)
        t.play(USSR, "Olympic Games", "event", "option:1")
        assert t.s.defcon == 3
        if t.d is not None and t.d.type is DecisionType.CARD_USE:
            t.answer("use:influence")
        for _ in range(4):
            t.answer("country:Poland")
        assert t.done and t.s.inf(USSR, "Poland") == 4

    def test_boycott_at_defcon_2_the_sponsor_loses(self):
        """R 8.1.3 EXAMPLE: "The US player plays Olympic Games, and the DEFCON status is
        at 2. The USSR player boycotts the game. ... The US player, as the phasing
        player, has lost.\""""
        t = Table(turn=2, defcon=2)
        t.play(USA, "Olympic Games", "event", "option:1")
        assert t.s.is_over and t.s.winner is USSR and t.s.win_reason is WinReason.DEFCON
        assert t.done


# --------------------------------------------------------------------------- #
# #21 NATO
# --------------------------------------------------------------------------- #


class TestC021NATO:
    def test_requires_marshall_plan_or_warsaw_pact(self):
        """PD #21: "Play after 'Marshall Plan' or 'Warsaw Pact'." Without either, the US
        may not choose the event."""
        t = Table(turn=2)
        t.hand(USA, "NATO")
        t.action_round(USA).answer("card:NATO")
        assert "use:event" not in t.keys()

    def test_without_prerequisite_ussr_ops_play_discards_it(self):
        """R 5.2 Example 1: the USSR plays NATO before either prerequisite: it gets the 4
        ops, the US does not get the event, and "despite being asterisked, the 'NATO'
        card would not be removed from play. It would be placed in the discard pile"."""
        t = Table(turn=2)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "NATO", "influence")
        t.finish_with()
        assert not t.s.in_play("NATO")
        assert t.in_cycle("NATO")

    def test_blocks_ussr_coups_and_realignments_in_us_controlled_europe(self):
        """PD #21: "USSR player may no longer make Coup or Realignment rolls in any US
        Controlled countries in Europe." Uncontrolled Europe stays open."""
        t = Table(turn=2, defcon=5)
        t.resolved("Marshall Plan")
        t.inf(usa={"Italy": 3, "Greece": 1}, ussr={"Italy": 1, "Greece": 1})
        t.play(USA, "NATO", "event")
        assert t.done and t.s.in_play("NATO")
        assert not rules.can_coup(t.s, USSR, "Italy")
        assert not rules.can_realign(t.s, USSR, "Italy")
        assert rules.can_coup(t.s, USSR, "Greece")
        assert rules.can_coup(t.s, USA, "Greece")

    def test_brush_war_cannot_attack_us_controlled_europe(self):
        """PD #21: "US Controlled countries in Europe may not be attacked by play of the
        Brush War event." (FAQ #21; V:B12.) Italy (stab 2, US-controlled) must not be a
        Brush War target for the USSR."""
        t = Table(turn=4)
        t.effect("NATO", USA)
        t.inf(usa={"Italy": 3, "Greece": 2})
        t.play(USSR, "Brush War", "event")
        assert t.d is not None and t.d.type is DecisionType.CHOOSE_COUNTRY
        assert "country:Italy" not in t.keys()
        assert "country:Greece" not in t.keys()
        assert "country:Turkey" in t.keys()


# --------------------------------------------------------------------------- #
# #22 Independent Reds
# --------------------------------------------------------------------------- #


class TestC022IndependentReds:
    def test_matches_ussr_influence(self):
        """PD #22: "Adds sufficient US Influence in either Yugoslavia, Romania, Bulgaria,
        Hungary, or Czechoslovakia to equal USSR Influence.\""""
        t = Table(turn=2)
        t.inf(ussr={"Hungary": 3, "Poland": 4, "Romania": 2}, usa={"Romania": 1})
        t.play(USA, "Independent Reds", "event")
        assert "country:Poland" not in t.keys()
        t.answer("country:Hungary")
        assert t.done
        assert t.s.inf(USA, "Hungary") == 3 and t.s.inf(USA, "Romania") == 1
        assert t.gone_for_good("Independent Reds")


# --------------------------------------------------------------------------- #
# #23 Marshall Plan
# --------------------------------------------------------------------------- #


class TestC023MarshallPlan:
    def test_seven_non_ussr_controlled_western_european_countries(self):
        """PD #23: "Add one US Influence in each of seven non-USSR Controlled Western
        European countries.\""""
        t = Table(turn=2)
        t.inf(ussr={"Italy": 2})
        t.play(USA, "Marshall Plan", "event")
        assert "country:Italy" not in t.keys() and "country:Poland" not in t.keys()
        picks = ["UK", "France", "West Germany", "Benelux", "Denmark", "Norway", "Greece"]
        t.answer(t.keys()[0] if False else f"country:{picks[0]}")
        assert f"country:{picks[0]}" not in t.keys()
        t.answer(*[f"country:{p}" for p in picks[1:]])
        assert t.done
        for p in picks:
            assert t.s.inf(USA, p) == 1

    def test_allows_nato_and_leaves_play(self):
        """PD #23: "Allows play of NATO." Asterisked: out of the draw cycle."""
        t = Table(turn=2)
        t.play(USA, "Marshall Plan", "event")
        t.finish_with()
        assert t.gone_for_good("Marshall Plan")
        t.hand(USA, "NATO")
        t.action_round(USA).answer("card:NATO")
        assert "use:event" in t.keys()


# --------------------------------------------------------------------------- #
# #24 Indo-Pakistani War
# --------------------------------------------------------------------------- #


class TestC024IndoPakistaniWar:
    def test_victory_and_penalty(self):
        """PD #24: "India or Pakistan invades the other (player's choice). Roll one die and
        subtract 1 for every opponent-controlled country adjacent to the target of the
        invasion. Player Victory on modified die roll of 4-6. Player adds 2 to Military
        Ops Track." USSR attacks Pakistan; US controls Iran (adjacent): 5-1=4 victory."""
        t = Table(turn=2)
        t.inf(usa={"Pakistan": 2, "Iran": 2})
        t.dice(5)
        t.play(USSR, "Indo-Pakistani War", "event", "country:Pakistan")
        assert t.done
        assert t.s.inf(USA, "Pakistan") == 0 and t.s.inf(USSR, "Pakistan") == 2
        assert t.s.vp == 2 and t.s.military_ops[USSR] == 2
        assert t.in_cycle("Indo-Pakistani War")

    def test_failure_still_counts_military_ops(self):
        """R 7.6 / FAQ #24: military ops are credited irrespective of success."""
        t = Table(turn=2)
        t.dice(3)
        t.play(USA, "Indo-Pakistani War", "event", "country:India")
        assert t.s.vp == 0 and t.s.military_ops[USA] == 2


# --------------------------------------------------------------------------- #
# #25 Containment
# --------------------------------------------------------------------------- #


class TestC025Containment:
    def test_plus_one_to_us_cards_max_four(self):
        """PD #25: "All further Operations cards played by US this turn add one to their
        value (to a maximum of 4)." (R 7.4.)"""
        t = Table(turn=2)
        t.play(USA, "Containment", "event")
        assert t.s.in_play("Containment")
        assert t.g.effective_ops("CIA Created", USA) == 2
        assert t.g.effective_ops("NATO", USA) == 4
        assert t.g.effective_ops("CIA Created", USSR) == 1, "US only (R 7.4.1)"

    def test_expires_at_turn_end(self):
        """PD #25: "this turn" -- gone after the turn."""
        t = Table(turn=2)
        t.effect("Containment", USA, expires="end_of_turn")
        t.turn_end()
        assert not t.s.in_play("Containment")
        assert t.gone_for_good("Containment")


# --------------------------------------------------------------------------- #
# #26 CIA Created
# --------------------------------------------------------------------------- #


class TestC026CIACreated:
    def test_one_op_of_operations(self):
        """PD #26: "USSR reveals hand this turn. Then the US may Conduct Operations as if
        they played a 1 Op card.\""""
        t = Table(turn=2)
        t.hand(USSR, "Fidel")
        t.play(USA, "CIA Created", "event")
        if t.d is not None and t.d.type is DecisionType.CARD_USE:
            t.answer("use:influence")
        t.answer("country:Mexico")
        assert t.done and t.s.inf(USA, "Mexico") == 1
        assert t.gone_for_good("CIA Created")

    # 'as if N Ops' coverage (Containment -> 2 ops, military ops) lives in
    # test_mech_as_if_ops.py under the unverified_rule marker.


# --------------------------------------------------------------------------- #
# #27 US/Japan Mutual Defense Pact
# --------------------------------------------------------------------------- #


class TestC027USJapanPact:
    def test_control_and_immunity(self):
        """PD #27: "US gains sufficient Influence in Japan for Control. USSR may no
        longer make Coup or Realignment rolls in Japan.\""""
        t = Table(turn=2, defcon=5)
        t.inf(ussr={"Japan": 2})
        t.play(USA, "US/Japan Mutual Defense Pact", "event")
        assert t.s.inf(USA, "Japan") == 6
        assert not rules.can_coup(t.s, USSR, "Japan")
        assert not rules.can_realign(t.s, USSR, "Japan")
        assert t.where("US/Japan Mutual Defense Pact") in ("removed", "in_play")


# --------------------------------------------------------------------------- #
# #28 Suez Crisis
# --------------------------------------------------------------------------- #


class TestC028SuezCrisis:
    def test_four_from_three_countries_max_two_each(self):
        """PD #28: "Remove a total of 4 US Influence from France, the United Kingdom or
        Israel. Remove no more than 2 Influence per country.\""""
        t = Table(turn=2)
        t.inf(usa={"France": 3, "UK": 5, "Israel": 1, "Italy": 2})
        t.play(USSR, "Suez Crisis", "event", "country:UK", "country:UK")
        assert "country:UK" not in t.keys() and "country:Italy" not in t.keys()
        t.answer("country:France", "country:France")
        assert t.done
        assert t.s.inf(USA, "UK") == 3 and t.s.inf(USA, "France") == 1
        assert t.gone_for_good("Suez Crisis")


# --------------------------------------------------------------------------- #
# #29 East European Unrest
# --------------------------------------------------------------------------- #


class TestC029EastEuropeanUnrest:
    def test_early_war_one_from_three_countries(self):
        """PD #29: "In Early or Mid War: Remove 1 USSR Influence from three countries in
        Eastern Europe.\""""
        t = Table(turn=2)
        t.inf(ussr={"Poland": 3, "Hungary": 2, "Romania": 1, "Bulgaria": 1})
        t.play(USA, "East European Unrest", "event", "country:Poland")
        assert "country:Poland" not in t.keys()
        t.answer("country:Hungary", "country:Romania")
        assert t.done
        assert (t.s.inf(USSR, "Poland"), t.s.inf(USSR, "Hungary"), t.s.inf(USSR, "Romania")) == (2, 1, 0)
        assert t.in_cycle("East European Unrest")

    def test_late_war_two_from_each_of_three_countries(self):
        """PD #29: "In Late War: Remove 2 USSR Influence from three countries in Eastern
        Europe." (V:B12.) Exactly three countries are hit; the removal may not be
        spread as 1+1+1+1+1+1 over six countries."""
        t = Table(turn=8)
        t.inf(ussr={n: 3 for n in ("Poland", "Hungary", "Romania", "Bulgaria", "Yugoslavia", "Czechoslovakia")})
        t.play(USA, "East European Unrest", "event")
        touched = []
        while t.d is not None:
            key = next(k for k in t.keys() if k.startswith("country:") and k[8:] not in touched)
            touched.append(key[8:])
            t.answer(key)
        hit = [n for n in ("Poland", "Hungary", "Romania", "Bulgaria", "Yugoslavia", "Czechoslovakia") if t.s.inf(USSR, n) < 3]
        assert len(hit) == 3, f"influence removed from {hit}"
        assert all(t.s.inf(USSR, n) == 1 for n in hit)


# --------------------------------------------------------------------------- #
# #30 Decolonization
# --------------------------------------------------------------------------- #


class TestC030Decolonization:
    def test_four_african_or_se_asian_countries(self):
        """PD #30: "Add one USSR Influence in each of any four African and/or SE Asian
        countries.\""""
        t = Table(turn=2)
        t.play(USSR, "Decolonization", "event", "country:Angola")
        assert "country:Angola" not in t.keys()
        assert "country:India" not in t.keys(), "India is Asia but not SE Asia"
        t.answer("country:Zaire", "country:Vietnam", "country:Thailand")
        assert t.done
        for n in ("Angola", "Zaire", "Vietnam", "Thailand"):
            assert t.s.inf(USSR, n) == 1
        assert t.in_cycle("Decolonization")


# --------------------------------------------------------------------------- #
# #31 Red Scare/Purge
# --------------------------------------------------------------------------- #


class TestC031RedScarePurge:
    def test_minus_one_to_opponent_minimum_one(self):
        """PD #31: "All further Operations cards played by your opponent this turn are -1
        to their value (to a minimum of 1 Op).\""""
        t = Table(turn=2)
        t.play(USSR, "Red Scare/Purge", "event")
        assert t.g.effective_ops("Duck and Cover", USA) == 2
        assert t.g.effective_ops("CIA Created", USA) == 1
        assert t.g.effective_ops("Duck and Cover", USSR) == 3
        assert t.in_cycle("Red Scare/Purge") or t.s.in_play("Red Scare/Purge")

    def test_applies_to_the_space_race_requirement(self):
        """FAQ #31: "The reduction is for any use of Ops, including Space Race." A
        printed 2-op card cannot attempt Earth Satellite (needs 2) under Red Scare."""
        t = Table(turn=2)
        t.effect("Red Scare/Purge", USSR, expires="end_of_turn")
        t.inf(usa={"Mexico": 1})
        t.hand(USA, "Independent Reds")
        t.action_round(USA).answer("card:Independent Reds")
        assert "use:space" not in t.keys()

    def test_coup_military_ops_credit_is_reduced(self):
        """R 7.4.2 EXAMPLE 1: under Red Scare a 2-op card used for a coup adds 1 to the
        roll and "would receive 1 point of the Required Military Operations track"."""
        t = Table(turn=2)
        t.effect("Red Scare/Purge", USSR, expires="end_of_turn")
        t.inf(ussr={"Angola": 1})
        t.dice(1)
        t.play(USA, "Independent Reds", "coup")  # Angola is the only target
        assert t.s.military_ops[USA] == 1


# --------------------------------------------------------------------------- #
# #32 UN Intervention
# --------------------------------------------------------------------------- #


class TestC032UNIntervention:
    def test_cancels_opponent_event_and_uses_its_ops(self):
        """PD #32: "Play this card simultaneously with a card containing your opponent's
        associated Event. The Event is cancelled, but you may use its Operations value to
        Conduct Operations. The cancelled event returns to the discard pile." The US
        cancels Fidel (2 ops) and places 2 influence; Cuba is untouched."""
        t = Table(turn=2)
        t.inf(usa={"Cuba": 1})
        t.hand(USA, "Fidel")
        t.play(USA, "UN Intervention", "event")
        if t.d is not None and t.d.type is DecisionType.CHOOSE_CARD:
            t.answer("card:Fidel")
        if t.d is not None and t.d.type is DecisionType.CARD_USE:
            t.answer("use:influence")
        t.answer("country:Mexico", "country:Mexico")
        assert t.done
        assert t.s.inf(USA, "Cuba") == 1 and t.s.inf(USSR, "Cuba") == 0
        assert t.in_cycle("Fidel"), "cancelled starred event returns to the discard"
        assert t.in_cycle("UN Intervention")

    def test_needs_an_opponent_event_card(self):
        """PD #32: must be played with a card containing the opponent's event."""
        t = Table(turn=2)
        t.hand(USA, "Duck and Cover")
        t.hand(USA, "UN Intervention")
        t.action_round(USA).answer("card:UN Intervention")
        assert "use:event" not in t.keys()

    def test_cannot_be_headlined(self):
        """PD #32: "May not be played during headline phase.\""""
        t = Table(turn=2)
        t.hand(USA, "UN Intervention", "Fidel")
        t.hand(USSR, "Duck and Cover")
        t.headline_phase()
        while t.d is not None and t.d.player is not USA:
            t.answer(t.keys()[0])
        assert t.d is not None and t.d.type is DecisionType.HEADLINE
        assert "card:UN Intervention" not in t.keys()

    def test_us_cancelling_its_own_card_does_not_work(self):
        """PD #32: the companion card must carry the *opponent's* event; a neutral card
        such as Olympic Games is not eligible."""
        t = Table(turn=2)
        t.hand(USA, "Olympic Games", "UN Intervention")
        t.action_round(USA).answer("card:UN Intervention")
        assert "use:event" not in t.keys()


# --------------------------------------------------------------------------- #
# #33 De-Stalinization
# --------------------------------------------------------------------------- #


class TestC033DeStalinization:
    def test_relocation_is_up_to_four(self):
        """PD #33: "USSR may relocate *up to* 4 Influence points". After moving one
        point the USSR may stop."""
        t = Table(turn=2)
        t.inf(ussr={"Poland": 4, "East Germany": 4})
        t.play(USSR, "De-Stalinization", "event", "country:Poland")
        assert t.d is not None and t.d.type is DecisionType.REMOVE_INFLUENCE
        assert "pass" in t.keys(), "'up to 4': the USSR may relocate fewer"

    def test_may_relocate_into_ussr_controlled_countries(self):
        """PD #33: "...to non-US controlled countries." (V:B12.) A USSR-controlled
        country is non-US-controlled and is a legal destination; a US-controlled one
        is not."""
        t = Table(turn=2)
        t.inf(ussr={"Poland": 4, "East Germany": 4}, usa={"Italy": 2})
        t.play(USSR, "De-Stalinization", "event")
        t.answer(*["country:Poland"] * 4)
        assert t.d is not None and t.d.type is DecisionType.PLACE_INFLUENCE
        assert "country:East Germany" in t.keys(), "USSR-controlled is non-US-controlled"
        assert "country:Italy" not in t.keys(), "US-controlled"

    def test_at_most_two_per_destination(self):
        """PD #33: "No more than 2 Influence may be placed in the same country.\""""
        t = Table(turn=2)
        t.inf(ussr={"Poland": 4})
        t.play(USSR, "De-Stalinization", "event")
        # Poland is the only source, but "up to 4" lets the USSR stop at any point, so
        # each removal is still asked (with a pass); take all four from Poland.
        t.answer(*["country:Poland"] * 4)
        t.answer("country:Angola", "country:Angola")
        assert "country:Angola" not in t.keys()
        t.answer("country:Zaire", "country:Zaire")
        assert t.done
        assert t.s.inf(USSR, "Angola") == 2 and t.s.inf(USSR, "Zaire") == 2
        assert t.s.inf(USSR, "Poland") == 0
        assert t.gone_for_good("De-Stalinization")


# --------------------------------------------------------------------------- #
# #34 Nuclear Test Ban
# --------------------------------------------------------------------------- #


class TestC034NuclearTestBan:
    def test_scores_then_improves(self):
        """PD #34: "Player earns VPs equal to current DEFCON level minus 2, then improve
        DEFCON two levels.\""""
        t = Table(turn=2, defcon=4)
        t.play(USA, "Nuclear Test Ban", "event")
        assert t.s.vp == -2 and t.s.defcon == 5
        assert t.in_cycle("Nuclear Test Ban")

    def test_at_defcon_2_scores_nothing(self):
        """PD #34 at DEFCON 2: 0 VP, DEFCON 4."""
        t = Table(turn=2, defcon=2)
        t.play(USSR, "Nuclear Test Ban", "event")
        assert t.s.vp == 0 and t.s.defcon == 4 and not t.s.is_over


# --------------------------------------------------------------------------- #
# #35 Formosan Resolution
# --------------------------------------------------------------------------- #


class TestC035FormosanResolution:
    def test_taiwan_counts_as_battleground_in_asia_scoring(self):
        """PD #35: "If this Event is in play, Taiwan shall be treated as a Battleground
        country for scoring purposes, if the US controls Taiwan when the Asia Scoring
        Card is played." US controls Taiwan only: presence 3 + 1 BG = 4."""
        t = Table(turn=2)
        t.effect("Formosan Resolution", USA)
        t.inf(usa={"Taiwan": 3})
        t.play(USA, "Asia Scoring")
        assert t.s.vp == -4

    def test_not_a_battleground_for_coups(self):
        """PD #35 (C1): "Taiwan is not a battleground country for any other game
        purpose." A coup in Taiwan does not degrade DEFCON."""
        t = Table(turn=2, defcon=5)
        t.effect("Formosan Resolution", USA)
        t.inf(usa={"Taiwan": 3}, ussr={"North Korea": 3})
        t.dice(1)
        t.play(USSR, "Duck and Cover", "coup", "yes")
        t.finish_with(lambda d: "country:Taiwan" if "country:Taiwan" in d.legal_keys else d.legal_keys[0])
        assert t.s.defcon == 4, "only Duck and Cover's own degrade"

    def test_cancelled_by_us_china_card_play(self):
        """PD #6 / #35: "This card is discarded after US play of The China Card." FAQ
        #6: also when the US plays the China Card on the Space Race."""
        t = Table(turn=2, china=USA)
        t.effect("Formosan Resolution", USA)
        t.inf(usa={"Japan": 1})
        t.play(USA, "The China Card", "space")
        assert t.done
        assert not t.s.in_play("Formosan Resolution")
        assert t.gone_for_good("Formosan Resolution") or t.in_cycle("Formosan Resolution")

    def test_applies_in_final_scoring(self):
        """FAQ #35/#101: "Does Taiwan count as a Battleground country if Formosan
        Resolution is in effect for Final Scoring at the end of turn 10? A. Yes.\""""
        t = Table(turn=10)
        t.effect("Formosan Resolution", USA)
        t.inf(usa={"Taiwan": 3})
        t.final_scoring()
        assert t.s.vp == -4

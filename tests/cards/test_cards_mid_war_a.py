"""Mid War cards #36-#58 (#37 and #38 are scoring cards, tested with the others).

Card text quoted from the official digital edition text (``PD``, the Deluxe card faces)
and the twilightstrategy.com transcription (``TS``) as collected in
``scratchpad/rules/cards.md``; rule numbers are the Deluxe rulebook (``R``); ``FAQ`` is
the designer-authorised FAQ v5 (2010); ``V:Bn`` points at an entry in ``verdicts.md``.
Expectations follow those sources, not the current implementation.
"""

from __future__ import annotations

import pytest

from _table import USA, USSR, Table
from twilight import rules
from twilight.decisions import DecisionType
from twilight.enums import WinReason


def option_with(t: Table, text: str) -> str:
    """The key of the pending option whose label contains *text*."""
    assert t.d is not None, "no decision pending"
    for a in t.d.options:
        if text in a.label:
            return a.key
    raise AssertionError(f"no option labelled {text!r} in {[a.label for a in t.d.options]}")


def answer_use(t: Table, use: str) -> None:
    """Answer a CARD_USE decision if one is pending (it is skipped when forced)."""
    if t.d is not None and t.d.type is DecisionType.CARD_USE:
        t.answer(f"use:{use}")


# --------------------------------------------------------------------------- #
# #36 Brush War
# --------------------------------------------------------------------------- #


class TestC036BrushWar:
    def test_victory_on_three(self):
        """PD #36: "Attack any country with a stability of 1 or 2. Roll a die and subtract
        1 for every adjacent enemy controlled country. Success on 3-6. Player adds 3 to
        his Military Ops Track. Effects of Victory: Player gains 1 VP and replaces all
        opponent's Influence with his Influence.\""""
        t = Table(turn=4)
        t.inf(usa={"Angola": 2})
        t.dice(3)
        t.play(USSR, "Brush War", "event", "country:Angola")
        assert t.done
        assert t.s.inf(USA, "Angola") == 0 and t.s.inf(USSR, "Angola") == 2
        assert t.s.vp == 1 and t.s.military_ops[USSR] == 3
        assert t.in_cycle("Brush War"), "not starred"

    def test_only_stability_one_or_two(self):
        """PD #36: "any country with a stability of 1 or 2" -- Cuba (3) is not a target."""
        t = Table(turn=4)
        t.play(USA, "Brush War", "event")
        assert "country:Nicaragua" in t.keys() and "country:Mexico" in t.keys()
        assert "country:Cuba" not in t.keys() and "country:UK" not in t.keys()

    def test_adjacent_enemy_control_subtracts_military_ops_anyway(self):
        """PD #36: -1 per adjacent enemy-controlled country; R 7.6: military ops are
        credited "irrespective of success". US controls Honduras next to Nicaragua:
        3 - 1 = 2, no victory, USSR still +3 military ops."""
        t = Table(turn=4)
        t.inf(usa={"Nicaragua": 1, "Honduras": 2})
        t.dice(3)
        t.play(USSR, "Brush War", "event", "country:Nicaragua")
        assert t.s.inf(USA, "Nicaragua") == 1 and t.s.vp == 0
        assert t.s.military_ops[USSR] == 3

    def test_enemy_superpower_counts_as_adjacent_controlled(self):
        """R 2.1.5: the superpower spaces "are considered 'adjacent controlled countries'
        for the purposes of events"; FAQ #36 (Deluxe): "When applying die roll modifiers
        for a Brush war, do the adjacent Superpowers count? A. Yes". USSR attacks Mexico
        (adjacent to the USA): 3 - 1 = 2, no victory."""
        t = Table(turn=4)
        t.inf(usa={"Mexico": 1})
        t.dice(3)
        t.play(USSR, "Brush War", "event", "country:Mexico")
        assert t.s.inf(USA, "Mexico") == 1 and t.s.vp == 0, "USA counts -1 against the USSR"
        assert t.s.military_ops[USSR] == 3

    def test_war_with_no_influence_still_scores(self):
        """R 7.6: "If there is no influence at risk in the war, the Event's player still
        receives victory points upon success". US attacks empty Angola, rolls 6."""
        t = Table(turn=4)
        t.dice(6)
        t.play(USA, "Brush War", "event", "country:Angola")
        assert t.s.vp == -1 and t.s.military_ops[USA] == 3


# --------------------------------------------------------------------------- #
# #39 Arms Race
# --------------------------------------------------------------------------- #


class TestC039ArmsRace:
    def test_three_vp_when_ahead_and_requirement_met(self):
        """PD #39: "If Phasing Player has more Military Operations points and has met the
        Required Military Operations amount, he scores 3 VP instead." DEFCON 3, USSR 3
        vs US 1."""
        t = Table(turn=4, defcon=3)
        t.s.military_ops = [3, 1]
        t.play(USSR, "Arms Race", "event")
        assert t.done and t.s.vp == 3
        assert t.in_cycle("Arms Race")

    def test_one_vp_when_ahead_only(self):
        """PD #39: "If Phasing Player has more Military Operations points, he scores 1
        VP." DEFCON 4, USSR 2 vs US 1 (requirement 4 not met)."""
        t = Table(turn=4, defcon=4)
        t.s.military_ops = [2, 1]
        t.play(USSR, "Arms Race", "event")
        assert t.s.vp == 1

    def test_nothing_when_not_ahead(self):
        """PD #39: only "more" scores; a tie scores nothing."""
        t = Table(turn=4, defcon=2)
        t.s.military_ops = [2, 2]
        t.play(USA, "Arms Race", "event")
        assert t.s.vp == 0

    def test_scores_for_the_us_as_phasing_player(self):
        """PD #39 for the US: US 5 vs USSR 0 at DEFCON 5 -> US +3."""
        t = Table(turn=4, defcon=5)
        t.s.military_ops = [0, 5]
        t.play(USA, "Arms Race", "event")
        assert t.s.vp == -3


# --------------------------------------------------------------------------- #
# #40 Cuban Missile Crisis
# --------------------------------------------------------------------------- #


class TestC040CubanMissileCrisis:
    def test_sets_defcon_to_two(self):
        """PD #40: "Set DEFCON to Level 2." From DEFCON 5 nobody loses."""
        t = Table(turn=4, defcon=5)
        t.play(USSR, "Cuban Missile Crisis", "event")
        assert t.done
        assert t.s.defcon == 2 and not t.s.is_over
        assert t.s.in_play("Cuban Missile Crisis")

    def test_opponent_coup_with_operations_loses(self):
        """PD #40: "Any further Coup attempt by your opponent this turn, anywhere on the
        board, will result in Global Thermonuclear War. Your opponent will lose the
        game." US coup in Zimbabwe (Africa, non-battleground) -> USSR wins."""
        t = Table(turn=4, defcon=5)
        t.play(USSR, "Cuban Missile Crisis", "event")
        t.inf(ussr={"Zimbabwe": 1})
        t.dice(6)
        t.play(USA, "Kitchen Debates", "coup")
        assert t.s.is_over
        assert t.s.winner is USSR and t.s.win_reason is WinReason.CUBAN_MISSILE_CRISIS

    def test_opponent_event_granted_coup_also_loses(self):
        """FAQ #40: "If a player is operating under Cuban Missile Crisis, the second he
        chose to initiate a coup, whether or not brought about by an opponent's card
        play, he loses." US free coup from Junta -> US loses."""
        t = Table(turn=4, defcon=5)
        t.effect("Cuban Missile Crisis", USSR, expires="end_of_turn")
        t.s.defcon = 2
        t.inf(ussr={"Guatemala": 1})
        t.dice(6)
        t.play(USA, "Junta", "event", "country:Mexico")
        t.answer(option_with(t, "coup"))
        if t.d is not None and t.d.type is DecisionType.CHOOSE_COUNTRY:
            t.answer("country:Guatemala")
        assert t.s.is_over
        assert t.s.winner is USSR and t.s.win_reason is WinReason.CUBAN_MISSILE_CRISIS

    def test_the_player_of_the_crisis_may_coup(self):
        """PD #40 restricts only "your opponent". The USSR's own coup is legal."""
        t = Table(turn=4, defcon=5)
        t.effect("Cuban Missile Crisis", USSR, expires="end_of_turn")
        t.s.defcon = 2
        t.inf(usa={"Zimbabwe": 1})
        t.dice(6)
        t.play(USSR, "Summit", "coup")
        assert not t.s.is_over
        assert t.s.inf(USSR, "Zimbabwe") == 4

    def test_us_cancels_with_two_from_turkey(self):
        """PD #40: "This event may be cancelled at any time if ... the US player removes 2
        Influence from either West Germany or Turkey." After cancelling, a US coup is
        safe."""
        t = Table(turn=4, defcon=5)
        t.effect("Cuban Missile Crisis", USSR, expires="end_of_turn")
        t.s.defcon = 2
        t.inf(usa={"Turkey": 2}, ussr={"Zimbabwe": 1})
        t.hand(USA, "Kitchen Debates")
        t.action_round(USA)
        assert t.d.type is DecisionType.CONFIRM and t.d.player is USA
        t.answer("yes")
        assert t.s.inf(USA, "Turkey") == 0
        assert not t.s.in_play("Cuban Missile Crisis")
        t.dice(6)
        t.answer("card:Kitchen Debates")
        answer_use(t, "coup")
        assert not t.s.is_over and t.s.inf(USA, "Zimbabwe") == 4

    def test_ussr_cancels_with_two_from_cuba(self):
        """PD #40: "... if the USSR player removes two Influence from Cuba"."""
        t = Table(turn=4, defcon=5)
        t.effect("Cuban Missile Crisis", USA, expires="end_of_turn")
        t.s.defcon = 2
        t.inf(ussr={"Cuba": 3})
        t.hand(USSR, "Summit")
        t.action_round(USSR)
        assert t.d.type is DecisionType.CONFIRM and t.d.player is USSR
        t.answer("yes")
        assert t.s.inf(USSR, "Cuba") == 1
        assert not t.s.in_play("Cuban Missile Crisis")

    def test_cancel_needs_two_from_one_country_not_one_from_each(self):
        """PD #40: "removes 2 Influence from either West Germany or Turkey"; FAQ #40: "the
        Deluxe Edition clarifies the 2 influence must be removed '...from either West
        Germany or Turkey' and thus, not 1 influence from both." With 1 + 1 the US has
        no way to cancel, so no cancel offer should be made."""
        t = Table(turn=4, defcon=5)
        t.effect("Cuban Missile Crisis", USSR, expires="end_of_turn")
        t.s.defcon = 2
        t.inf(usa={"West Germany": 1, "Turkey": 1})
        t.hand(USA, "Kitchen Debates")
        t.action_round(USA)
        assert t.d.type is DecisionType.PLAY_CARD, "1+1 cannot cancel the crisis"
        assert t.s.inf(USA, "West Germany") == 1 and t.s.inf(USA, "Turkey") == 1

    def test_cancel_removes_both_points_from_the_same_country(self):
        """FAQ #40 as above. US has West Germany 1, Turkey 2: the only legal
        cancellation takes both points from Turkey."""
        t = Table(turn=4, defcon=5)
        t.effect("Cuban Missile Crisis", USSR, expires="end_of_turn")
        t.s.defcon = 2
        t.inf(usa={"West Germany": 1, "Turkey": 2})
        t.hand(USA, "Kitchen Debates")
        t.action_round(USA).answer("yes")
        if t.d is not None and t.d.type is DecisionType.REMOVE_INFLUENCE:
            assert "country:West Germany" not in t.keys()
        assert t.s.inf(USA, "West Germany") == 1

    def test_cancel_is_possible_at_any_time(self):
        """PD #40: "may be cancelled at any time"; FAQ #40: "The card says 'at any time'
        and that would include during one of your [the opponent's] action rounds." The
        USSR plays CIA Created for ops; before using the event's 1 Op the US must be able
        to buy the crisis off (Turkey 2), so some cancel decision has to be offered to
        the US before the operations choice. (docs/known_gaps.md documents this gap.)"""
        t = Table(turn=4, defcon=5)
        t.effect("Cuban Missile Crisis", USSR, expires="end_of_turn")
        t.s.defcon = 2
        t.inf(usa={"Turkey": 2}, ussr={"Poland": 1})
        t.play(USSR, "CIA Created", "influence", "yes")
        assert t.d is not None and t.d.player is USA
        assert t.d.type is DecisionType.CONFIRM, (
            f"US offered {t.d.type} without a chance to cancel the crisis first"
        )

    def test_ends_with_the_turn_and_leaves_the_game(self):
        """FAQ errata #40: "Should not be underlined. The card does not have effects that
        carry beyond the current turn." Asterisked: removed after the event."""
        t = Table(turn=4, defcon=5)
        t.play(USSR, "Cuban Missile Crisis", "event")
        t.turn_end()
        assert not t.s.in_play("Cuban Missile Crisis")
        assert t.gone_for_good("Cuban Missile Crisis")


# --------------------------------------------------------------------------- #
# #41 Nuclear Subs
# --------------------------------------------------------------------------- #


class TestC041NuclearSubs:
    def test_us_battleground_coup_does_not_degrade(self):
        """PD #41: "US Coup attempts in Battleground Countries do not affect the DEFCON
        track for the remainder of the turn." Military ops still count (R 8.2.3)."""
        t = Table(turn=4, defcon=5)
        t.play(USA, "Nuclear Subs", "event")
        assert t.s.in_play("Nuclear Subs")
        t.inf(ussr={"Iran": 1})
        t.dice(6)
        t.play(USA, "Kitchen Debates", "coup")
        assert t.s.defcon == 5
        assert t.s.military_ops[USA] == 1

    def test_ussr_coups_still_degrade(self):
        """PD #41 covers only US coups."""
        t = Table(turn=4, defcon=5)
        t.effect("Nuclear Subs", USA, expires="end_of_turn")
        t.inf(usa={"Iran": 1})
        t.dice(6)
        t.play(USSR, "Summit", "coup")
        assert t.s.defcon == 4

    def test_events_that_move_defcon_are_unaffected(self):
        """TS #41: "This card's Event does not apply to any Event that would affect the
        DEFCON level"; FAQ #41: "'Actions' should be read as referring to use of
        Operations". Duck and Cover still degrades."""
        t = Table(turn=4, defcon=5)
        t.effect("Nuclear Subs", USA, expires="end_of_turn")
        t.play(USA, "Duck and Cover", "event")
        assert t.s.defcon == 4

    def test_does_not_protect_against_cuban_missile_crisis(self):
        """PD #41: "(does not affect Cuban Missile Crisis)". A US coup under a USSR
        crisis still loses."""
        t = Table(turn=4, defcon=5)
        t.effect("Nuclear Subs", USA, expires="end_of_turn")
        t.effect("Cuban Missile Crisis", USSR, expires="end_of_turn")
        t.s.defcon = 2
        t.inf(ussr={"Zimbabwe": 1})
        t.dice(6)
        t.play(USA, "Kitchen Debates", "coup")
        assert t.s.is_over and t.s.winner is USSR

    def test_ends_with_the_turn_and_leaves_the_game(self):
        """PD #41 "for the remainder of the turn"; asterisked."""
        t = Table(turn=4)
        t.play(USA, "Nuclear Subs", "event")
        t.turn_end()
        assert not t.s.in_play("Nuclear Subs")
        assert t.gone_for_good("Nuclear Subs")

    def test_ussr_playing_it_for_ops_gives_the_us_the_effect(self):
        """R 5.2: the opponent's event still occurs when the card is used for ops."""
        t = Table(turn=4)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Nuclear Subs", "influence", "yes", "country:Poland", "country:Poland")
        t.finish_with()
        assert t.s.in_play("Nuclear Subs", owner=USA)


# --------------------------------------------------------------------------- #
# #42 Quagmire (card level; the trap loop is in test_mech_traps.py)
# --------------------------------------------------------------------------- #


class TestC042Quagmire:
    def test_event_places_the_trap_on_the_us(self):
        """PD #42: "On next action round, US player must discard ..." -- the card stays
        in front of the US until cancelled (not discarded, not yet reshuffleable)."""
        t = Table(turn=4)
        t.play(USSR, "Quagmire", "event")
        assert t.done
        assert t.s.in_play("Quagmire", owner=USA)
        assert t.where("Quagmire") == "in_play"

    def test_cancels_norad(self):
        """TS #42: "This Event cancels the effect(s) of the '#106 - NORAD' Event"; PD #106:
        "This event cancelled by 'Quagmire'." NORAD is asterisked, so it leaves the
        game."""
        t = Table(turn=4)
        t.effect("NORAD", USA)
        t.play(USSR, "Quagmire", "event")
        assert not t.s.in_play("NORAD")
        assert t.gone_for_good("NORAD")

    def test_us_playing_it_for_ops_traps_itself(self):
        """R 5.2: the US using Quagmire for operations triggers the USSR event."""
        t = Table(turn=4)
        t.play(USA, "Quagmire", "influence", "no", "country:Mexico", "country:Mexico", "country:Mexico")
        t.finish_with()
        assert t.s.in_play("Quagmire", owner=USA)


# --------------------------------------------------------------------------- #
# #43 SALT Negotiations
# --------------------------------------------------------------------------- #


class TestC043SaltNegotiations:
    def test_improves_defcon_two_levels_capped_at_five(self):
        """PD #43: "Improve DEFCON two levels." R 8.1.2: never above 5."""
        t = Table(turn=4, defcon=4)
        t.play(USSR, "Salt Negotiations", "event")
        t.finish_with()
        assert t.s.defcon == 5

    def test_reclaims_one_non_scoring_card(self):
        """PD #43: "Player may sort through discard pile and reclaim one non-scoring
        card". Scoring cards are not offered; declining is allowed ("may")."""
        t = Table(turn=4, defcon=2)
        t.discard("Duck and Cover", "Asia Scoring")
        t.play(USA, "Salt Negotiations", "event")
        assert t.s.defcon == 4
        assert set(t.keys()) == {"card:Duck and Cover", "pass"}
        t.answer("card:Duck and Cover")
        assert t.done
        assert "Duck and Cover" in t.s.hands[USA] and "Duck and Cover" not in t.s.discard

    @pytest.mark.parametrize("side", [USSR, USA])
    def test_coups_are_minus_one_for_both_players(self, side):
        """PD #43: "Further Coup attempts incur -1 die roll modifier for both players for
        the remainder of the turn." 1-op coup in Angola (needs > 2): roll 2 + 1 - 1 = 2
        fails."""
        t = Table(turn=4, defcon=5)
        t.effect("Salt Negotiations", USSR, expires="end_of_turn")
        t.inf(ussr={"Angola": 1}, usa={"Angola": 1})
        t.dice(2)
        t.play(side, "Summit", "coup")
        assert t.s.inf(side.opponent, "Angola") == 1, "the coup must fail"

    def test_ends_with_the_turn_and_leaves_the_game(self):
        """PD #43 "for the remainder of the turn"; asterisked."""
        t = Table(turn=4)
        t.play(USSR, "Salt Negotiations", "event")
        t.finish_with()
        t.turn_end()
        assert not t.s.in_play("Salt Negotiations")
        assert t.gone_for_good("Salt Negotiations")
        assert rules.coup_modifier(t.s, USSR, "Angola") == 0


# --------------------------------------------------------------------------- #
# #44 Bear Trap (card level; the trap loop is in test_mech_traps.py)
# --------------------------------------------------------------------------- #


class TestC044BearTrap:
    def test_event_places_the_trap_on_the_ussr(self):
        """PD #44: "On next action round, USSR player must discard ..." -- the card stays
        in front of the USSR."""
        t = Table(turn=4)
        t.play(USA, "Bear Trap", "event")
        assert t.done
        assert t.s.in_play("Bear Trap", owner=USSR)
        assert t.where("Bear Trap") == "in_play"

    def test_ussr_playing_it_for_ops_traps_itself(self):
        """R 5.2: the USSR using Bear Trap for operations triggers the US event."""
        t = Table(turn=4)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Bear Trap", "influence", "no", "country:Poland", "country:Poland", "country:Poland")
        t.finish_with()
        assert t.s.in_play("Bear Trap", owner=USSR)


# --------------------------------------------------------------------------- #
# #45 Summit
# --------------------------------------------------------------------------- #


def _ussr_dominates_middle_east(t: Table) -> None:
    """USSR controls Iraq (battleground) and Syria (non-battleground); US nothing."""
    t.inf(ussr={"Iraq": 3, "Syria": 2})


class TestC045Summit:
    def test_region_bonus_and_winner_moves_defcon(self):
        """PD #45: "Both players roll a die. Each adds 1 for each Region they Dominate or
        Control. High roller gains 2 VP and may move DEFCON marker one level in either
        direction." Both roll 3; USSR dominates the Middle East -> 4 vs 3."""
        t = Table(turn=4, defcon=4)
        _ussr_dominates_middle_east(t)
        t.dice(3, 3)
        t.play(USA, "Summit", "event")
        assert t.s.vp == 2
        assert t.d is not None and t.d.player is USSR, "the winner chooses"
        t.answer(option_with(t, "degrade"))
        assert t.done and t.s.defcon == 3
        assert t.in_cycle("Summit")

    def test_winner_may_improve_or_leave_defcon(self):
        """PD #45: "may move DEFCON marker one level in either direction"."""
        t = Table(turn=4, defcon=3)
        _ussr_dominates_middle_east(t)
        t.dice(3, 3)
        t.play(USSR, "Summit", "event")
        assert {option_with(t, "improve"), option_with(t, "leave")} <= set(t.keys())
        t.answer(option_with(t, "improve"))
        assert t.s.defcon == 4

    def test_ties_are_not_rerolled(self):
        """PD #45: "Do not reroll ties." Nothing happens, and only two dice are rolled."""
        t = Table(turn=4, defcon=4)
        t.dice(4, 4, 1, 6)
        t.play(USSR, "Summit", "event")
        assert t.done
        assert t.s.vp == 0 and t.s.defcon == 4
        assert t.s.rng.rolls_made == [4, 4]

    def test_degrading_to_one_loses_for_the_phasing_player(self):
        """V:B9 table: "Summit 胜者下调 DEFCON 至 1 -> phasing player（Summit 的打出者），
        无论谁掷赢" (R 8.1.3). The USSR plays Summit at DEFCON 2, the US wins and
        degrades: the USSR loses."""
        t = Table(turn=4, defcon=2)
        t.inf(usa={"Iraq": 3, "Syria": 2})
        t.dice(3, 3)
        t.play(USSR, "Summit", "event")
        assert t.d.player is USA
        t.answer(option_with(t, "degrade"))
        assert t.s.is_over and t.s.winner is USA and t.s.win_reason is WinReason.DEFCON

    def test_shuttle_diplomacy_does_not_affect_summit(self):
        """FAQ #73 / V:B2 point 5: Shuttle Diplomacy "will not affect Summit or Kitchen
        Debates". With it in play the USSR still dominates the Middle East: 3+1 vs 3."""
        t = Table(turn=4, defcon=4)
        t.effect("Shuttle Diplomacy", USA)
        _ussr_dominates_middle_east(t)
        t.dice(3, 3)
        t.play(USA, "Summit", "event")
        assert t.s.vp == 2, "the USSR should win the Summit roll 4 vs 3"
        assert t.s.in_play("Shuttle Diplomacy"), "Summit does not spend Shuttle Diplomacy"


# --------------------------------------------------------------------------- #
# #46 How I Learned to Stop Worrying
# --------------------------------------------------------------------------- #

HILTSW = "How I Learned to Stop Worrying"


class TestC046HowILearned:
    def test_sets_any_level_and_counts_five_military_ops(self):
        """PD #46: "Set the DEFCON at any level you want (1-5). This event counts as 5
        Military Operations for the purpose of required Military Operations.\""""
        t = Table(turn=4, defcon=2)
        t.s.military_ops = [1, 0]
        t.play(USSR, HILTSW, "event")
        assert {f"option:{i}" for i in range(5)} <= set(t.keys())
        t.answer("option:4")
        assert t.done and t.s.defcon == 5
        assert t.s.military_ops[USSR] == 5
        assert t.gone_for_good(HILTSW)

    def test_setting_defcon_one_loses_for_the_player(self):
        """R 8.1.3: the phasing player who moves DEFCON to 1 loses."""
        t = Table(turn=4, defcon=4)
        t.play(USA, HILTSW, "event", "option:0")
        assert t.s.is_over and t.s.winner is USSR and t.s.win_reason is WinReason.DEFCON

    def test_via_grain_sales_the_phasing_ussr_loses(self):
        """FAQ #67: "The USSR plays Grain Sales to Soviets and US player chooses How I
        Learned to Stop Worrying. Can the US player set the DEFCON at 1? The USSR would
        be the phasing player, right? ... A Correct". V:B9 table row 2."""
        t = Table(turn=4, defcon=4)
        t.inf(ussr={"Poland": 1})
        t.hand(USSR, HILTSW)
        t.play(USSR, "Grain Sales to Soviets", "influence", "yes")
        # Grain Sales: the US takes the only USSR card and plays it.
        assert t.d.type is DecisionType.CONFIRM and t.d.player is USA
        t.answer("yes")
        answer_use(t, "event")
        t.answer("option:0")
        assert t.s.defcon == 1
        assert t.s.is_over and t.s.winner is USA and t.s.win_reason is WinReason.DEFCON


# --------------------------------------------------------------------------- #
# #47 Junta
# --------------------------------------------------------------------------- #


class TestC047Junta:
    def test_places_two_in_one_latin_american_country(self):
        """PD #47: "Place 2 Influence in any one Central or South American country.\""""
        t = Table(turn=4)
        t.play(USSR, "Junta", "event")
        assert "country:Chile" in t.keys() and "country:Cuba" in t.keys()
        assert "country:Angola" not in t.keys()
        t.answer("country:Chile")
        assert t.s.inf(USSR, "Chile") == 2
        t.finish_with(lambda d: option_with(t, "neither") if d.type is DecisionType.CHOOSE_OPTION else d.legal_keys[0])
        assert t.in_cycle("Junta"), "not starred"

    def test_free_coup_does_not_count_military_ops(self):
        """PD #47: "Then you may make a free Coup attempt ... (using this card's
        Operations Value)"; R 8.2.5: "Events that allow a free Coup roll do not count
        towards required Military Operations." (FAQ #47.) R 6.3.5: a free coup in a
        battleground still lowers DEFCON. Panama (BG, stab 2): 6 + 2 - 4 = 4."""
        t = Table(turn=4, defcon=5)
        t.inf(usa={"Panama": 1})
        t.dice(6)
        t.play(USSR, "Junta", "event", "country:Chile")
        t.answer(option_with(t, "coup"))
        if t.d is not None and t.d.type is DecisionType.CHOOSE_COUNTRY:
            t.answer("country:Panama")
        assert t.s.inf(USA, "Panama") == 0 and t.s.inf(USSR, "Panama") == 3
        assert t.s.military_ops[USSR] == 0
        assert t.s.defcon == 4

    def test_realignment_rolls_use_the_card_value(self):
        """PD #47 / FAQ #47: "the number of Realignment rolls will depend on the cards Op
        Value, which is normally 2" -- two rolls in Central/South America."""
        t = Table(turn=4)
        t.inf(usa={"Panama": 3, "Chile": 3})
        t.dice(6, 1, 6, 1)
        t.play(USSR, "Junta", "event", "country:Mexico")
        t.answer(option_with(t, "realign"))
        t.answer("country:Panama")
        assert t.d is not None, "a second realignment roll should be available"
        t.answer("country:Chile")
        assert t.done
        assert t.s.inf(USA, "Panama") < 3 and t.s.inf(USA, "Chile") < 3


# --------------------------------------------------------------------------- #
# #48 Kitchen Debates
# --------------------------------------------------------------------------- #


class TestC048KitchenDebates:
    def test_two_vp_when_us_controls_more_battlegrounds(self):
        """PD #48: "If the US controls more Battleground countries than the USSR, poke
        opponent in chest and gain 2 VP!\""""
        t = Table(turn=4)
        t.inf(usa={"Iran": 2, "Panama": 2}, ussr={"Iraq": 3})
        t.play(USA, "Kitchen Debates", "event")
        assert t.done and t.s.vp == -2
        assert t.gone_for_good("Kitchen Debates")

    def test_ussr_ops_play_triggers_it(self):
        """R 5.2: the USSR using it for operations gives the US its 2 VP; asterisk
        removes it."""
        t = Table(turn=4)
        t.inf(usa={"Iran": 2, "Panama": 2}, ussr={"Poland": 1})
        t.play(USSR, "Kitchen Debates", "influence", "yes")
        t.finish_with()
        assert t.s.vp == -2
        assert t.gone_for_good("Kitchen Debates")

    def test_condition_unmet_ussr_ops_play_returns_it_to_the_discard(self):
        """R 5.2: "If a card play triggers an opponent's Event, but that Event cannot
        occur because ... a condition expressed in the Event has not been met, the Event
        does not occur. In this instance, cards with an asterisk Event are returned to
        the discard pile"."""
        t = Table(turn=4)
        t.inf(usa={"Iran": 2}, ussr={"Iraq": 3, "Poland": 4})
        t.play(USSR, "Kitchen Debates", "influence")
        t.finish_with()
        assert t.s.vp == 0
        assert t.in_cycle("Kitchen Debates")

    def test_shuttle_diplomacy_does_not_affect_it(self):
        """FAQ #73: Shuttle Diplomacy "will not affect Summit or Kitchen Debates". US 2
        battlegrounds vs USSR 1 (Iraq) -> US +2 regardless."""
        t = Table(turn=4)
        t.effect("Shuttle Diplomacy", USA)
        t.inf(usa={"Iran": 2, "Panama": 2}, ussr={"Iraq": 3})
        t.play(USA, "Kitchen Debates", "event")
        assert t.s.vp == -2
        assert t.s.in_play("Shuttle Diplomacy")


# --------------------------------------------------------------------------- #
# #49 Missile Envy
# --------------------------------------------------------------------------- #


class TestC049MissileEnvy:
    def test_takes_highest_card_opponent_event_used_for_ops(self):
        """PD #49: "Exchange this card for your opponent's highest valued Operations card
        in his hand. ... If it contains opponent's event, use Operations value without
        triggering event." NATO (4, US event) is taken; the USSR spends 4 ops; Missile
        Envy goes to the US hand."""
        t = Table(turn=4)
        t.resolved("Marshall Plan")
        t.inf(ussr={"Poland": 1})
        t.hand(USA, "NATO", "Duck and Cover")
        t.play(USSR, "Missile Envy", "event")
        assert t.d.player is USA and t.keys() == ("card:NATO",), "only the highest card"
        t.answer("card:NATO")
        answer_use(t, "influence")
        t.answer(*["country:Poland"] * 4)
        assert t.done
        assert t.s.inf(USSR, "Poland") == 5
        assert not t.s.in_play("NATO") and t.in_cycle("NATO"), "no NATO event"
        assert "Missile Envy" in t.s.hands[USA]

    def test_ties_are_chosen_by_the_opponent(self):
        """PD #49: "If two or more cards are tied, opponent chooses.\""""
        t = Table(turn=4)
        t.hand(USA, "Duck and Cover", "Five-Year Plan", "CIA Created")
        t.play(USSR, "Missile Envy", "event")
        assert t.d.player is USA
        assert set(t.keys()) == {"card:Duck and Cover", "card:Five-Year Plan"}

    def test_own_event_occurs_immediately(self):
        """PD #49: "If the exchanged card contains your event, or an event applicable to
        both players, it occurs immediately." Taking Fidel fires it for the USSR."""
        t = Table(turn=4)
        t.inf(usa={"Cuba": 2})
        t.hand(USA, "Fidel")
        t.play(USSR, "Missile Envy", "event", "card:Fidel")
        t.finish_with()
        assert t.s.inf(USA, "Cuba") == 0 and t.s.inf(USSR, "Cuba") == 3
        assert t.gone_for_good("Fidel")

    def test_neutral_event_occurs_for_the_taker(self):
        """PD #49 "an event applicable to both players": Nuclear Test Ban for the USSR at
        DEFCON 4 -> USSR +2 VP, DEFCON 5."""
        t = Table(turn=4, defcon=4)
        t.hand(USA, "Nuclear Test Ban")
        t.play(USSR, "Missile Envy", "event", "card:Nuclear Test Ban")
        t.finish_with()
        assert t.s.vp == 2 and t.s.defcon == 5

    def test_scoring_cards_are_not_operations_cards(self):
        """PD #49 "highest valued Operations card": a scoring card (no ops) is never
        taken."""
        t = Table(turn=4)
        t.hand(USA, "Asia Scoring", "CIA Created")
        t.play(USSR, "Missile Envy", "event")
        assert t.keys() == ("card:CIA Created",)

    def test_taker_keeps_its_own_ops_modifiers(self):
        """R 7.4.1: ops modifiers stay with the player. Under Brezhnev Doctrine the USSR
        spends a taken 3-op US card as 4 ops."""
        t = Table(turn=4)
        t.effect("Brezhnev Doctrine", USSR, expires="end_of_turn")
        t.inf(ussr={"Poland": 1})
        t.hand(USA, "Duck and Cover")
        t.play(USSR, "Missile Envy", "event", "card:Duck and Cover")
        answer_use(t, "influence")
        t.answer(*["country:Poland"] * 4)
        assert t.done and t.s.inf(USSR, "Poland") == 5

    def test_opponent_must_spend_missile_envy_for_ops_next_round(self):
        """PD #49: "Opponent must use this card for Operations during his next action
        round." The US gets no card choice, and no event or space option."""
        t = Table(turn=4)
        t.hand(USA, "CIA Created", "Duck and Cover")
        t.play(USSR, "Missile Envy", "event", "card:Duck and Cover")
        t.finish_with()
        t.action_round(USA)
        assert t.d is not None and t.d.type is not DecisionType.PLAY_CARD
        if t.d.type is DecisionType.CARD_USE:
            assert "use:event" not in t.keys() and "use:space" not in t.keys()
        t.finish_with()
        assert "Missile Envy" not in t.s.hands[USA]

    def test_missile_envy_is_discarded_once_spent(self):
        """R 4.3 / 5.1: a card played for operations goes to the discard pile; #49 is not
        asterisked. After the US spends it on its forced action round it must be in the
        discard pile (available to SALT / Star Wars / the next reshuffle), not held off
        the board until the turn ends."""
        t = Table(turn=4)
        t.hand(USA, "CIA Created", "Duck and Cover")
        t.play(USSR, "Missile Envy", "event", "card:Duck and Cover")
        t.finish_with()
        t.action_round(USA)
        t.finish_with()
        assert t.in_cycle("Missile Envy"), f"Missile Envy is {t.where('Missile Envy')}"


# --------------------------------------------------------------------------- #
# #50 We Will Bury You
# --------------------------------------------------------------------------- #

WWBY = "We Will Bury You"


class TestC050WeWillBuryYou:
    def test_degrades_then_scores_after_the_next_us_round(self):
        """PD #50: "Unless UN Invervention is played as an Event on the US player's next
        round, USSR gains 3 VP ... Degrade DEFCON one level.\""""
        t = Table(turn=4, defcon=4)
        t.play(USSR, WWBY, "event")
        assert t.s.defcon == 3 and t.s.vp == 0
        t.play(USA, "Panama Canal Returned", "event")
        assert t.done and t.s.vp == 3
        assert t.gone_for_good(WWBY)

    def test_un_intervention_cancels_the_vp(self):
        """PD #50 as above: UN Intervention played as an event on that round."""
        t = Table(turn=4, defcon=4)
        t.play(USSR, WWBY, "event")
        t.hand(USA, "Fidel")
        t.play(USA, "UN Intervention", "event")
        t.finish_with(lambda d: "country:Mexico" if "country:Mexico" in d.legal_keys else d.legal_keys[0])
        assert t.s.vp == 0
        assert not t.s.has_deferred(card=WWBY)

    def test_vp_come_before_any_us_vp_award(self):
        """PD #50: "USSR gains 3 VP prior to any US VP award." FAQ #49/#50: score USSR
        +17, the US does not play UN Intervention but plays Duck and Cover: "The USSR
        will win an Automatic Victory in this case"."""
        t = Table(turn=4, defcon=5, vp=17)
        t.play(USSR, WWBY, "event")
        t.play(USA, "Duck and Cover", "event")
        assert t.s.is_over and t.s.winner is USSR
        assert t.s.win_reason is WinReason.VICTORY_POINTS

    def test_at_defcon_2_the_ussr_loses_and_scores_nothing(self):
        """FAQ #49/#50: "because the DEFCON was immediately degraded with the Event,
        Nuclear war occurs first and the USSR player loses (8.1.3)." V:B10 item 6."""
        t = Table(turn=4, defcon=2)
        t.play(USSR, WWBY, "event")
        assert t.s.is_over and t.s.winner is USA and t.s.win_reason is WinReason.DEFCON
        assert t.s.vp == 0
        assert not t.s.has_deferred(card=WWBY)

    def test_us_ops_play_scores_after_the_following_us_round(self):
        """R 5.2 + PD #50 "the US player's next round"; FAQ #50: "there has to be a next
        Action Round for the VP to be awarded". When the US itself uses the card for
        ops, the 3 VP wait for the US's *next* action round."""
        t = Table(turn=4, defcon=5)
        t.play(USA, WWBY, "influence", "yes", *["country:Mexico"] * 4)
        t.finish_with()
        assert t.s.defcon == 4 and t.s.vp == 0
        t.play(USA, "Panama Canal Returned", "event")
        assert t.s.vp == 3


# --------------------------------------------------------------------------- #
# #51 Brezhnev Doctrine
# --------------------------------------------------------------------------- #


class TestC051BrezhnevDoctrine:
    def test_plus_one_to_ussr_cards_max_four(self):
        """PD #51: "All further Operations cards played by the USSR this turn increase
        their Ops value by one (to a maximum of 4).\""""
        t = Table(turn=4)
        t.play(USSR, "Brezhnev Doctrine", "event")
        assert t.s.in_play("Brezhnev Doctrine")
        assert t.g.effective_ops("Summit", USSR) == 2
        assert t.g.effective_ops("Muslim Revolution", USSR) == 4
        assert t.g.effective_ops("Summit", USA) == 1, "USSR only (R 7.4.1)"

    def test_a_one_op_card_places_two(self):
        """PD #51 applied to an actual play: Summit (1) places 2 influence."""
        t = Table(turn=4)
        t.effect("Brezhnev Doctrine", USSR, expires="end_of_turn")
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Summit", "influence", "country:Poland")
        assert t.d is not None, "a second point should be available"
        t.answer("country:Poland")
        assert t.done and t.s.inf(USSR, "Poland") == 3

    def test_expires_at_turn_end(self):
        """PD #51 "this turn"; asterisked."""
        t = Table(turn=4)
        t.play(USSR, "Brezhnev Doctrine", "event")
        t.turn_end()
        assert not t.s.in_play("Brezhnev Doctrine")
        assert t.gone_for_good("Brezhnev Doctrine")


# --------------------------------------------------------------------------- #
# #52 Portuguese Empire Crumbles
# --------------------------------------------------------------------------- #


class TestC052PortugueseEmpireCrumbles:
    def test_two_in_each(self):
        """PD #52: "Add 2 USSR Influence in both SE African States and Angola.\""""
        t = Table(turn=4)
        t.play(USSR, "Portuguese Empire Crumbles", "event")
        assert t.done
        assert t.s.inf(USSR, "SE African States") == 2 and t.s.inf(USSR, "Angola") == 2
        assert t.gone_for_good("Portuguese Empire Crumbles")

    def test_us_ops_play_triggers_it(self):
        """R 5.2: the US using it for ops still gives the USSR its influence."""
        t = Table(turn=4)
        t.play(USA, "Portuguese Empire Crumbles", "influence", "yes", "country:Mexico", "country:Mexico")
        t.finish_with()
        assert t.s.inf(USSR, "Angola") == 2
        assert t.gone_for_good("Portuguese Empire Crumbles")


# --------------------------------------------------------------------------- #
# #53 South African Unrest
# --------------------------------------------------------------------------- #


class TestC053SouthAfricanUnrest:
    def test_two_in_south_africa(self):
        """PD #53: "USSR either adds 2 Influence in South Africa ...\""""
        t = Table(turn=4)
        t.play(USSR, "South African Unrest", "event", "option:0")
        assert t.done and t.s.inf(USSR, "South Africa") == 2
        assert t.in_cycle("South African Unrest")

    def test_one_plus_two_in_adjacent_countries_split_allowed(self):
        """PD #53: "... or adds 1 Influence in South Africa and 2 Influence in any
        countries adjacent to South Africa." FAQ #53: "effectively allowing one each in
        Angola and Botswana."""
        t = Table(turn=4)
        t.play(USSR, "South African Unrest", "event", "option:1")
        assert set(t.keys()) == {"country:Angola", "country:Botswana"}
        t.answer("country:Angola", "country:Botswana")
        assert t.done
        assert t.s.inf(USSR, "South Africa") == 1
        assert t.s.inf(USSR, "Angola") == 1 and t.s.inf(USSR, "Botswana") == 1


# --------------------------------------------------------------------------- #
# #54 Allende
# --------------------------------------------------------------------------- #


class TestC054Allende:
    def test_two_in_chile(self):
        """PD #54: "USSR receives 2 Influence in Chile.\""""
        t = Table(turn=4)
        t.play(USSR, "Allende", "event")
        assert t.done and t.s.inf(USSR, "Chile") == 2
        assert t.gone_for_good("Allende")


# --------------------------------------------------------------------------- #
# #55 Willy Brandt
# --------------------------------------------------------------------------- #


class TestC055WillyBrandt:
    def test_vp_influence_and_nato_cancelled_for_west_germany(self):
        """PD #55: "USSR receives gains 1 VP. USSR receives 1 Influence in West Germany.
        Cancels NATO for West Germany." US-controlled West Germany becomes open to USSR
        coups under NATO at DEFCON 5; France stays protected."""
        t = Table(turn=4, defcon=5)
        t.effect("NATO", USA)
        t.inf(usa={"West Germany": 6, "France": 4}, ussr={"France": 1})
        assert not rules.can_coup(t.s, USSR, "West Germany")
        t.play(USSR, "Willy Brandt", "event")
        assert t.s.vp == 1 and t.s.inf(USSR, "West Germany") == 1
        assert rules.controls(t.s, USA, "West Germany")
        assert rules.can_coup(t.s, USSR, "West Germany")
        assert not rules.can_coup(t.s, USSR, "France")
        assert t.gone_for_good("Willy Brandt")

    def test_prevented_by_tear_down_this_wall(self):
        """PD #55: "This event unplayable and/or cancelled by Tear Down This Wall." The
        USSR cannot choose the event; the US using it for ops triggers nothing and it
        returns to the discard (R 5.2)."""
        t = Table(turn=8)
        t.resolved("Tear Down This Wall")
        t.hand(USSR, "Willy Brandt")
        t.action_round(USSR).answer("card:Willy Brandt")
        assert t.d is None or "use:event" not in t.keys()

        t = Table(turn=8)
        t.resolved("Tear Down This Wall")
        t.play(USA, "Willy Brandt", "influence", "country:Mexico", "country:Mexico")
        t.finish_with()
        assert t.s.vp == 0 and t.s.inf(USSR, "West Germany") == 0
        assert t.in_cycle("Willy Brandt")

    def test_tear_down_this_wall_restores_nato_in_west_germany(self):
        """PD #55 / #96: Tear Down This Wall "Cancels/prevent Willy Brandt" (V:B12). Once
        cancelled, NATO again protects a US-controlled West Germany."""
        t = Table(turn=8, defcon=5)
        t.effect("NATO", USA)
        t.inf(usa={"West Germany": 6})
        t.play(USSR, "Willy Brandt", "event")
        assert rules.can_coup(t.s, USSR, "West Germany")
        t.play(USA, "Tear Down This Wall", "event")
        t.finish_with(
            lambda d: option_with(t, "no coup") if d.type is DecisionType.CHOOSE_OPTION else
            ("pass" if "pass" in d.legal_keys else d.legal_keys[0])
        )
        assert rules.controls(t.s, USA, "West Germany")
        assert not rules.can_coup(t.s, USSR, "West Germany")


# --------------------------------------------------------------------------- #
# #56 Muslim Revolution
# --------------------------------------------------------------------------- #


class TestC056MuslimRevolution:
    def test_clears_two_listed_countries(self):
        """PD #56: "Remove all US Influence in two of the following countries: Sudan,
        Iran, Iraq, Egypt, Libya, Saudi Arabia, Syria, Jordan.\""""
        t = Table(turn=4)
        t.inf(usa={"Iran": 3, "Egypt": 2, "Jordan": 1, "Israel": 4})
        t.play(USSR, "Muslim Revolution", "event")
        assert "country:Israel" not in t.keys()
        t.answer("country:Iran", "country:Egypt")
        assert t.done
        assert t.s.inf(USA, "Iran") == 0 and t.s.inf(USA, "Egypt") == 0
        assert t.s.inf(USA, "Jordan") == 1 and t.s.inf(USA, "Israel") == 4
        assert t.in_cycle("Muslim Revolution"), "not starred"

    def test_us_ops_play_triggers_it(self):
        """R 5.2: the US using it for ops still fires the USSR event."""
        t = Table(turn=4)
        t.inf(usa={"Iran": 3, "Egypt": 2, "Jordan": 1})
        t.play(USA, "Muslim Revolution", "influence", "yes", "country:Iran", "country:Egypt")
        t.finish_with(lambda d: "country:Mexico" if "country:Mexico" in d.legal_keys else d.legal_keys[0])
        assert t.s.inf(USA, "Iran") == 0 and t.s.inf(USA, "Egypt") == 0

    def test_prevented_by_awacs(self):
        """TS #56: "This Event cannot be used after the '#110 - AWACS Sale to Saudis'
        Event has been played." PD #110: "Muslim Revolution may no longer be played as an
        event." The US ops play then triggers nothing."""
        t = Table(turn=8)
        t.effect("AWACS Sale to Saudis", USA)
        t.inf(usa={"Iran": 3})
        t.play(USA, "Muslim Revolution", "influence", *["country:Mexico"] * 4)
        t.finish_with()
        assert t.s.inf(USA, "Iran") == 3
        assert t.in_cycle("Muslim Revolution")


# --------------------------------------------------------------------------- #
# #57 ABM Treaty
# --------------------------------------------------------------------------- #


class TestC057ABMTreaty:
    def test_improves_then_four_ops(self):
        """PD #57: "Improve DEFCON one level. Then player may Conduct Operations as if
        they played a 4 Ops card.\""""
        t = Table(turn=4, defcon=3)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "ABM Treaty", "event")
        assert t.s.defcon == 4
        answer_use(t, "influence")
        t.answer(*["country:Poland"] * 4)
        assert t.done and t.s.inf(USSR, "Poland") == 5
        assert t.in_cycle("ABM Treaty")

    def test_operations_respect_defcon_after_the_improvement(self):
        """V:B13: "as if they played an N Ops card" operations are not free coups and
        are restricted by DEFCON (R 8.1.5). From DEFCON 2 the event improves to 3: the
        Middle East is open, Europe is not."""
        t = Table(turn=4, defcon=2)
        t.inf(ussr={"Iran": 1, "Iraq": 1, "Poland": 1})
        t.play(USA, "ABM Treaty", "event")
        assert t.s.defcon == 3
        answer_use(t, "coup")
        assert t.d is not None and t.d.type is DecisionType.COUP_TARGET
        assert {"country:Iran", "country:Iraq"} <= set(t.keys())
        assert "country:Poland" not in t.keys()

    def test_event_operations_cannot_go_to_space(self):
        """FAQ §6.4: "players must play a card in order to advance on the space race and
        may not use ops generated by events to do so.\""""
        t = Table(turn=4, defcon=4)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "ABM Treaty", "event")
        assert t.d is not None
        assert "use:space" not in t.keys()

    def test_headlined_it_still_gives_operations(self):
        """FAQ #57: "Operations affiliated with the Event ... are allowed" when it is
        resolved in the headline phase (R 4.5C: "Unless the headline event specifically
        refers to availability of operations points...")."""
        t = Table(turn=4, defcon=4)
        t.inf(ussr={"Poland": 1})
        t.hand(USSR, "ABM Treaty")
        t.hand(USA, "Panama Canal Returned")
        t.s.phase = t.s.phase.HEADLINE
        t.headline_phase()
        t.answer("card:ABM Treaty", "card:Panama Canal Returned")
        assert t.s.defcon == 5
        answer_use(t, "influence")
        assert t.d is not None and t.d.player is USSR
        assert t.d.type is DecisionType.PLACE_INFLUENCE

    # ABM military-ops credit: see test_mech_as_if_ops.py (unverified_rule).


# --------------------------------------------------------------------------- #
# #58 Cultural Revolution
# --------------------------------------------------------------------------- #


class TestC058CulturalRevolution:
    @pytest.mark.parametrize("face_up", [True, False])
    def test_claims_the_china_card_face_up(self, face_up):
        """PD #58: "If the US has 'The China Card', claim it face up and available for
        play." R 9.4: passed by event face up and playable this turn."""
        t = Table(turn=4, china=USA, china_face_up=face_up)
        t.play(USSR, "Cultural Revolution", "event")
        assert t.done
        assert t.s.china_card_owner is USSR and t.s.china_card_face_up
        assert t.s.vp == 0
        assert t.gone_for_good("Cultural Revolution")

    def test_ussr_already_holding_it_scores_one(self):
        """PD #58: "If the USSR already had it, USSR gains 1 VP." Also when face down."""
        t = Table(turn=4, china=USSR, china_face_up=False)
        t.play(USSR, "Cultural Revolution", "event")
        assert t.s.vp == 1 and t.s.china_card_owner is USSR

    def test_us_ops_play_hands_the_china_card_over(self):
        """R 5.2: the US using it for ops triggers the USSR event."""
        t = Table(turn=4, china=USA)
        t.play(USA, "Cultural Revolution", "influence", "yes", *["country:Mexico"] * 3)
        t.finish_with()
        assert t.s.china_card_owner is USSR and t.s.china_card_face_up

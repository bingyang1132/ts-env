"""Instant victory timing, game over, turn-end order and final scoring (R 10.3,
V:B10/B11)."""

from __future__ import annotations

import pytest

from _table import USA, USSR, Table
from twilight.enums import WinReason


class TestNoDecisionsAfterGameOver:
    def test_defcon_1_from_triggered_event_ends_before_operations(self):
        """R 8.1.1 "the game ends immediately" (V:B11). The USSR plays Duck and Cover
        for ops at DEFCON 2 with the event first: DEFCON 1, the USSR loses, and no
        influence-placement decision may follow."""
        t = Table(turn=2, defcon=2)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Duck and Cover", "influence", "yes")
        assert t.s.is_over
        assert t.done, f"decision issued after game over: {t.d and t.d.type}"

    def test_vp_20_from_triggered_event_ends_before_operations(self):
        """R 10.3.1: "The instant one player reaches a score of 20 VP, the game is over".
        VP +19, the US plays Willy Brandt for ops with the event first: USSR +1 = 20 ->
        USSR wins, no further decision."""
        t = Table(turn=5, vp=19)
        t.inf(usa={"Mexico": 1})
        t.play(USA, "Willy Brandt", "influence", "yes")
        assert t.s.is_over and t.s.winner is USSR
        assert t.s.win_reason is WinReason.VICTORY_POINTS
        assert t.done, f"decision issued after game over: {t.d and t.d.type}"

    def test_scoring_card_reaching_20(self):
        """R 10.3.1 applies to scoring cards (after all of the card's VP are applied)."""
        t = Table(turn=5, vp=-17)
        t.inf(usa={"Iraq": 3, "Iran": 2})
        t.play(USA, "Middle East Scoring")
        assert t.s.is_over and t.s.winner is USA and t.done

    def test_step_after_the_end_is_refused(self):
        """Engine contract: once the game is over there is no decision to answer."""
        t = Table(turn=5, vp=-17)
        t.inf(usa={"Iraq": 3, "Iran": 2})
        t.play(USA, "Middle East Scoring")
        with pytest.raises(RuntimeError):
            t.g.step("pass")


class TestTurnEndOrder:
    def test_military_ops_victory_precedes_held_scoring_card_check(self):
        """R 4.5: E (Check Military Operations) comes before F (Reveal Held Card). FAQ
        §8.5: "if a Player achieves an Auto-Victory due to a VP award at that time the
        game is over and Phase F (Reveal Held Card) will not be reached." VP +18, US
        short by 2 at DEFCON 2 -> USSR reaches 20 and wins, even though the USSR holds a
        scoring card (V:B10/B11 item 3)."""
        t = Table(turn=5, vp=18, defcon=2)
        t.s.military_ops = [2, 0]
        t.hand(USSR, "Asia Scoring")
        t.turn_end()
        assert t.s.is_over and t.s.winner is USSR
        assert t.s.win_reason is WinReason.VICTORY_POINTS

    def test_china_vp_at_end_of_turn_10_is_not_an_automatic_victory(self):
        """R 10.3.2: "Reaching 20 VPs does not result in Automatic Victory during scoring
        at the end of turn 10". The China Card's +1 (PD #6 "at the end of Turn 10") is
        part of that scoring (V:B10/B11 item 4). VP +19 with the USSR holding the China
        Card, then the US out-scores the USSR in final scoring -> US wins."""
        t = Table(turn=10, vp=19, china=USSR)
        t.inf(usa={"Japan": 4, "India": 3, "North Korea": 3, "South Korea": 3,
                   "Pakistan": 2, "Thailand": 2, "Iraq": 3, "Iran": 2, "Israel": 4})
        t.rest_of_game()
        t.finish_with()
        assert t.s.is_over and t.s.win_reason is WinReason.FINAL_SCORING
        assert t.s.winner is USA


class TestFinalScoring:
    def test_full_loop_reaches_final_scoring(self):
        """R 4.5 I / 10.3.2: after turn 10 every region is scored and the game ends.
        Empty hands and deck: turn 10 passes, the USSR presence in the Middle East (Iraq)
        scores 3+1 = 4 and wins."""
        t = Table(turn=10, china=None, china_face_up=False)
        t.s.china_card_owner = USA
        t.inf(ussr={"Iraq": 3})
        t.rest_of_game()
        t.finish_with()
        assert t.done and t.s.is_over
        # China +1 to the US at the end of turn 10: 4 - 1 = 3.
        assert t.s.vp == 3 and t.s.winner is USSR

    def test_southeast_asia_is_only_counted_inside_asia(self):
        """R 10.3.2: "Southeast Asia is not scored separately: it is included in the Asia
        scoring calculations." USSR controls only Thailand: Asia presence 3 + 1 BG = 4,
        no extra SE Asia points."""
        t = Table(turn=10)
        t.inf(ussr={"Thailand": 2})
        t.final_scoring()
        assert t.s.vp == 4

    def test_europe_control_wins_regardless(self):
        """R 10.3.2: "Control of Europe does grant automatic victory to the controlling
        player, regardless of scoring elsewhere." VP -15 but the USSR controls Europe."""
        t = Table(turn=10, vp=-15)
        t.inf(ussr={"East Germany": 3, "Poland": 3, "West Germany": 4, "France": 3, "Italy": 2})
        t.final_scoring()
        assert t.s.winner is USSR and t.s.win_reason is WinReason.EUROPE_CONTROL

    def test_zero_is_a_draw(self):
        """R 10.3.2: "If the VP marker is on zero, the game ends in a draw.\""""
        t = Table(turn=10)
        t.final_scoring()
        assert t.s.is_over and t.s.winner is None

    def test_vp_beyond_20_during_final_scoring_is_not_lost(self):
        """R 10.3.2: in final scoring "these new VPs are added to the current score",
        "Reaching 20 VPs does not result in Automatic Victory", and "Every Region's
        score must be calculated before final victory is determined." The result must
        not depend on the order regions are scored in, so intermediate totals may not
        be clipped at +/-20. VP +19; Europe USSR domination +9 (Poland, East Germany,
        Hungary); the US then scores Asia control 16, Middle East presence 4 and Africa
        presence 2 = 22. True total 19 + 9 - 22 = +6 -> USSR wins. Clipping at 20 after
        Europe would give 20 - 22 = -2 -> US."""
        t = Table(turn=10, vp=19)
        t.inf(
            ussr={"Poland": 3, "East Germany": 3, "Hungary": 3},
            usa={"Japan": 4, "India": 3, "North Korea": 3, "South Korea": 3,
                 "Pakistan": 2, "Thailand": 2, "Israel": 4, "Angola": 1},
        )
        t.final_scoring()
        assert t.s.winner is USSR

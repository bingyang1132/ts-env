"""Quagmire (#42) / Bear Trap (#44), the whole action-round flow (V:B3).

PD #42: "On next action round, US player must discard an Operations card worth 2 or more
and roll 1-4 to cancel this event. Repeat each US player Action round until successful or
no appropriate cards remain. If out of appropriate cards, the US player may only play
scoring cards until the next turn." (#44 Bear Trap is the USSR mirror.)
FAQ #42/#44 three steps: "2) If you have a 2 or greater value ops card in your hand you
must discard one of them (it has NO effect); ... otherwise your action round is over.
3) Roll a die. If the number is less than 5 the card's effect is ended, you may resume
normal play on your NEXT action round."
"""

from __future__ import annotations

from _table import USA, USSR, Table
from twilight.decisions import DecisionType
from twilight.enums import WinReason


def trapped_us(**kw) -> Table:
    t = Table(turn=5, **kw)
    t.effect("Quagmire", USA)
    return t


class TestTrapFlow:
    def test_discard_and_escape_on_four(self):
        """FAQ step 2-3: discard a 2+ card (no event), roll 1-4 -> the trap ends; that
        is the whole action round. The discarded US event (Duck and Cover) does not
        fire (R 5.4)."""
        t = trapped_us(defcon=4)
        t.hand(USA, "Duck and Cover", "CIA Created")
        t.dice(4)
        t.action_round(USA)
        assert t.d.type is DecisionType.DISCARD_CARD
        assert set(t.keys()) == {"card:Duck and Cover"}, "only 2+ ops cards qualify"
        t.answer("card:Duck and Cover")
        assert t.done, "the discard and roll are the entire action round"
        assert not t.s.in_play("Quagmire")
        assert t.in_cycle("Duck and Cover") and t.s.defcon == 4 and t.s.vp == 0
        assert t.gone_for_good("Quagmire")
        assert t.s.hands[USA] == ["CIA Created"]

    def test_roll_five_stays_trapped(self):
        """FAQ: "If not, you must do the same on your next action round.\""""
        t = trapped_us()
        t.hand(USA, "Duck and Cover", "Containment")
        t.dice(5)
        t.action_round(USA).answer("card:Duck and Cover")
        assert t.done and t.s.in_play("Quagmire")
        t.dice(6)
        t.action_round(USA)
        assert t.d.type is DecisionType.DISCARD_CARD

    def test_opponent_event_discarded_without_effect(self):
        """FAQ: "A player may discard his opponent's events (without effect – see rule
        5.4)." The US discards Fidel; Cuba is untouched and Fidel is not removed."""
        t = trapped_us()
        t.inf(usa={"Cuba": 1})
        t.hand(USA, "Fidel")
        t.dice(6)
        t.action_round(USA)
        if t.d is not None:
            t.answer("card:Fidel")
        assert t.s.inf(USA, "Cuba") == 1 and t.s.inf(USSR, "Cuba") == 0
        assert t.in_cycle("Fidel")

    def test_china_card_is_not_a_legal_discard(self):
        """R 9.5 / FAQ #6: the China Card may not be discarded to a Quagmire/Bear Trap."""
        t = Table(turn=5, china=USSR)
        t.effect("Bear Trap", USSR)
        t.hand(USSR, "Nasser", "Duck and Cover")
        t.dice(6)
        t.action_round(USSR)
        assert t.d.type is DecisionType.DISCARD_CARD
        assert "card:The China Card" not in t.keys()

    def test_trap_survives_the_turn_end(self):
        """FAQ: "The end of a turn will not end a Beartrap or Quagmire. Only the required
        die roll will do so.\""""
        t = trapped_us()
        t.turn_end()
        assert t.s.in_play("Quagmire")


class TestTrapModifiedOps:
    def test_brezhnev_lets_a_one_op_card_satisfy_bear_trap(self):
        """R 7.4.2 EXAMPLE 2: "If the Soviet player played 'Brezhnev Doctrine' earlier in
        the turn, he could play a 1 Op card to make a required roll under 'Bear Trap'."
        """
        t = Table(turn=5)
        t.effect("Bear Trap", USSR)
        t.effect("Brezhnev Doctrine", USSR, expires="end_of_turn")
        t.hand(USSR, "Nasser")
        t.dice(1)
        t.action_round(USSR)
        if t.d is not None and t.d.type is DecisionType.DISCARD_CARD:
            t.answer("card:Nasser")
        assert "Nasser" in t.s.discard
        assert not t.s.in_play("Bear Trap")

    def test_red_scare_disqualifies_a_two_op_card(self):
        """FAQ §7.4: modified ops apply "for all purposes (Beartrap/Quagmire ...)". Under
        Red Scare a printed-2 US card is worth 1 and cannot be the Quagmire discard."""
        t = trapped_us()
        t.effect("Red Scare/Purge", USSR, expires="end_of_turn")
        t.hand(USA, "Independent Reds", "Duck and Cover")
        t.dice(6)
        t.action_round(USA)
        assert t.d.type is DecisionType.DISCARD_CARD
        assert "card:Independent Reds" not in t.keys()


class TestTrapNoQualifyingCard:
    def test_trap_is_not_lifted(self):
        """PD #42 "Repeat each US player Action round until successful or no appropriate
        cards remain. If out of appropriate cards, the US player may only play scoring
        cards until the next turn." FAQ: "If a player has no cards that meet the Ops
        requirement of the discard, he must hold his cards until the next turn." With
        only a 1-op card the trap stays and the card is not played."""
        t = trapped_us()
        t.inf(usa={"Mexico": 1})
        t.hand(USA, "CIA Created")
        t.action_round(USA)
        t.finish_with()
        assert t.s.in_play("Quagmire"), "the trap persists"
        assert t.s.hands[USA] == ["CIA Created"], "no normal card play while trapped"

    def test_scoring_card_can_still_be_played(self):
        """FAQ: "Scoring cards are the only exception, as they must be played and cannot
        be held." R 10.1.5: "A player may not be forced to Hold a Scoring Card through
        the effects of an Event(s)." The trapped US holding only a scoring card plays it."""
        t = trapped_us()
        t.inf(usa={"Iraq": 3})
        t.hand(USA, "Middle East Scoring")
        t.action_round(USA)
        if t.d is not None:
            assert "card:Middle East Scoring" in t.keys()
            t.answer("card:Middle East Scoring")
        assert "Middle East Scoring" not in t.s.hands[USA]
        assert t.s.vp == -4

    def test_scoring_card_and_one_op_card(self):
        """Same as above with a 1-op card beside the scoring card: the trap stays, the
        scoring card is playable, the 1-op card is not."""
        t = trapped_us()
        t.inf(usa={"Iraq": 3})
        t.hand(USA, "Middle East Scoring", "CIA Created")
        t.action_round(USA)
        assert t.d is not None and t.d.type is DecisionType.PLAY_CARD
        assert set(t.keys()) == {"card:Middle East Scoring"}

    def test_trapped_player_is_not_forced_to_hold_a_scoring_card(self):
        """V:B3 item 4 / FAQ "The requirement that scoring cards must be played has
        precedence over other text": a trapped USSR with a scoring card left at the end
        of the turn must not lose for holding it -- it must have been able to play it.
        Here the USSR's last action round offers it the scoring card."""
        t = Table(turn=5)
        t.effect("Bear Trap", USSR)
        t.inf(ussr={"Iraq": 3})
        t.hand(USSR, "Middle East Scoring", "Duck and Cover")
        t.s.action_round = 7
        t.action_round(USSR)
        assert t.d is not None
        assert "card:Middle East Scoring" in t.keys()


class TestTrapMisc:
    def test_quagmire_cancels_norad(self):
        """PD #106 NORAD: "This event cancelled by 'Quagmire'.\""""
        t = Table(turn=5)
        t.effect("NORAD", USA)
        t.play(USSR, "Quagmire", "event")
        assert not t.s.in_play("NORAD")
        assert t.s.in_play("Quagmire")

    def test_missile_envy_must_be_the_next_discard(self):
        """FAQ #49/#42: "A player under Bear trap/Quagmire who has had Missile Envy played
        against him MUST use it as the next discard if its value has not been degraded by
        Red Scare/Purge." (V:B3 item 5.)"""
        t = trapped_us()
        t.hand(USA, "Missile Envy", "Duck and Cover")
        t.s.must_play[int(USA)] = "Missile Envy"
        t.dice(6)
        t.action_round(USA)
        assert t.d.type is DecisionType.DISCARD_CARD
        assert set(t.keys()) == {"card:Missile Envy"}

    def test_bear_trap_starts_on_the_next_action_round(self):
        """PD #44 "On next action round": playing Bear Trap does not take the current
        action round away; the USSR's next one is the trapped one."""
        t = Table(turn=5)
        t.play(USA, "Bear Trap", "event")
        assert t.done and t.s.in_play("Bear Trap", owner=USSR)
        t.hand(USSR, "Duck and Cover", "Nasser")
        t.dice(6)
        t.action_round(USSR)
        assert t.d.type is DecisionType.DISCARD_CARD

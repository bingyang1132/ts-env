"""Extra card tests for rulings added in verdicts.md v2 (O1-O16, B12b ...), kept apart
from the per-range files. Class names carry the card number like everywhere else."""

from __future__ import annotations

import pytest

from _table import USA, USSR, Table
from twilight.decisions import DecisionType
from twilight.enums import WinReason


class TestC005FiveYearPlanPrerequisite:
    def test_unplayable_us_event_does_not_fire_and_is_not_removed(self):
        """V:B12b + R 5.2: "If a card play triggers an opponent's Event, but that Event
        cannot occur because a prerequisite card has not been played ... cards with an
        asterisk Event (marked *) are returned to the discard pile". The random discard
        is NATO with neither Marshall Plan nor Warsaw Pact played."""
        t = Table(turn=2)
        t.hand(USSR, "NATO")
        t.play(USA, "Five-Year Plan", "event")
        t.finish_with()
        assert not t.s.in_play("NATO")
        assert t.in_cycle("NATO")


class TestC006ChinaCardNeverCompelled:
    def test_only_the_china_card_left_may_pass(self):
        """R 9.8: "Play of 'The China Card' can never be compelled by events or a
        shortage of cards during the action rounds." (V:O11.) With nothing but the China
        Card, the USSR must not be forced to play it."""
        t = Table(turn=2, china=USSR)
        t.inf(ussr={"Poland": 1})
        t.action_round(USSR)
        if t.d is not None:
            assert "pass" in t.keys(), "only the China Card left: the player must be able to pass"
        assert t.s.china_card_owner is USSR, "the China Card was played by compulsion"


class TestC035FormosanDoesNotAffectSummit:
    def test_taiwan_is_not_a_battleground_for_summit(self):
        """PD #35 / C1: "Taiwan is not a battleground country for any other game purpose"
        (V:B2 item 5 / checklist: Formosan only affects scoring). The US controls Japan,
        Taiwan and Australia; with Taiwan as a non-BG it has 1 BG + 2 non-BG ->
        domination; with Taiwan counted as a BG it would still dominate, so use a case
        where it matters: the US controls Taiwan and Australia only (no BG unless Taiwan
        counts) -> no Asia domination bonus in Summit. Rolls US 3, USSR 3: tie."""
        t = Table(turn=5)
        t.effect("Formosan Resolution", USA)
        t.inf(usa={"Taiwan": 3, "Australia": 4})
        t.dice(3, 3)  # USSR roll, US roll
        t.play(USSR, "Summit", "event")
        t.finish_with()
        assert t.s.vp == 0, "tie: Taiwan must not give the US an Asia domination bonus"


class TestC049MissileEnvyV2:
    def test_opponent_event_card_used_for_ops_is_a_normal_play(self):
        """V:O4 (2): "use Operations value without triggering event" spends a real
        card's ops, so a coup with it counts toward military operations (R 6.3.3). The US
        plays Missile Envy and receives Fidel (USSR event, 2 ops): coup Zimbabwe ->
        2 US military ops."""
        t = Table(turn=5)
        t.inf(ussr={"Zimbabwe": 1})
        t.hand(USSR, "Fidel")
        t.dice(1)
        t.play(USA, "Missile Envy", "event")
        if t.d is not None and t.d.type is DecisionType.CARD_USE:
            t.answer("use:coup")
        t.finish_with()
        assert t.s.military_ops[USA] == 2

    def test_scoring_card_takes_precedence_over_the_compulsion(self):
        """V:O4 (3) / FAQ (RULES QUESTIONS): "if a player has two cards left and one of
        them is a scoring card, when his opponent plays Missile Envy and takes the other
        card, then the player would still have to play the scoring card and hold Missile
        Envy." Last US action round, US holds Missile Envy (compelled) and a scoring
        card: the scoring card must be playable."""
        t = Table(turn=5, action_round=7)
        t.inf(usa={"Iraq": 3})
        t.hand(USA, "Missile Envy", "Middle East Scoring")
        t.s.must_play[int(USA)] = "Missile Envy"
        t.action_round(USA)
        assert t.d is not None and "card:Middle East Scoring" in t.keys()


class TestC059FlowerPowerBannedWarCard:
    def test_prohibited_war_event_does_not_score(self):
        """FAQ #59: war cards are subject to the penalty "unless made unplayable by a
        subsequent card" (V:O14). After Camp David the US playing Arab-Israeli War for
        ops gives the USSR nothing."""
        t = Table(turn=5)
        t.effect("Flower Power", USSR)
        t.effect("Camp David Accords", USA)
        t.play(USA, "Arab-Israeli War", "influence", "country:Mexico", "country:Mexico")
        t.finish_with()
        assert t.s.vp == 0


class TestC047JuntaOneRegion:
    def test_realignments_stay_in_one_region(self):
        """FAQ #47: the realignment rolls "may only be made in one of the Regions"
        (V:O15). After a first realignment in Central America (Cuba), the second may not
        be in South America (Brazil)."""
        t = Table(turn=5)
        t.inf(usa={"Cuba": 3, "Brazil": 2})
        t.dice(1, 6, 1, 6)
        t.play(USSR, "Junta", "event", "country:Mexico")
        # choose the realignment option
        realign = next(i for i, a in enumerate(t.d.options) if "realign" in a.label)
        t.answer(f"option:{realign}")
        t.answer("country:Cuba")
        assert t.d is not None
        assert "country:Brazil" not in t.keys()


class TestPlacementSnapshotAtRoundStart:
    def test_event_first_changes_do_not_widen_or_narrow_reach(self):
        """R 6.1.1: markers must be placed "with, or adjacent to, friendly markers that
        were in place at the start of the phasing player's Action Round" (V:O16). The US
        plays Fidel for ops with the event first: Fidel strips the US out of Cuba, but
        Haiti (adjacent to Cuba, where the US had influence at the start of the round)
        must remain a legal placement."""
        t = Table(turn=2)
        t.inf(usa={"Cuba": 1})
        t.play(USA, "Fidel", "influence", "yes")
        assert t.d is not None and t.d.type is DecisionType.PLACE_INFLUENCE
        assert "country:Haiti" in t.keys()


class TestC086NorthSeaOilWithSpaceStation:
    def test_still_eight_rounds(self):
        """V:B7: "若美方本来已有 8 个 AR（Space Station），不再增加" -- PD #86 "US may play 8
        cards this turn". US at box 8 and North Sea Oil headlined: 8 action rounds, not
        9. The US starts with 10 cards: 1 headline + 8 rounds leaves 1."""
        t = Table(turn=8, china_face_up=False)
        t.s.space_race = [0, 8]
        usa = ["North Sea Oil", "Colonial Rear Guards", "Panama Canal Returned",
               "OAS Founded", "Puppet Governments", "Independent Reds", "Truman Doctrine",
               "Containment", "Sadat Expels Soviets", "Alliance for Progress"]
        ussr = ["Decolonization", "Socialist Governments", "Fidel", "Nasser", "Allende",
                "Liberation Theology", "COMECON", "Romanian Abdication"]
        t.hand(USA, *usa)
        t.hand(USSR, *ussr)
        t.inf(ussr={"Poland": 2}, usa={"Mexico": 2})

        def policy(d):
            if d.type is DecisionType.HEADLINE and d.player is USA:
                return "card:North Sea Oil"
            if d.type is DecisionType.CARD_USE and "use:influence" in d.legal_keys:
                return "use:influence"
            if d.type is DecisionType.PLAY_CARD:
                # The 8th round may be forgone (user ruling 2026-11-25); take it.
                return next(k for k in d.legal_keys if k != "pass")
            return "pass" if "pass" in d.legal_keys else d.legal_keys[0]

        t.play_turn().finish_with(policy)
        assert len(t.s.hands[USA]) == 1


class TestC100WargamesMay:
    def test_triggered_wargames_asks_whether_to_end(self):
        """PD #100 "you *may* immediately end the game"; FAQ #100: "The word 'may' is
        meant to communicate options not requirements" (V:O12). Received through Missile
        Envy (neutral event, occurs immediately) at DEFCON 2, the USSR must be asked."""
        t = Table(turn=8, defcon=2)
        t.hand(USA, "Wargames")
        t.play(USSR, "Missile Envy", "event", "card:Wargames")
        assert not t.s.is_over, "the game must not end without the player's choice"
        assert t.d is not None and t.d.player is USSR

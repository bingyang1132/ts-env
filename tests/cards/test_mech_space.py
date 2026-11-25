"""The Space Race (R 6.4) and its four special abilities (V:B4, V:B5)."""

from __future__ import annotations

import pytest

from _table import USA, USSR, Table
from twilight import spacerace
from twilight.decisions import DecisionType
from twilight.enums import WinReason


def attempt(t: Table, side, card: str, roll: int) -> Table:
    t.dice(roll)
    return t.play(side, card, "space")


class TestTrackValues:
    def test_lunar_orbit_pays_three_then_one(self):
        """R 6.4.3: "Five boxes on the Space Race track are marked with two numbers
        divided by a slash, e.g. Lunar Orbit has the numbers 3/1. The left-hand number is
        the number of Victory Points awarded to the first player to reach that box; the
        right-hand number is the number of Victory Points awarded to the second player".
        """
        t = Table(turn=5)
        t.s.space_race = [4, 4]
        attempt(t, USSR, "Duck and Cover", 1)
        assert t.s.space_race[USSR] == 5 and t.s.vp == 3
        attempt(t, USA, "Containment", 1)
        assert t.s.space_race[USA] == 5 and t.s.vp == 2

    def test_five_vp_boxes_and_abilities_on_2_4_6_8(self):
        """R 6.4.3 (five boxes carry VP) and R 6.4.4 (abilities on spaces 2, 4, 6, 8)."""
        vp_boxes = [b.number for b in spacerace.SPACE_TRACK[1:] if b.vp_first]
        assert len(vp_boxes) == 5
        assert {b.number for b in spacerace.SPACE_TRACK if b.ability} == {2, 4, 6, 8}

    @pytest.mark.skip(
        reason="box-by-box ops requirement / roll range / VP are printed on the map; "
        "no text source available here (rulebook R 6.4 only quotes Lunar Orbit 3/1) "
        "-- not verifiable, deliberately not asserted from memory"
    )
    def test_full_track_table(self):
        """Printed Space Race track (map) -- unverified."""


class TestAttempts:
    def test_event_is_not_implemented(self):
        """R 6.4.5: "Regardless of text on the card, an Event discarded to make a die
        roll to advance on the Space Race track is not implemented." The US spaces Fidel:
        Cuba untouched, Fidel discarded (not removed)."""
        t = Table(turn=2)
        t.inf(usa={"Cuba": 1})
        attempt(t, USA, "Fidel", 6)
        assert t.done
        assert t.s.inf(USA, "Cuba") == 1 and t.s.inf(USSR, "Cuba") == 0
        assert t.in_cycle("Fidel")
        assert t.s.space_race[USA] == 0, "6 fails"

    def test_one_attempt_per_turn(self):
        """R 6.4.2: "A player may only play 1 card per turn in an attempt to advance in
        the Space Race.\""""
        t = Table(turn=2)
        attempt(t, USA, "Duck and Cover", 6)
        t.hand(USA, "Containment")
        t.action_round(USA).answer("card:Containment")
        assert "use:space" not in t.keys()

    def test_failed_attempt_still_counts(self):
        """R 6.4.2 counts cards played, so a failed attempt uses up the turn's attempt;
        a new turn restores it."""
        t = Table(turn=2)
        attempt(t, USSR, "Socialist Governments", 6)
        assert t.s.space_attempts[USSR] == 1

    def test_animal_in_space_allows_two(self):
        """R 6.4.4: "Upon reaching space 2 (Animal in Space), the player is allowed to
        play two Space Race cards per turn (instead of the usual one).\""""
        t = Table(turn=2)
        t.s.space_race = [2, 0]
        attempt(t, USSR, "Socialist Governments", 6)
        t.hand(USSR, "Decolonization")
        t.action_round(USSR).answer("card:Decolonization")
        assert "use:space" in t.keys()

    def test_ability_cancelled_when_opponent_arrives(self):
        """R 6.4.4: "The special effect is immediately cancelled when the second player
        reaches that box.\""""
        t = Table(turn=2)
        t.s.space_race = [2, 2]
        attempt(t, USSR, "Socialist Governments", 6)
        t.hand(USSR, "Decolonization")
        t.action_round(USSR).answer("card:Decolonization")
        assert "use:space" not in t.keys()

    def test_end_of_track(self):
        """R 6.4.6: "If a player reaches the final box in the Space Race, no more cards
        may be expended in the Space Race by that player for the remainder of the game.\""""
        t = Table(turn=8)
        t.s.space_race = [8, 0]
        t.inf(ussr={"Poland": 1})
        t.hand(USSR, "Duck and Cover")
        t.action_round(USSR).answer("card:Duck and Cover")
        assert "use:space" not in t.keys()

    def test_red_scare_counts_for_the_ops_requirement(self):
        """FAQ #31: Red Scare's reduction "is for any use of Ops, including Space Race".
        At box 4 (next box Lunar Orbit, which needs more than a 2-op card) a printed
        3-op US card reduced to 2 cannot attempt; unreduced it can."""
        t = Table(turn=5)
        t.s.space_race = [0, 4]
        t.inf(usa={"Mexico": 1})
        t.hand(USA, "Containment")
        t.action_round(USA).answer("card:Containment")
        can_without = "use:space" in t.keys()
        t2 = Table(turn=5)
        t2.s.space_race = [0, 4]
        t2.inf(usa={"Mexico": 1})
        t2.effect("Red Scare/Purge", USSR, expires="end_of_turn")
        t2.hand(USA, "Containment")
        t2.action_round(USA).answer("card:Containment")
        assert can_without and "use:space" not in t2.keys()

    def test_space_vp_reaching_20_ends_the_game_at_once(self):
        """R 6.4.3: "Victory Points granted take effect immediately." R 10.3.1: "The
        instant one player reaches a score of 20 VP, the game is over" (V:B10)."""
        t = Table(turn=5, vp=-19)
        attempt(t, USA, "Containment", 1)
        assert t.s.vp == -20
        assert t.s.is_over and t.s.winner is USA and t.s.win_reason is WinReason.VICTORY_POINTS


class TestEagleBearHasLanded:
    def test_holder_may_discard_held_card_at_turn_end(self):
        """R 6.4.4: "Upon reaching space 6 (Eagle/Bear has Landed), the player may
        discard their Held Card at the end of the turn". FAQ §6.4: "choose and discard
        one card". (V:B5.) The USSR at box 6 holding Fidel is offered the discard."""
        t = Table(turn=5)
        t.s.space_race = [6, 3]
        t.hand(USSR, "Fidel")
        t.turn_end()
        assert t.d is not None and t.d.player is USSR, "a discard choice must be offered"
        assert "card:Fidel" in t.keys() and "pass" in t.keys()

    def test_no_offer_without_the_ability(self):
        """R 6.4.4 cancellation: both at 6 -> nobody holds the ability."""
        t = Table(turn=5)
        t.s.space_race = [6, 6]
        t.hand(USSR, "Fidel")
        t.turn_end()
        assert t.done and t.s.hands[USSR] == ["Fidel"]


class TestSpaceStation:
    def test_eight_action_rounds(self):
        """R 6.4.4: "Upon reaching space 8 (Space Station), the player may play eight (8)
        Action Rounds per turn." Mid War turn (7 rounds): the USSR at box 8 empties a
        9-card hand (1 headline + 8 action rounds); the US plays 1 + 7."""
        t = Table(turn=5, china_face_up=False)  # keep the China Card out of it
        t.s.space_race = [8, 0]
        ussr = ["Decolonization", "Socialist Governments", "Fidel", "Nasser", "Allende",
                "Liberation Theology", "COMECON", "Romanian Abdication",
                "Portuguese Empire Crumbles"]
        usa = ["Colonial Rear Guards", "Panama Canal Returned", "OAS Founded",
               "Puppet Governments", "Independent Reds", "Truman Doctrine",
               "Containment", "Sadat Expels Soviets", "Alliance for Progress"]
        t.hand(USSR, *ussr)
        t.hand(USA, *usa)
        t.inf(ussr={"Poland": 2}, usa={"Mexico": 2})

        def policy(d):
            if d.type is DecisionType.CARD_USE and "use:influence" in d.legal_keys:
                return "use:influence"
            if "pass" in d.legal_keys:
                return "pass"
            return d.legal_keys[0]

        t.play_turn().finish_with(policy)
        assert not t.s.is_over
        assert t.s.hands[USSR] == [], "USSR should have played 9 cards"
        assert len(t.s.hands[USA]) == 1, "US plays 8 (headline + 7)"

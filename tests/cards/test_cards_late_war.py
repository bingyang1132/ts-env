"""Late War cards #82-#102.

Card text quoted from the official digital edition text (``PD``, the Deluxe card faces)
as collected in ``scratchpad/rules/cards.md`` (``TS`` = twilightstrategy.com transcription
from the same file); rule numbers are the Deluxe rulebook (``R``); ``FAQ`` is the
designer-authorised FAQ v5 (2010); ``V:Bn`` points at an entry in ``verdicts.md``.
Expectations follow those sources, not the current implementation.
"""

from __future__ import annotations

import pytest

from _table import USA, USSR, Table
from twilight import rules
from twilight.decisions import DecisionType
from twilight.enums import WinReason


def _use_keys(t: Table) -> tuple[str, ...]:
    assert t.d is not None and t.d.type is DecisionType.CARD_USE, t.d
    return t.keys()


# --------------------------------------------------------------------------- #
# #82 Iranian Hostage Crisis
# --------------------------------------------------------------------------- #


class TestC082IranianHostageCrisis:
    def test_clears_us_and_adds_two(self):
        """PD #82: "Remove all US Influence in Iran. Add 2 USSR Influence in Iran.\""""
        t = Table(turn=8)
        t.inf(usa={"Iran": 3}, ussr={"Iran": 1})
        t.play(USSR, "Iranian Hostage Crisis", "event")
        assert t.done
        assert t.s.inf(USA, "Iran") == 0 and t.s.inf(USSR, "Iran") == 3
        assert t.gone_for_good("Iranian Hostage Crisis")

    def test_triggered_by_us_ops_play(self):
        """R 5.2: the US playing the USSR card for ops still triggers the event, and the
        asterisked card leaves the game."""
        t = Table(turn=8)
        t.inf(usa={"Iran": 2})
        t.play(USA, "Iranian Hostage Crisis", "influence", "yes")
        t.finish_with(lambda d: "country:Mexico" if "country:Mexico" in d.legal_keys else d.legal_keys[0])
        assert t.s.inf(USA, "Iran") == 0 and t.s.inf(USSR, "Iran") == 2
        assert t.gone_for_good("Iranian Hostage Crisis")

    def test_doubles_terrorism_against_the_us(self):
        """PD #82: "Doubles the effect of Terrorism card against US." (#92: "the US
        player must randomly discard two cards.")"""
        t = Table(turn=8)
        t.effect("Iranian Hostage Crisis", USSR)
        t.hand(USA, "Duck and Cover", "Five-Year Plan", "CIA Created")
        t.picks("Duck and Cover", "CIA Created")
        t.play(USSR, "Terrorism", "event")
        assert t.done
        assert t.s.hands[USA] == ["Five-Year Plan"]


# --------------------------------------------------------------------------- #
# #83 The Iron Lady
# --------------------------------------------------------------------------- #


class TestC083TheIronLady:
    def test_basic_effect(self):
        """PD #83: "US gains 1 VP. Add 1 USSR Influence in Argentina. Remove all USSR
        Influence from UK.\""""
        t = Table(turn=8)
        t.inf(ussr={"UK": 3, "Argentina": 1}, usa={"UK": 5})
        t.play(USA, "The Iron Lady", "event")
        assert t.done
        assert t.s.vp == -1
        assert t.s.inf(USSR, "Argentina") == 2 and t.s.inf(USSR, "UK") == 0
        assert t.s.inf(USA, "UK") == 5
        assert t.gone_for_good("The Iron Lady")

    def test_prevents_socialist_governments(self):
        """PD #83: "Socialist Governments event no longer playable." After the Iron Lady
        resolves, the USSR may not choose Socialist Governments' event."""
        t = Table(turn=8)
        t.play(USA, "The Iron Lady", "event")
        t.finish_with()
        t.hand(USSR, "Socialist Governments")
        t.action_round(USSR).answer("card:Socialist Governments")
        assert "use:event" not in _use_keys(t)

    def test_ussr_ops_play_triggers_it(self):
        """R 5.2: USSR plays the US card for ops -> the US event occurs (US +1 VP)."""
        t = Table(turn=8)
        t.inf(ussr={"UK": 1, "Poland": 1})
        t.play(USSR, "The Iron Lady", "influence", "yes")
        t.finish_with(lambda d: "country:Poland" if "country:Poland" in d.legal_keys else d.legal_keys[0])
        assert t.s.inf(USSR, "UK") == 0 and t.s.vp == -1
        assert t.gone_for_good("The Iron Lady")


# --------------------------------------------------------------------------- #
# #84 Reagan Bombs Libya
# --------------------------------------------------------------------------- #


class TestC084ReaganBombsLibya:
    def test_one_vp_per_two_influence_rounded_down(self):
        """PD #84: "US gains 1 VP for every 2 USSR Influence in Libya." 5 -> 2 VP."""
        t = Table(turn=8)
        t.inf(ussr={"Libya": 5})
        t.play(USA, "Reagan Bombs Libya", "event")
        assert t.done and t.s.vp == -2
        assert t.s.inf(USSR, "Libya") == 5, "influence is not removed"
        assert t.gone_for_good("Reagan Bombs Libya")

    def test_one_influence_scores_nothing(self):
        """PD #84 with 1 USSR influence: 0 VP; the event still counts as played (R 5.2
        bullet 4) so the asterisked card is removed."""
        t = Table(turn=8)
        t.inf(ussr={"Libya": 1})
        t.play(USA, "Reagan Bombs Libya", "event")
        assert t.s.vp == 0
        assert t.gone_for_good("Reagan Bombs Libya")


# --------------------------------------------------------------------------- #
# #85 Star Wars
# --------------------------------------------------------------------------- #


class TestC085StarWars:
    def test_takes_non_scoring_card_from_discard_and_event_occurs(self):
        """PD #85: "If the US is ahead on the Space Race Track, play this card to search
        through the discard pile for a non-scoring card of your choice. Event occurs
        immediately." Nuclear Test Ban at DEFCON 4: US +2 VP, DEFCON 5."""
        t = Table(turn=8, defcon=4)
        t.s.space_race = [1, 2]
        t.discard("Asia Scoring", "Nuclear Test Ban", "Duck and Cover")
        t.play(USA, "Star Wars", "event")
        assert t.d is not None and t.d.type is DecisionType.CHOOSE_CARD
        assert "card:Asia Scoring" not in t.keys(), "non-scoring cards only"
        t.answer("card:Nuclear Test Ban")
        assert t.done
        assert t.s.vp == -2 and t.s.defcon == 5
        assert t.in_cycle("Nuclear Test Ban")
        assert t.gone_for_good("Star Wars")

    def test_retrieved_starred_event_is_removed(self):
        """PD #85 + R 4.3: the retrieved event occurs; an asterisked one then leaves the
        game. Truman Doctrine strips Italy."""
        t = Table(turn=8)
        t.s.space_race = [0, 1]
        t.inf(ussr={"Italy": 1})
        t.discard("Truman Doctrine")
        t.play(USA, "Star Wars", "event")
        t.finish_with(lambda d: "card:Truman Doctrine" if "card:Truman Doctrine" in d.legal_keys else d.legal_keys[0])
        assert t.s.inf(USSR, "Italy") == 0
        assert t.gone_for_good("Truman Doctrine")

    def test_not_playable_unless_ahead(self):
        """PD #85: only "If the US is ahead on the Space Race Track". Level -> no event."""
        t = Table(turn=8)
        t.s.space_race = [2, 2]
        t.discard("Nuclear Test Ban")
        t.hand(USA, "Star Wars")
        t.action_round(USA).answer("card:Star Wars")
        assert "use:event" not in _use_keys(t)

    def test_headlined_when_not_ahead_is_discarded_not_removed(self):
        """R 5.2 EXAMPLE 4 logic in the headline: no effect, card returns to the discard
        pile."""
        t = Table(turn=8)
        t.s.space_race = [1, 1]
        t.discard("Nuclear Test Ban")
        t.hand(USA, "Star Wars")
        t.hand(USSR, "Allende")
        t.headline_phase()
        t.answer("card:Allende", "card:Star Wars")
        assert t.done and t.in_cycle("Star Wars")

    def test_ussr_ops_play_when_us_not_ahead_returns_it_to_discard(self):
        """R 5.2 EXAMPLE 4: "The USSR player plays 'Star Wars', but the US player is ahead
        on the Space Race track [sic: behind]. The result is no effect and the 'Star
        Wars' card is returned to the discard pile.\""""
        t = Table(turn=8, defcon=4)
        t.s.space_race = [3, 1]
        t.inf(ussr={"Poland": 1})
        t.discard("Nuclear Test Ban")
        t.play(USSR, "Star Wars", "influence")
        t.finish_with(lambda d: "country:Poland" if "country:Poland" in d.legal_keys else d.legal_keys[0])
        assert t.s.vp == 0 and t.s.defcon == 4
        assert t.in_cycle("Star Wars")


# --------------------------------------------------------------------------- #
# #86 North Sea Oil
# --------------------------------------------------------------------------- #


def _drive_turn(t: Table, *, headlines: dict, wanted: dict, event_cards=()):
    """Answer a whole turn: chosen headlines, then each player plays its first card for
    influence (cards in *event_cards* for their event). Returns cards played per side."""
    played = {USSR: [], USA: []}
    for _ in range(2000):
        d = t.d
        if d is None:
            return played
        keys = d.legal_keys
        if d.type is DecisionType.HEADLINE:
            choice = f"card:{headlines[d.player]}"
        elif d.type is DecisionType.PLAY_CARD:
            pref = [k for k in wanted.get(d.player, []) if f"card:{k}" in keys]
            choice = f"card:{pref[0]}" if pref else keys[0]
            played[d.player].append(choice[5:])
        elif d.type is DecisionType.CARD_USE:
            card = d.context.get("card")
            if card in event_cards and "use:event" in keys:
                choice = "use:event"
            elif "use:influence" in keys:
                choice = "use:influence"
            else:
                choice = keys[0]
        elif d.type is DecisionType.CONFIRM:
            choice = "no"
        elif "pass" in keys:
            choice = "pass"
        else:
            choice = keys[0]
        t.g.step(choice)
    raise AssertionError("turn did not finish")


class TestC086NorthSeaOil:
    def test_opec_no_longer_playable(self):
        """PD #86: "OPEC event is no longer playable.\""""
        t = Table(turn=8)
        t.play(USA, "North Sea Oil", "event")
        t.finish_with()
        assert t.gone_for_good("North Sea Oil")
        t.hand(USSR, "OPEC")
        t.action_round(USSR).answer("card:OPEC")
        assert "use:event" not in _use_keys(t)

    def test_headlined_gives_the_us_an_eighth_action_round(self):
        """PD #86: "US may play 8 cards this turn." TS: "(in 8 action rounds) for this
        turn only". Headlined on turn 8, the US plays 8 cards in action rounds."""
        t = Table(turn=8, china_face_up=False)
        t.inf(usa={"Canada": 1}, ussr={"Poland": 1})
        t.hand(USA, "North Sea Oil")
        t.filler(USA, 8)
        ussr = ["Fidel", "Nasser", "Allende", "Romanian Abdication", "Portuguese Empire Crumbles",
                "Vietnam Revolts", "Korean War", "Blockade", "COMECON"]
        t.hand(USSR, *ussr)
        t.play_turn()
        played = _drive_turn(t, headlines={USA: "North Sea Oil", USSR: "Fidel"}, wanted={})
        assert len(played[USA]) == 8
        assert len(played[USSR]) == 7

    def test_played_in_an_action_round_still_gives_eight(self):
        """PD #86 "US may play 8 cards this turn" (V: B7). Played in the US's first
        action round of turn 8, the US still gets an eighth action round this turn."""
        t = Table(turn=8, china_face_up=False)
        t.inf(usa={"Canada": 1}, ussr={"Poland": 1})
        t.hand(USA, "North Sea Oil")
        t.filler(USA, 8)
        ussr = ["Fidel", "Nasser", "Allende", "Romanian Abdication", "Portuguese Empire Crumbles",
                "Vietnam Revolts", "Korean War", "Blockade", "COMECON"]
        t.hand(USSR, *ussr)
        us_head = t.s.hands[USA][1]
        t.play_turn()
        played = _drive_turn(
            t,
            headlines={USA: us_head, USSR: "Fidel"},
            wanted={USA: ["North Sea Oil"]},
            event_cards=("North Sea Oil",),
        )
        assert played[USA][0] == "North Sea Oil"
        assert len(played[USA]) == 8, f"US played {len(played[USA])} cards in action rounds"

    def test_extra_round_is_this_turn_only(self):
        """TS #86: "for this turn only" -- after the turn the US is back to 7."""
        t = Table(turn=8)
        t.effect("North Sea Oil", USA, extra_action_round_turn=8)
        t.s.turn = 9
        t.inf(usa={"Canada": 1}, ussr={"Poland": 1})
        t.filler(USA, 9)
        ussr = ["Fidel", "Nasser", "Allende", "Romanian Abdication", "Portuguese Empire Crumbles",
                "Vietnam Revolts", "Korean War", "Blockade", "COMECON"]
        t.hand(USSR, *ussr)
        us_head = t.s.hands[USA][0]
        t.s.china_card_face_up = False
        t.play_turn()
        played = _drive_turn(t, headlines={USA: us_head, USSR: "Fidel"}, wanted={})
        assert len(played[USA]) == 7


# --------------------------------------------------------------------------- #
# #87 The Reformer
# --------------------------------------------------------------------------- #


class TestC087TheReformer:
    def test_four_influence_when_not_ahead(self):
        """PD #87: "Add 4 Influence in Europe (no more than 2 per country).\""""
        t = Table(turn=8, vp=0)
        t.play(USSR, "The Reformer", "event", "country:Poland", "country:Poland")
        assert "country:Poland" not in t.keys(), "max 2 per country"
        assert "country:Iraq" not in t.keys(), "Europe only"
        t.answer("country:Italy", "country:France")
        assert t.done
        assert t.s.inf(USSR, "Poland") == 2 and t.s.inf(USSR, "Italy") == 1
        assert t.gone_for_good("The Reformer")

    def test_six_when_ussr_ahead(self):
        """PD #87: "If USSR is ahead of US in VP, then 6 Influence may be added
        instead.\""""
        t = Table(turn=8, vp=3)
        t.play(USSR, "The Reformer", "event")
        # User ruling 2026-11-25: when ahead the USSR chooses 6 (option:0) or 4.
        t.answer("option:0")
        for n in ("Poland", "Poland", "Italy", "Italy", "France", "France"):
            t.answer(f"country:{n}")
        assert t.done
        assert sum(t.s.inf(USSR, n) for n in ("Poland", "Italy", "France")) == 6

    def test_ussr_may_no_longer_coup_in_europe(self):
        """PD #87: "USSR may no longer conduct Coup attempts in Europe." Realignments
        and the US are unaffected (DEFCON 5)."""
        t = Table(turn=8, defcon=5)
        t.inf(usa={"Italy": 1, "Iran": 1})
        t.play(USSR, "The Reformer", "event")
        t.finish_with()
        assert not rules.can_coup(t.s, USSR, "Italy")
        assert rules.can_realign(t.s, USSR, "Italy")
        assert rules.can_coup(t.s, USSR, "Iran")


# --------------------------------------------------------------------------- #
# #88 Marine Barracks Bombing
# --------------------------------------------------------------------------- #


class TestC088MarineBarracksBombing:
    def test_lebanon_plus_two_more(self):
        """PD #88: "Remove all US Influence in Lebanon plus remove 2 additional US
        Influence from anywhere in the Middle East.\""""
        t = Table(turn=8)
        t.inf(usa={"Lebanon": 3, "Israel": 4, "Iran": 1, "Italy": 2})
        t.play(USSR, "Marine Barracks Bombing", "event")
        assert "country:Italy" not in t.keys()
        assert "country:Lebanon" not in t.keys(), "Lebanon is already cleared"
        t.answer("country:Israel", "country:Iran")
        assert t.done
        assert t.s.inf(USA, "Lebanon") == 0 and t.s.inf(USA, "Israel") == 3
        assert t.s.inf(USA, "Iran") == 0 and t.s.inf(USA, "Italy") == 2
        assert t.gone_for_good("Marine Barracks Bombing")


# --------------------------------------------------------------------------- #
# #89 Soviets Shoot Down KAL-007
# --------------------------------------------------------------------------- #


class TestC089KAL007:
    def test_degrades_and_scores(self):
        """PD #89: "Degrade DEFCON one level. US gains 2 VP.\""""
        t = Table(turn=8, defcon=4)
        t.play(USA, "Soviets Shoot Down KAL-007", "event")
        assert t.done
        assert t.s.defcon == 3 and t.s.vp == -2
        assert t.gone_for_good("Soviets Shoot Down KAL-007")

    def test_with_south_korea_influence_or_realign_but_no_coup(self):
        """PD #89: "If South Korea is US Controlled, then the US may place Influence or
        attempt Realignment as if they played a 4 Ops card." No coup (V:B8 table)."""
        t = Table(turn=8, defcon=5)
        t.inf(usa={"South Korea": 3}, ussr={"North Korea": 3, "Angola": 1})
        t.play(USA, "Soviets Shoot Down KAL-007", "event")
        keys = _use_keys(t)
        assert "use:influence" in keys and "use:realign" in keys
        assert "use:coup" not in keys
        t.answer("use:influence")
        for _ in range(4):
            t.answer("country:South Korea")
        assert t.done and t.s.inf(USA, "South Korea") == 7

    def test_event_ops_reduced_by_red_scare(self):
        """Coordinator ruling for 'as if they played a 4 Ops card' (V:B8, Deluxe 7.4.3
        unavailable; FAQ #31 says the Deluxe rewording makes such ops subject to
        modifiers): treated as a normal 4-op play, so Red Scare/Purge against the US
        leaves 3 placements. (Was: FAQ #31 2nd-ed "Ops from events are not affected".)"""
        t = Table(turn=8, defcon=5)
        t.effect("Red Scare/Purge", USSR, expires="end_of_turn")
        t.inf(usa={"South Korea": 3})
        t.play(USA, "Soviets Shoot Down KAL-007", "event")
        if t.d is not None and t.d.type is DecisionType.CARD_USE:
            t.answer("use:influence")
        n = 0
        while t.d is not None and t.d.type is DecisionType.PLACE_INFLUENCE:
            t.answer("country:South Korea")
            n += 1
        assert n == 3

    def test_at_defcon_2_the_us_loses(self):
        """R 8.1.3: US plays it at DEFCON 2 -> DEFCON 1 -> US (phasing) loses."""
        t = Table(turn=8, defcon=2)
        t.play(USA, "Soviets Shoot Down KAL-007", "event")
        assert t.s.is_over and t.s.winner is USSR and t.s.win_reason is WinReason.DEFCON

    def test_ussr_ops_play_at_defcon_2_ussr_loses(self):
        """R 5.2 + 8.1.3: the USSR playing it for ops triggers the US event; DEFCON 1 in
        the USSR action round -> the USSR loses."""
        t = Table(turn=8, defcon=2)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Soviets Shoot Down KAL-007", "influence", "yes")
        assert t.s.is_over and t.s.winner is USA and t.s.win_reason is WinReason.DEFCON


# --------------------------------------------------------------------------- #
# #90 Glasnost
# --------------------------------------------------------------------------- #


class TestC090Glasnost:
    def test_basic(self):
        """PD #90: "USSR gains 2 VP. Improve DEFCON one level.\" Without The Reformer
        nothing more."""
        t = Table(turn=8, defcon=3)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Glasnost", "event")
        assert t.done
        assert t.s.vp == 2 and t.s.defcon == 4
        assert t.gone_for_good("Glasnost")

    def test_with_reformer_influence_or_realign_not_coup(self):
        """PD #90: "If The Reformer is in effect, then the USSR may place Influence or
        attempt Realignments as if they played a 4 Ops card.\""""
        t = Table(turn=8, defcon=5)
        t.effect("The Reformer", USSR)
        t.inf(ussr={"Iraq": 1}, usa={"Angola": 1, "Iraq": 1})
        t.play(USSR, "Glasnost", "event")
        keys = _use_keys(t)
        assert "use:influence" in keys and "use:realign" in keys
        assert "use:coup" not in keys

    def test_glasnost_placement_follows_ops_adjacency(self):
        """FAQ #90: the Glasnost placement "is subject to the rules for placing Influence
        in Rule 6.0" -- R 6.1.1 adjacency: no Soviet presence near Zaire, so Zaire is not
        offered."""
        t = Table(turn=8, defcon=5)
        t.effect("The Reformer", USSR)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Glasnost", "event")
        if t.d is not None and t.d.type is DecisionType.CARD_USE:
            t.answer("use:influence")
        assert "country:Zaire" not in t.keys() and "country:Poland" in t.keys()

    def test_glasnost_placement_respects_chernobyl(self):
        """FAQ #90: Glasnost's placement "is subject to the rules for placing Influence ...
        and therefore may not used to place Influence in the Region designated by #94
        Chernobyl." With Europe named, no European country is offered."""
        t = Table(turn=8, defcon=5)
        t.effect("The Reformer", USSR)
        t.effect("Chernobyl", USA, expires="end_of_turn", region="Europe")
        t.inf(ussr={"Poland": 2, "Iraq": 1})
        t.play(USSR, "Glasnost", "event")
        if t.d is not None and t.d.type is DecisionType.CARD_USE:
            t.answer("use:influence")
        assert t.d is not None and t.d.type is DecisionType.PLACE_INFLUENCE
        assert "country:Poland" not in t.keys() and "country:Iraq" in t.keys()


# --------------------------------------------------------------------------- #
# #91 Ortega Elected in Nicaragua
# --------------------------------------------------------------------------- #


class TestC091Ortega:
    def test_clears_nicaragua_and_coup_adjacent(self):
        """PD #91: "Remove all US Influence from Nicaragua. Then USSR may make one free
        Coup attempt (with this card's Operations value) in a country adjacent to
        Nicaragua." Cuba (stab 3, BG): 6+2-6 = 2 removed; BG coup degrades DEFCON (R
        6.3.5); free coup does not count military ops (R 8.2.5)."""
        t = Table(turn=8, defcon=5)
        t.inf(usa={"Nicaragua": 2, "Cuba": 2, "Honduras": 1, "Mexico": 2})
        t.dice(6)
        t.play(USSR, "Ortega Elected in Nicaragua", "event")
        assert t.s.inf(USA, "Nicaragua") == 0
        assert set(t.keys()) == {"country:Cuba", "country:Honduras", "pass"}
        t.answer("country:Cuba")
        assert t.done
        assert t.s.inf(USA, "Cuba") == 0
        assert t.s.defcon == 4
        assert t.s.military_ops[USSR] == 0
        assert t.gone_for_good("Ortega Elected in Nicaragua")

    def test_coup_is_optional(self):
        """PD #91: "may make one free Coup attempt".\""""
        t = Table(turn=8)
        t.inf(usa={"Cuba": 2})
        t.play(USSR, "Ortega Elected in Nicaragua", "event", "pass")
        assert t.done and t.s.inf(USA, "Cuba") == 2

    def test_coup_under_us_cuban_missile_crisis_loses_for_ussr(self):
        """FAQ #40: "If a player is operating under Cuban Missile Crisis, the second he
        chose to initiate a coup, whether or not brought about by an opponent's card
        play, he loses." US Cuban Missile Crisis in effect, USSR takes Ortega's coup."""
        t = Table(turn=8, defcon=4)
        t.effect("Cuban Missile Crisis", USA, expires="end_of_turn")
        t.inf(usa={"Honduras": 1})
        t.play(USSR, "Ortega Elected in Nicaragua", "event")
        # USSR has no Cuba influence, so the cancel offer is not made.
        t.answer("country:Honduras")
        assert t.s.is_over and t.s.winner is USA

    def test_triggered_by_us_ops_free_coup_to_defcon_1_loses_for_us(self):
        """V:B9 table: "美用 ops 打苏联 Ortega ... 苏联免费政变战场国使 DEFCON->1 -> 美国
        (phasing)". R 8.1.3: the phasing player is responsible."""
        t = Table(turn=8, defcon=2)
        t.inf(usa={"Cuba": 2, "Canada": 1})
        t.dice(6)
        t.play(USA, "Ortega Elected in Nicaragua", "influence", "yes")
        assert t.d is not None and t.d.player is USSR
        t.answer("country:Cuba")
        assert t.s.is_over and t.s.win_reason is WinReason.DEFCON
        assert t.s.winner is USSR


# --------------------------------------------------------------------------- #
# #92 Terrorism
# --------------------------------------------------------------------------- #


class TestC092Terrorism:
    def test_opponent_discards_one_at_random_without_event(self):
        """PD #92: "Opponent must randomly discard one card. ... (Events on discards do
        not occur.)\""""
        t = Table(turn=8, defcon=4)
        t.hand(USA, "Duck and Cover", "CIA Created")
        t.picks("Duck and Cover")
        t.play(USSR, "Terrorism", "event")
        assert t.done
        assert t.s.hands[USA] == ["CIA Created"]
        assert t.in_cycle("Duck and Cover") and t.s.defcon == 4 and t.s.vp == 0
        assert t.in_cycle("Terrorism"), "not starred"

    def test_doubling_only_against_the_us(self):
        """PD #92: "If played by USSR and Iranian Hostage Crisis is in effect, the US
        player must randomly discard two cards." Played by the US: the USSR discards
        only one."""
        t = Table(turn=8)
        t.effect("Iranian Hostage Crisis", USSR)
        t.hand(USSR, "Fidel", "Nasser", "Allende")
        t.play(USA, "Terrorism", "event")
        assert len(t.s.hands[USSR]) == 2

    def test_china_card_is_never_discarded(self):
        """R 9.5: the China Card cannot be a discard required by an event."""
        t = Table(turn=8, china=USA)
        t.play(USSR, "Terrorism", "event")
        assert t.done and t.s.china_card_owner is USA

    def test_us_ops_play_of_neutral_card_does_not_fire(self):
        """R 5.2 applies only to the opponent's event: Terrorism is neutral, so the US
        playing it for ops makes nobody discard."""
        t = Table(turn=8)
        t.hand(USA, "Duck and Cover")
        t.hand(USSR, "Fidel")
        t.play(USA, "Terrorism", "influence")
        t.finish_with(lambda d: "country:Mexico" if "country:Mexico" in d.legal_keys else d.legal_keys[0])
        assert t.s.hands[USA] == ["Duck and Cover"] and t.s.hands[USSR] == ["Fidel"]


# --------------------------------------------------------------------------- #
# #93 Iran-Contra Scandal
# --------------------------------------------------------------------------- #


class TestC092TerrorismExtra:
    def test_discarded_scoring_card_is_not_scored(self):
        """R 5.4: "When an event forces a player to discard a card, the Event on the
        discarded card is not implemented. This rule also applies to Scoring cards.\""""
        t = Table(turn=8)
        t.inf(usa={"Japan": 4, "South Korea": 3})
        t.hand(USA, "Asia Scoring")
        t.play(USSR, "Terrorism", "event")
        assert t.done and t.s.vp == 0 and t.in_cycle("Asia Scoring")


class TestC093IranContra:
    def test_us_realignment_rolls_minus_one(self):
        """PD #93: "All US Realignment rolls have a -1 die roll modifier for the remainder
        of the turn." Equal rolls 3 vs 3 in Angola (no other modifiers): the US total 2
        loses by 1."""
        t = Table(turn=8)
        t.play(USSR, "Iran-Contra Scandal", "event")
        t.inf(ussr={"Angola": 1}, usa={"Angola": 1})
        res = rules.resolve_realignment(t.s, USA, "Angola", rolls=(3, 3))
        assert res.totals == (3, 2)
        assert res.winner is USSR

    def test_ussr_rolls_unaffected_and_expires_at_turn_end(self):
        """PD #93: only US rolls; "for the remainder of the turn"."""
        t = Table(turn=8)
        t.effect("Iran-Contra Scandal", USSR, expires="end_of_turn")
        t.inf(ussr={"Angola": 1}, usa={"Angola": 1})
        res = rules.resolve_realignment(t.s, USSR, "Angola", rolls=(3, 3))
        assert res.totals[USSR] == 3
        t.turn_end()
        assert not t.s.in_play("Iran-Contra Scandal")
        assert t.gone_for_good("Iran-Contra Scandal")


# --------------------------------------------------------------------------- #
# #94 Chernobyl
# --------------------------------------------------------------------------- #


class TestC094Chernobyl:
    def _chernobyl_europe(self) -> Table:
        t = Table(turn=8, defcon=5)
        t.inf(ussr={"Poland": 2, "Iraq": 1}, usa={"Italy": 1})
        t.play(USA, "Chernobyl", "event")
        assert t.d is not None and t.d.type is DecisionType.CHOOSE_REGION
        assert t.d.player is USA
        t.answer("region:Europe")
        assert t.done
        return t

    def test_ussr_cannot_place_ops_influence_in_named_region(self):
        """PD #94: "For the remainder of the turn the USSR may not add additional
        Influence to that Region by the play of Operations Points via placing
        Influence.\""""
        t = self._chernobyl_europe()
        t.play(USSR, "Decolonization", "influence")
        assert t.d is not None and t.d.type is DecisionType.PLACE_INFLUENCE
        assert "country:Poland" not in t.keys() and "country:Finland" not in t.keys()
        assert "country:Iraq" in t.keys()

    def test_coups_realignments_and_event_placement_still_allowed(self):
        """FAQ #94: "Realignments do not add influence, so the USSR player could still
        make a Realignment roll. Coups are also allowed ... Further, the Soviet player
        could also add Influence in the selected region through the play of events.\""""
        t = self._chernobyl_europe()
        assert rules.can_coup(t.s, USSR, "Italy")
        assert rules.can_realign(t.s, USSR, "Italy")
        t.play(USSR, "COMECON", "event")
        assert t.d is not None and "country:Hungary" in t.keys()

    def test_naming_asia_includes_southeast_asia(self):
        """PD #94 names a Region; Southeast Asia is part of Asia (R 9.6 "Asia (including
        Southeast Asia)"), so naming Asia shuts Thailand too."""
        t = Table(turn=8)
        t.inf(ussr={"Thailand": 1, "Iraq": 1})
        t.play(USA, "Chernobyl", "event", "region:Asia")
        t.play(USSR, "Decolonization", "influence")
        assert "country:Thailand" not in t.keys() and "country:Iraq" in t.keys()

    def test_us_unaffected_and_lasts_the_turn(self):
        """PD #94: restricts only the USSR, only for the turn; starred."""
        t = self._chernobyl_europe()
        assert "Italy" in rules.placeable_countries(t.s, USA)
        t.turn_end()
        assert not t.s.in_play("Chernobyl")
        assert t.gone_for_good("Chernobyl")
        assert "Poland" in rules.placeable_countries(t.s, USSR)


# --------------------------------------------------------------------------- #
# #95 Latin American Debt Crisis
# --------------------------------------------------------------------------- #


class TestC095LatinAmericanDebtCrisis:
    def test_us_discards_a_three_op_card(self):
        """PD #95: "Unless the US Player immediately discards a '3' or greater Operations
        card, double USSR Influence in two countries in South America.\""""
        t = Table(turn=8)
        t.inf(ussr={"Argentina": 2})
        t.hand(USA, "Duck and Cover", "CIA Created")
        t.play(USSR, "Latin American Debt Crisis", "event")
        assert set(t.keys()) == {"card:Duck and Cover", "pass"}
        t.answer("card:Duck and Cover")
        assert t.done
        assert t.s.inf(USSR, "Argentina") == 2 and t.in_cycle("Duck and Cover")
        assert t.in_cycle("Latin American Debt Crisis"), "not starred"

    def test_otherwise_doubles_in_two_south_american_countries(self):
        """PD #95 as above: Argentina 2 -> 4, Chile 3 -> 6; Mexico (Central America)
        is not eligible."""
        t = Table(turn=8)
        t.inf(ussr={"Argentina": 2, "Chile": 3, "Peru": 1, "Mexico": 2})
        t.hand(USA, "CIA Created")
        t.play(USSR, "Latin American Debt Crisis", "event")
        assert "country:Mexico" not in t.keys()
        t.answer("country:Argentina")
        assert "country:Argentina" not in t.keys(), "two different countries"
        t.answer("country:Chile")
        assert t.done
        assert t.s.inf(USSR, "Argentina") == 4 and t.s.inf(USSR, "Chile") == 6
        assert t.s.inf(USSR, "Peru") == 1

    def test_red_scare_reduced_card_does_not_qualify(self):
        """R 7.4.2: ops modifiers apply "for all purposes"; under Red Scare/Purge a
        printed 3-op US card is worth 2 and cannot satisfy the '3 or greater' discard."""
        t = Table(turn=8)
        t.effect("Red Scare/Purge", USSR, expires="end_of_turn")
        t.inf(ussr={"Argentina": 2})
        t.hand(USA, "Duck and Cover")
        t.play(USSR, "Latin American Debt Crisis", "event")
        assert t.done and t.s.inf(USSR, "Argentina") == 4
        assert t.s.hands[USA] == ["Duck and Cover"]

    def test_china_card_not_an_acceptable_discard(self):
        """R 9.5: the China Card may not be a discard required by an event."""
        t = Table(turn=8, china=USA)
        t.inf(ussr={"Argentina": 2})
        t.play(USSR, "Latin American Debt Crisis", "event")
        assert t.done and t.s.inf(USSR, "Argentina") == 4
        assert t.s.china_card_owner is USA


# --------------------------------------------------------------------------- #
# #96 Tear Down This Wall
# --------------------------------------------------------------------------- #


class TestC096TearDownThisWall:
    def test_east_germany_and_free_coup_in_europe(self):
        """PD #96: "Add 3 US Influence in East Germany. Then US may make a free Coup
        attempt or Realignment rolls in Europe using this card's Ops Value." Coup in
        Poland (BG, stab 3): 6+3-6 = 3. A BG coup still degrades DEFCON (R 6.3.5); free
        coups do not count military ops (R 8.2.5, FAQ #96)."""
        t = Table(turn=8, defcon=5)
        t.inf(ussr={"Poland": 2, "East Germany": 3})
        t.dice(6)
        t.play(USA, "Tear Down This Wall", "event")
        assert t.s.inf(USA, "East Germany") == 3
        assert t.d is not None and t.d.type is DecisionType.CHOOSE_OPTION
        coup = next(a.key for a in t.d.options if "coup" in a.label)
        t.answer(coup, "country:Poland")
        assert t.done
        assert t.s.inf(USSR, "Poland") == 0 and t.s.inf(USA, "Poland") == 1
        assert t.s.defcon == 4
        assert t.s.military_ops[USA] == 0
        assert t.gone_for_good("Tear Down This Wall")

    def test_free_coup_ignores_defcon_geography(self):
        """V:B13 / R 6.3.5 / FAQ #96: "If the card states that the player may coup or
        realign in a particular region, he may do so regardless of the DEFCON status."
        At DEFCON 3 the US may still coup in Europe."""
        t = Table(turn=8, defcon=3)
        t.inf(ussr={"Italy": 1})
        t.play(USA, "Tear Down This Wall", "event")
        assert t.d is not None and t.d.type is DecisionType.CHOOSE_OPTION
        assert any("coup" in a.label for a in t.d.options)

    def test_cancels_willy_brandt_and_nato_protects_west_germany_again(self):
        """PD #96 "Cancels/prevent Willy Brandt"; PD #55 Willy Brandt "Cancels NATO for
        West Germany" (V:B12). Once Willy Brandt is cancelled, NATO again bars USSR
        coups in US-controlled West Germany."""
        t = Table(turn=8, defcon=5)
        t.effect("NATO", USA)
        t.resolved("Marshall Plan")
        t.effect("Willy Brandt", USSR)
        t.inf(usa={"West Germany": 6}, ussr={"West Germany": 1})
        assert rules.can_coup(t.s, USSR, "West Germany")
        t.play(USA, "Tear Down This Wall", "event")
        t.finish_with()
        assert not t.s.in_play("Willy Brandt")
        assert not rules.can_coup(t.s, USSR, "West Germany")

    def test_prevents_willy_brandt(self):
        """PD #96 "prevent Willy Brandt"; PD #55 "This event canceled/unplayable by #96"."""
        t = Table(turn=8)
        t.resolved("Tear Down This Wall")
        t.inf(ussr={"Poland": 1})
        t.hand(USSR, "Willy Brandt")
        t.action_round(USSR).answer("card:Willy Brandt")
        assert "use:event" not in _use_keys(t)


# --------------------------------------------------------------------------- #
# #97 An Evil Empire
# --------------------------------------------------------------------------- #


class TestC097AnEvilEmpire:
    def test_cancels_flower_power(self):
        """PD #97: "Cancels/Prevents Flower Power. US gains 1 VP." After it, a US war card
        play no longer pays the USSR (PD #59)."""
        t = Table(turn=8)
        t.effect("Flower Power", USSR)
        t.play(USA, "An Evil Empire", "event")
        assert t.done and t.s.vp == -1
        assert not t.s.in_play("Flower Power")
        assert t.gone_for_good("Flower Power")
        assert t.gone_for_good("An Evil Empire")
        t.dice(1)
        t.play(USA, "Brush War", "event", "country:Turkey")
        assert t.s.vp == -1

    def test_prevents_flower_power(self):
        """PD #97 "Prevents Flower Power": the USSR may not play its event afterwards."""
        t = Table(turn=8)
        t.resolved("An Evil Empire")
        t.inf(ussr={"Poland": 1})
        t.hand(USSR, "Flower Power")
        t.action_round(USSR).answer("card:Flower Power")
        assert "use:event" not in _use_keys(t)


# --------------------------------------------------------------------------- #
# #98 Aldrich Ames Remix
# --------------------------------------------------------------------------- #


class TestC098AldrichAmesRemix:
    def test_ussr_picks_a_us_card_to_discard(self):
        """PD #98: "USSR then chooses one card from US hand; this card is discarded." R
        5.4: no event; R 9.5: never the China Card."""
        t = Table(turn=8, defcon=4, china=USA)
        t.hand(USA, "Duck and Cover", "Fidel")
        t.play(USSR, "Aldrich Ames Remix", "event")
        assert t.d is not None and t.d.player is USSR
        assert set(t.keys()) == {"card:Duck and Cover", "card:Fidel"}
        t.answer("card:Duck and Cover")
        assert t.done
        assert t.s.hands[USA] == ["Fidel"] and t.in_cycle("Duck and Cover")
        assert t.s.defcon == 4 and t.s.china_card_owner is USA
        assert t.gone_for_good("Aldrich Ames Remix")

    def test_exposure_lasts_for_the_turn(self):
        """PD #98: "exposes his hand ... for remainder of turn" -- the effect ends at the
        turn end and the card is out of the game."""
        t = Table(turn=8)
        t.hand(USA, "Fidel")
        t.play(USSR, "Aldrich Ames Remix", "event")
        t.finish_with()
        t.turn_end()
        assert not t.s.in_play("Aldrich Ames Remix")
        assert t.gone_for_good("Aldrich Ames Remix")


# --------------------------------------------------------------------------- #
# #99 Pershing II Deployed
# --------------------------------------------------------------------------- #


class TestC099PershingII:
    def test_vp_and_one_from_each_of_three_western_european(self):
        """PD #99: "USSR gains 1 VP. Remove 1 US Influence from up to three countries in
        Western Europe.\""""
        t = Table(turn=8)
        t.inf(usa={"UK": 5, "France": 3, "Italy": 2, "Poland": 1})
        t.play(USSR, "Pershing II Deployed", "event", "country:UK")
        assert "country:UK" not in t.keys(), "1 per country"
        assert "country:Poland" not in t.keys(), "Western Europe only"
        t.answer("country:France")  # Italy is then the only target left
        assert t.done
        assert (t.s.inf(USA, "UK"), t.s.inf(USA, "France"), t.s.inf(USA, "Italy")) == (4, 2, 1)
        assert t.s.inf(USA, "Poland") == 1
        assert t.s.vp == 1
        assert t.gone_for_good("Pershing II Deployed")

    def test_us_ops_play_triggers(self):
        """R 5.2: the US playing it for ops triggers the USSR event (USSR +1)."""
        t = Table(turn=8)
        t.inf(usa={"Italy": 2})
        t.play(USA, "Pershing II Deployed", "influence", "yes")
        t.finish_with(lambda d: "country:Mexico" if "country:Mexico" in d.legal_keys else d.legal_keys[0])
        assert t.s.vp == 1 and t.s.inf(USA, "Italy") == 1


# --------------------------------------------------------------------------- #
# #100 Wargames
# --------------------------------------------------------------------------- #


class TestC100Wargames:
    def test_at_defcon_2_ends_the_game_on_vp(self):
        """PD #100: "If DEFCON Status 2, you may immediately end the game (without Final
        Scoring Phase) after giving opponent 6 VPs." USSR at +3 plays it: US +6 -> -3,
        the US wins, even though the USSR controls Europe (no final scoring)."""
        t = Table(turn=8, defcon=2, vp=3)
        for n in ("Poland", "East Germany", "France", "Italy", "West Germany"):
            t.inf(ussr={n: 6})
        t.play(USSR, "Wargames", "event")
        # User ruling 2026-11-25: the "may" is a choice even when played as an event
        # (give 6 VP and end / give and go on / neither); option:1 = give and end.
        t.answer("option:1")
        assert t.s.is_over and t.s.win_reason is WinReason.WARGAMES
        assert t.s.vp == -3 and t.s.winner is USA
        assert t.done

    def test_not_playable_above_defcon_2(self):
        """PD #100 "If DEFCON Status 2"; FAQ #100 "Wargames is only considered used when
        DEFCON is at 2"."""
        t = Table(turn=8, defcon=3)
        t.inf(ussr={"Poland": 1})
        t.hand(USSR, "Wargames")
        t.action_round(USSR).answer("card:Wargames")
        assert "use:event" not in _use_keys(t)

    def test_headlined_above_defcon_2_is_not_removed(self):
        """FAQ #100: "Can Wargames be played as an event for no effect at DEFCON 3-5 to
        remove it from the deck? ... it would not be removed if DEFCON was at 3-5.\""""
        t = Table(turn=8, defcon=3)
        t.hand(USSR, "Wargames")
        t.hand(USA, "Panama Canal Returned")
        t.headline_phase()
        t.answer("card:Wargames", "card:Panama Canal Returned")
        t.finish_with()
        assert not t.s.is_over
        assert t.in_cycle("Wargames")


# --------------------------------------------------------------------------- #
# #101 Solidarity
# --------------------------------------------------------------------------- #


class TestC101Solidarity:
    def test_headlined_without_prerequisite_is_discarded_not_removed(self):
        """R 5.2 bullet 2 / R 4.5C: an event that cannot occur for want of a prerequisite
        card is not removed; the asterisked card returns to the discard pile."""
        t = Table(turn=8)
        t.hand(USA, "Solidarity")
        t.hand(USSR, "Allende")
        t.headline_phase()
        t.answer("card:Allende", "card:Solidarity")
        assert t.done and t.s.inf(USA, "Poland") == 0
        assert t.in_cycle("Solidarity")

    def test_requires_john_paul_ii(self):
        """PD #101: "Playable as an event only if John Paul II Elected Pope is in
        effect." Without it the US may not choose the event."""
        t = Table(turn=8)
        t.hand(USA, "Solidarity")
        t.action_round(USA).answer("card:Solidarity")
        assert "use:event" not in _use_keys(t)

    def test_adds_three_in_poland(self):
        """PD #101: "Add 3 US Influence in Poland.\""""
        t = Table(turn=8)
        t.effect("John Paul II Elected Pope", USA)
        t.inf(ussr={"Poland": 3})
        t.play(USA, "Solidarity", "event")
        assert t.done and t.s.inf(USA, "Poland") == 3
        assert t.gone_for_good("Solidarity")

    def test_ussr_ops_play_without_prerequisite_discards_it(self):
        """R 5.2 bullet 2: an opponent's event whose prerequisite card has not been
        played does not occur, and the asterisked card goes to the discard pile."""
        t = Table(turn=8)
        t.inf(ussr={"Poland": 1})
        t.play(USSR, "Solidarity", "influence")
        t.finish_with(lambda d: "country:Poland" if "country:Poland" in d.legal_keys else d.legal_keys[0])
        assert t.s.inf(USA, "Poland") == 0
        assert t.in_cycle("Solidarity")


# --------------------------------------------------------------------------- #
# #102 Iran-Iraq War
# --------------------------------------------------------------------------- #


class TestC102IranIraqWar:
    def test_victory_with_adjacency_penalty(self):
        """PD #102: "Iran or Iraq invades the other (player's choice). Roll one die and
        subtract 1 for every opponent-controlled country adjacent to target of invasion.
        Player Victory on modified die roll of 4-6. Player adds 2 to Military Ops Track /
        Effects of Victory: Player gains 2 VP and replaces opponent's Influence in target
        country with his own." USSR attacks Iran; US controls Pakistan: 5-1 = 4."""
        t = Table(turn=8)
        t.inf(usa={"Iran": 2, "Pakistan": 2})
        t.dice(5)
        t.play(USSR, "Iran-Iraq War", "event", "country:Iran")
        assert t.done
        assert t.s.inf(USA, "Iran") == 0 and t.s.inf(USSR, "Iran") == 2
        assert t.s.vp == 2 and t.s.military_ops[USSR] == 2
        assert t.gone_for_good("Iran-Iraq War")

    def test_failure_still_military_ops(self):
        """R 7.6: military ops irrespective of success. 4 - 1 = 3: no victory."""
        t = Table(turn=8)
        t.inf(ussr={"Iraq": 2, "Saudi Arabia": 3})
        t.dice(4)
        t.play(USA, "Iran-Iraq War", "event", "country:Iraq")
        assert t.s.inf(USSR, "Iraq") == 2 and t.s.vp == 0
        assert t.s.military_ops[USA] == 2

    def test_flower_power_pays_on_us_ops_play(self):
        """PD #59: "(played as an Event or Operations) unless played on the Space Race"."""
        t = Table(turn=8)
        t.effect("Flower Power", USSR)
        t.play(USA, "Iran-Iraq War", "influence", "country:Mexico", "country:Mexico")
        assert t.done and t.s.vp == 2

    def test_flower_power_pays_on_us_play(self):
        """PD #59: "USSR gains 2 VP for every subsequently US played 'war card' ... War
        Cards: ... Iran-Iraq War.\""""
        t = Table(turn=8)
        t.effect("Flower Power", USSR)
        t.dice(1)
        t.play(USA, "Iran-Iraq War", "event", "country:Iraq")
        assert t.s.vp == 2

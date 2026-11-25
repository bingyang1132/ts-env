"""Regression tests for the 2026-11 engine rules audit.

Each test is named after the audit item it pins down (``test_b10_...`` for item B10 and
so on), and each failed on the engine as it stood at ``6791b34``. The items, the
original behaviour and the rule each fix follows are listed in ``docs/known_gaps.md``.
"""

from __future__ import annotations

import random

import pytest

from twilight import Game, Side, rules
from twilight.baselines import AGENTS
from twilight.data import COUNTRY_ORDER
from twilight.engine import EventContext
from twilight.enums import Phase, Region, WinReason
from twilight.events import handler_for
from twilight.state import VP_LIMIT


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class FixedRoll(random.Random):
    """An rng whose die rolls are scripted; everything else is a seeded Random."""

    def __init__(self, rolls, seed: int = 0) -> None:
        super().__init__(seed)
        self._rolls = list(rolls)

    def randint(self, a, b):  # noqa: D401 - mirrors random.Random
        if self._rolls:
            return self._rolls.pop(0)
        return super().randint(a, b)


def clear_board(game: Game) -> None:
    for side in Side:
        for name in COUNTRY_ORDER:
            game.state.set_inf(side, name, 0)


def drive(gen, chooser=None, log=None):
    """Drive an event/engine generator; *chooser(decision)* picks, default first option."""
    try:
        decision = next(gen)
    except StopIteration as stop:
        return stop.value
    while True:
        if log is not None:
            log.append(decision)
        action = chooser(decision) if chooser else decision.options[0]
        try:
            decision = gen.send(action)
        except StopIteration as stop:
            return stop.value


def prefer(*keys):
    """Chooser taking the first listed key that is legal, else the first option."""

    def chooser(decision):
        for key in keys:
            action = decision.find(key)
            if action is not None:
                return action
        return decision.options[0]

    return chooser


def run_event(game: Game, card: str, player: Side, chooser=None, log=None):
    ctx = EventContext(card=card, player=player, ops=0)
    return drive(handler_for(card)(game, ctx), chooser, log)


def to_action_round(game: Game, seed: int = 0) -> Game:
    """Play random moves until the first PLAY_CARD decision of an action round."""
    rng = random.Random(seed)
    while game.decision is not None and not (
        game.state.phase is Phase.ACTION_ROUND and game.decision.type.value == "play_card"
    ):
        game.step(rng.choice(game.decision.options))
    return game


def play_agents(seed: int, ussr: str, usa: str, optional: bool = False):
    """Play a full game between two baseline agents, checking terminal invariants."""
    game = Game(seed=seed, optional_cards=optional)
    agents = {Side.USSR: AGENTS[ussr](seed=seed), Side.USA: AGENTS[usa](seed=seed + 9999)}
    steps = 0
    while game.decision is not None:
        state = game.state
        assert not state.is_over, "a decision was issued after the game ended"
        assert abs(state.vp) < VP_LIMIT, f"VP {state.vp} reached the limit but play went on"
        decision = game.decision
        chosen = agents[decision.player].act(game, decision)
        game.step(chosen[0] if isinstance(chosen, tuple) else chosen)
        steps += 1
        assert steps < 100_000
    assert game.state.is_over
    return game


# --------------------------------------------------------------------------- #
# B10 -- every VP change ends the game at +/-20, immediately
# --------------------------------------------------------------------------- #


def test_b10_space_race_vp_reaching_twenty_ends_the_game():
    game = to_action_round(Game(seed=3))
    state = game.state
    side = game.decision.player
    state.vp = 19 if side is Side.USSR else -19
    state.rng = FixedRoll([1])  # every space race box succeeds on a 1
    card = next(
        n for n in game.decision.legal_keys
        if n.startswith("card:") and n != "card:The China Card"
        and game.effective_ops(n[5:], side) >= 2
    )
    game.step(card)
    game.step("use:space")
    assert state.is_over
    assert state.winner is side
    assert state.win_reason is WinReason.VICTORY_POINTS
    assert game.decision is None


def test_b10_award_vp_itself_finishes_the_game():
    game = Game(seed=1)
    state = game.state
    state.vp = -18
    state.award_vp(Side.USA, 3)
    assert state.is_over and state.winner is Side.USA
    assert state.win_reason is WinReason.VICTORY_POINTS


def test_b10_first_ending_wins_and_is_not_overwritten():
    game = Game(seed=1)
    state = game.state
    state.vp = 19
    state.award_vp(Side.USSR, 1)
    state.finish(Side.USA, WinReason.DEFCON)
    assert state.winner is Side.USSR
    assert state.win_reason is WinReason.VICTORY_POINTS


def test_b10_yuri_and_samantha_vp_ends_the_game_before_the_coup():
    game = Game(seed=1, optional_cards=True)
    clear_board(game)
    state = game.state
    state.vp = 19
    state.add_effect("Yuri and Samantha", owner=Side.USSR, expires="end_of_turn")
    state.set_inf(Side.USSR, "Angola", 1)
    game.coup(Side.USA, "Angola", 3)
    assert state.is_over and state.winner is Side.USSR


@pytest.mark.parametrize(
    "seed, ussr, usa",
    [(100054, "random", "random"), (110092, "safe_random", "safe_random")],
)
def test_b10_audit_games_end_when_vp_reaches_twenty(seed, ussr, usa):
    play_agents(seed, ussr, usa)


# --------------------------------------------------------------------------- #
# B11 -- once the game is over, no further decision is issued
# --------------------------------------------------------------------------- #


def test_b11_no_ops_decisions_after_a_defcon_loss_mid_card():
    """A US event fired by USSR ops play ends the game; the USSR is asked nothing more."""
    game = to_action_round(Game(seed=5))
    state = game.state
    if game.decision.player is not Side.USSR:
        game = to_action_round(Game(seed=6))
        state = game.state
    assert game.decision.player is Side.USSR
    state.defcon = 2
    # Duck and Cover (US event) played by the USSR for influence: the event degrades
    # DEFCON to 1 and the USSR, as phasing player, loses.
    state.hands[Side.USSR].append("Duck and Cover")
    game.decision = None
    game._driver.close()
    game._driver = _single_play(game, "Duck and Cover", Side.USSR)
    game.decision = game._advance(None)
    while game.decision is not None and game.decision.type.value == "card_use":
        game.step("use:influence")
        break
    while game.decision is not None and game.decision.type.value == "confirm":
        game.step("yes")  # event first
    assert state.is_over
    assert state.win_reason is WinReason.DEFCON
    assert game.decision is None


def _single_play(game, card, side):
    yield from game.play_card(card, side)


@pytest.mark.parametrize(
    "seed, ussr, usa, optional",
    [
        (100010, "random", "random", False),
        (100021, "random", "random", True),
        (110004, "safe_random", "safe_random", False),
        (120000, "greedy", "safe_random", False),
    ],
)
def test_b11_audit_games_issue_no_post_terminal_decisions(seed, ussr, usa, optional):
    play_agents(seed, ussr, usa, optional)


# --------------------------------------------------------------------------- #
# B10 (verdict) -- turn-end order, and no VP auto-victory in final scoring
# --------------------------------------------------------------------------- #


def _turn_end_game(turn: int = 3) -> Game:
    game = Game(seed=2)
    state = game.state
    state.turn = turn
    state.phase = Phase.ACTION_ROUND
    for side in Side:
        state.hands[side][:] = []
    return game


def test_b10_military_ops_check_comes_before_the_held_scoring_card_check():
    """Phase E (military ops) precedes Phase F (held cards): a VP win there stands."""
    game = _turn_end_game()
    state = game.state
    state.vp = 18
    state.defcon = 3
    state.military_ops = [3, 0]  # US short by 3: USSR +3 VP -> 20
    state.hands[Side.USSR].append("Asia Scoring")  # would lose on held card
    drive(game._turn_end())
    assert state.is_over
    assert state.winner is Side.USSR
    assert state.win_reason is WinReason.VICTORY_POINTS


def test_b10_china_card_vp_at_end_of_turn_ten_is_not_an_automatic_victory():
    """Rule 10.3.2: reaching 20 during end-of-turn-10 scoring does not win outright."""
    game = _turn_end_game(turn=10)
    state = game.state
    clear_board(game)
    state.vp = 19
    state.china_card_owner = Side.USSR
    state.set_inf(Side.USA, "Japan", 4)
    state.set_inf(Side.USA, "South Korea", 3)
    state.set_inf(Side.USA, "Thailand", 2)
    drive(game._turn_end())
    assert not state.is_over
    game._final_scoring()
    assert state.is_over
    assert state.win_reason is WinReason.FINAL_SCORING


def test_b10_final_scoring_reaching_twenty_is_not_an_automatic_victory():
    game = _turn_end_game(turn=10)
    state = game.state
    clear_board(game)
    state.final_scoring = True
    state.vp = 19
    state.set_inf(Side.USSR, "Poland", 4)  # USSR scores Europe first: 19 + 4
    state.set_inf(Side.USA, "Japan", 4)  # then the US scores Asia
    state.set_inf(Side.USA, "South Korea", 3)
    state.set_inf(Side.USA, "Pakistan", 2)
    game._final_scoring()
    assert state.win_reason is WinReason.FINAL_SCORING


# --------------------------------------------------------------------------- #
# B5 -- space race box 6: discard a held card at the end of the turn
# --------------------------------------------------------------------------- #


def test_b5_eagle_bear_has_landed_lets_the_holder_discard_a_held_card():
    game = _turn_end_game()
    state = game.state
    state.space_race = [0, 6]
    state.hands[Side.USA][:] = ["Duck and Cover", "Containment"]
    state.hands[Side.USSR][:] = ["Fidel"]
    log = []
    drive(game._turn_end(), prefer("card:Duck and Cover"), log)
    asks = [d for d in log if d.type.value == "discard_card"]
    assert len(asks) == 1 and asks[0].player is Side.USA
    assert "pass" in asks[0].legal_keys
    assert state.hands[Side.USA] == ["Containment"]
    assert "Duck and Cover" in state.discard
    assert state.hands[Side.USSR] == ["Fidel"]


def test_b5_the_ability_is_cancelled_once_the_opponent_arrives():
    game = _turn_end_game()
    state = game.state
    state.space_race = [6, 6]
    state.hands[Side.USA][:] = ["Duck and Cover"]
    log = []
    drive(game._turn_end(), None, log)
    assert not [d for d in log if d.type.value == "discard_card"]
    assert state.hands[Side.USA] == ["Duck and Cover"]


def test_b5_passing_keeps_the_card():
    game = _turn_end_game()
    state = game.state
    state.space_race = [6, 2]
    state.hands[Side.USSR][:] = ["Fidel"]
    drive(game._turn_end(), prefer("pass"))
    assert state.hands[Side.USSR] == ["Fidel"]


# --------------------------------------------------------------------------- #
# B4 -- Man in Earth Orbit: the opponent's headline is revealed before the holder picks
# --------------------------------------------------------------------------- #


def test_b4_man_in_earth_orbit_holder_sees_the_opponent_headline():
    from twilight.observe import observe
    from twilight.render import render

    # The ability only changes the order from turn 2 on, when the headline phase
    # starts with the space race already advanced, so set it before the first phase.
    game = Game(seed=4)
    game.state.space_race = [0, 4]
    rng = random.Random(4)
    while game.state.phase is not Phase.HEADLINE:
        game.step(rng.choice(game.decision.options))
    assert game.decision.type.value == "headline"
    assert game.decision.player is Side.USSR  # the non-holder picks first
    pick = game.decision.options[0].value
    game.step(game.decision.options[0])
    assert game.decision.player is Side.USA
    us_view = observe(game.state, Side.USA, game.decision)
    assert us_view.opponent_headline == pick
    assert pick not in us_view.unseen
    assert pick in render(us_view, include_log=False)
    ussr_view = observe(game.state, Side.USSR, game.decision)
    assert ussr_view.opponent_headline is None
    assert ussr_view.own_headline == pick


def test_b4_without_the_ability_headlines_stay_secret_until_both_are_chosen():
    from twilight.observe import observe

    game = Game(seed=4)
    rng = random.Random(4)
    while game.state.phase is not Phase.HEADLINE:
        game.step(rng.choice(game.decision.options))
    first = game.decision.player
    pick = game.decision.options[0].value
    game.step(game.decision.options[0])
    second = game.decision.player
    assert second is first.opponent
    view = observe(game.state, second, game.decision)
    assert view.opponent_headline is None
    assert pick in view.unseen


# --------------------------------------------------------------------------- #
# B6 -- Vietnam Revolts: +1 op for a USSR card spent entirely in Southeast Asia
# --------------------------------------------------------------------------- #

_SEA = ("Vietnam", "Laos/Cambodia", "Thailand", "Burma", "Malaysia", "Indonesia", "Philippines")


def _vietnam_game() -> Game:
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.phase = Phase.ACTION_ROUND
    state.phasing_player = Side.USSR
    state.set_inf(Side.USSR, "Vietnam", 2)
    state.add_effect("Vietnam Revolts", owner=Side.USSR, expires="end_of_turn")
    return game


def _place_in(*names):
    order = list(names)

    def chooser(decision):
        if decision.type.value == "card_use":
            return decision.find("use:influence")
        for n in order:
            action = decision.find(f"country:{n}")
            if action is not None:
                return action
        return decision.options[0]

    return chooser


def _ussr_total(game, names):
    return sum(game.state.inf(Side.USSR, n) for n in names)


def test_b6_vietnam_revolts_adds_one_op_when_all_points_go_to_southeast_asia():
    game = _vietnam_game()
    game.state.hands[Side.USSR].append("Fidel")  # 2 ops, USSR event
    before = _ussr_total(game, _SEA)
    drive(game.play_card("Fidel", Side.USSR), _place_in("Thailand", "Laos/Cambodia"))
    assert _ussr_total(game, _SEA) - before == 3


def test_b6_no_bonus_when_a_point_leaves_southeast_asia():
    game = _vietnam_game()
    state = game.state
    state.set_inf(Side.USSR, "India", 1)
    state.hands[Side.USSR].append("Fidel")
    drive(game.play_card("Fidel", Side.USSR), _place_in("India", "Thailand"))
    assert state.inf(Side.USSR, "India") + _ussr_total(game, _SEA) == 1 + 2 + 2


def test_b6_vietnam_revolts_stacks_with_the_china_card():
    """Rule 7.4 example: the China Card all in SE Asia under Vietnam Revolts is 6 ops."""
    game = _vietnam_game()
    state = game.state
    state.china_card_owner = Side.USSR
    state.china_card_face_up = True
    before = _ussr_total(game, _SEA)
    drive(game.play_card("The China Card", Side.USSR), _place_in("Thailand", "Laos/Cambodia"))
    assert _ussr_total(game, _SEA) - before == 6


def test_b6_coup_in_southeast_asia_gets_the_bonus():
    game = _vietnam_game()
    state = game.state
    state.set_inf(Side.USA, "Thailand", 1)
    state.hands[Side.USSR].append("Fidel")
    state.rng = FixedRoll([1])

    def chooser(decision):
        if decision.type.value == "card_use":
            return decision.find("use:coup")
        return decision.find("country:Thailand") or decision.options[0]

    drive(game.play_card("Fidel", Side.USSR), chooser)
    assert state.military_ops[Side.USSR] == 3  # 2 ops + 1 from Vietnam Revolts


def test_b6_the_us_gets_no_bonus():
    game = _vietnam_game()
    state = game.state
    state.set_inf(Side.USA, "Thailand", 1)
    state.hands[Side.USA].append("Containment")  # 3 ops
    before = sum(state.inf(Side.USA, n) for n in _SEA)

    def chooser(decision):
        if decision.type.value == "card_use":
            return decision.find("use:influence")
        return decision.find("country:Thailand") or decision.options[0]

    drive(game.play_card("Containment", Side.USA), chooser)
    assert sum(state.inf(Side.USA, n) for n in _SEA) - before == 3


# --------------------------------------------------------------------------- #
# B7 -- North Sea Oil's eighth action round, whenever the card is played
# --------------------------------------------------------------------------- #


def _rounds_taken(game: Game, side: Side, turn: int) -> int:
    return len({
        e.action_round for e in game.state.log
        if e.turn == turn and e.player is side and e.action_round > 0
    })


def test_b7_round_count_is_read_every_round():
    """The US plays North Sea Oil in its 7th (last) action round: an 8th follows."""
    from twilight.engine import ACTION_ROUNDS
    from twilight.enums import Stage

    game = Game(seed=12)
    state = game.state
    state.turn = 8
    state.stages_in_deck.update({Stage.MID_WAR, Stage.LATE_WAR})
    assert ACTION_ROUNDS[Stage.LATE_WAR] == 7
    # Record who acts in which round through the engine's own action-round hook.
    seen: list[tuple[int, Side]] = []
    original = game._action_round

    def spy(side):
        seen.append((state.action_round, side))
        if state.action_round == 7 and side is Side.USA:
            state.add_effect("North Sea Oil", owner=Side.USA, extra_action_round_turn=state.turn)
        state.hands[side][:] = []  # nothing to play: the round is skipped
        return (yield from original(side))

    game._action_round = spy
    game._headline_phase = lambda: iter(())
    drive(game._play_turn())
    assert (8, Side.USA) in seen
    assert (8, Side.USSR) not in seen


# --------------------------------------------------------------------------- #
# B9 -- DEFCON 1 always loses for the phasing player (Cuban Missile Crisis aside)
# --------------------------------------------------------------------------- #


def test_b9_opponent_event_coup_to_defcon_one_loses_for_the_phasing_player():
    """FAQ #62 logic: the US plays Ortega for ops, the USSR's free coup hits DEFCON 1."""
    game = Game(seed=3)
    clear_board(game)
    state = game.state
    state.phase = Phase.ACTION_ROUND
    state.phasing_player = Side.USA
    state.defcon = 2
    state.set_inf(Side.USA, "Cuba", 1)
    state.set_inf(Side.USA, "Nicaragua", 1)
    state.hands[Side.USA].append("Ortega Elected in Nicaragua")
    drive(
        game.play_card("Ortega Elected in Nicaragua", Side.USA),
        prefer("use:influence", "yes", "country:Cuba", "country:Canada", "country:UK"),
    )
    assert state.is_over and state.win_reason is WinReason.DEFCON
    assert state.winner is Side.USSR  # the US, as phasing player, loses


def test_b9_ops_coup_to_defcon_one_still_loses_for_the_coup_maker_when_phasing():
    game = Game(seed=3)
    clear_board(game)
    state = game.state
    state.phasing_player = Side.USSR
    state.defcon = 2
    state.set_inf(Side.USA, "Angola", 1)  # Africa battleground, open at DEFCON 2
    game.coup(Side.USSR, "Angola", 3)
    assert state.is_over and state.winner is Side.USA


def test_b9_cuban_missile_crisis_loses_for_the_coup_maker_even_if_not_phasing():
    game = Game(seed=3)
    clear_board(game)
    state = game.state
    state.phasing_player = Side.USA
    state.defcon = 2
    state.add_effect("Cuban Missile Crisis", owner=Side.USA, expires="end_of_turn")
    state.set_inf(Side.USA, "Angola", 1)
    game.coup(Side.USSR, "Angola", 3)
    assert state.is_over
    assert state.win_reason is WinReason.CUBAN_MISSILE_CRISIS
    assert state.winner is Side.USA


# --------------------------------------------------------------------------- #
# B1 -- Warsaw Pact Formed protects nothing; it only enables NATO
# --------------------------------------------------------------------------- #


def test_b1_warsaw_pact_gives_no_coup_or_realignment_protection():
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.set_inf(Side.USSR, "Poland", 4)
    state.set_inf(Side.USA, "Poland", 1)
    run_event(game, "Warsaw Pact Formed", Side.USSR, prefer("option:1"))
    assert rules.can_coup(state, Side.USA, "Poland")
    assert rules.can_realign(state, Side.USA, "Poland")


def test_b1_warsaw_pact_enables_nato_and_leaves_play():
    from twilight.events import is_playable

    game = Game(seed=1)
    state = game.state
    assert not is_playable(game, "NATO", Side.USA)
    for pile in (state.deck, state.hands[Side.USSR], state.hands[Side.USA]):
        if "Warsaw Pact Formed" in pile:
            pile.remove("Warsaw Pact Formed")
    state.hands[Side.USSR].append("Warsaw Pact Formed")
    drive(game.play_card("Warsaw Pact Formed", Side.USSR), prefer("use:event", "option:1"))
    assert is_playable(game, "NATO", Side.USA)
    assert not state.in_play("Warsaw Pact Formed")
    assert "Warsaw Pact Formed" in state.removed


# --------------------------------------------------------------------------- #
# B2 -- Shuttle Diplomacy
# --------------------------------------------------------------------------- #


def _shuttle_game() -> Game:
    game = Game(seed=1)
    clear_board(game)
    game.state.add_effect("Shuttle Diplomacy", owner=Side.USA)
    return game


def test_b2_the_discounted_battleground_leaves_the_count_entirely():
    """FAQ #73: the country is removed from the equation, not turned non-battleground."""
    game = _shuttle_game()
    state = game.state
    for name, n in (("Iraq", 3), ("Iran", 2), ("Saudi Arabia", 3)):
        state.set_inf(Side.USSR, name, n)
    state.set_inf(Side.USA, "Israel", 4)
    state.set_inf(Side.USA, "Jordan", 2)
    status = rules.region_status(state, Region.MIDDLE_EAST, shuttle=True)
    ussr = status.sides[Side.USSR]
    assert (ussr.countries, ussr.battlegrounds, ussr.non_battlegrounds) == (2, 2, 0)
    assert status.tier(Side.USSR) == "presence"
    # Without the card the USSR has 3 battlegrounds, presence (no non-battleground).
    assert rules.region_vp(state, Region.MIDDLE_EAST, Side.USSR) == 3 + 3
    assert rules.region_vp(state, Region.MIDDLE_EAST, Side.USSR, shuttle=True) == 3 + 2


def test_b2_only_the_scoring_card_sees_it_not_summit_or_kitchen_debates():
    game = _shuttle_game()
    state = game.state
    state.set_inf(Side.USSR, "Iraq", 3)
    state.set_inf(Side.USSR, "Syria", 2)
    state.set_inf(Side.USSR, "Lebanon", 1)
    assert rules.region_status(state, Region.MIDDLE_EAST).tier(Side.USSR) == "domination"


def test_b2_only_one_region_is_discounted_and_the_card_is_discarded():
    game = _shuttle_game()
    state = game.state
    state.set_inf(Side.USSR, "Iraq", 3)
    state.set_inf(Side.USSR, "North Korea", 3)
    asia_before = rules.region_vp(state, Region.ASIA, Side.USSR)
    game.score_region(Region.MIDDLE_EAST)
    assert not state.in_play("Shuttle Diplomacy")
    assert "Shuttle Diplomacy" in state.discard
    assert rules.region_vp(state, Region.ASIA, Side.USSR, shuttle=True) == asia_before


def test_b2_southeast_asia_scoring_neither_uses_nor_spends_it():
    game = _shuttle_game()
    state = game.state
    state.set_inf(Side.USSR, "Thailand", 3)
    game.score_region(Region.SOUTHEAST_ASIA)
    assert state.in_play("Shuttle Diplomacy")


def test_b2_the_adjacency_bonus_goes_with_the_discounted_country():
    """The US picks the discounted battleground; Japan also carries the +1 adjacency."""
    game = _shuttle_game()
    state = game.state
    state.set_inf(Side.USSR, "Japan", 4)
    state.set_inf(Side.USSR, "North Korea", 3)
    state.set_inf(Side.USSR, "Afghanistan", 2)
    # Without: domination 7 + 2 bg + 1 adjacent (Japan) = 10.
    assert rules.region_vp(state, Region.ASIA, Side.USSR) == 10
    # With: Japan removed -> 1 bg + 1 non-bg, domination needs more bg than the US
    # (0): domination 7 + 1 bg = 8, no adjacency bonus.
    assert rules.region_vp(state, Region.ASIA, Side.USSR, shuttle=True) == 8


def test_b2_not_applied_in_final_scoring():
    game = _shuttle_game()
    state = game.state
    state.turn = 10
    state.set_inf(Side.USSR, "Iraq", 3)
    state.set_inf(Side.USSR, "Iran", 2)
    state.set_inf(Side.USSR, "Syria", 2)
    expected = rules.region_vp(state, Region.MIDDLE_EAST, Side.USSR)
    vp0 = state.vp
    game._final_scoring()
    assert state.vp - vp0 == expected


def test_b2_formosan_resolution_counts_for_scoring_but_not_for_summit():
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.add_effect("Formosan Resolution", owner=Side.USA)
    state.set_inf(Side.USA, "Taiwan", 3)
    state.set_inf(Side.USA, "Afghanistan", 2)
    assert rules.region_status(state, Region.ASIA).sides[Side.USA].battlegrounds == 0
    assert rules.region_status(state, Region.ASIA, scoring=True).sides[Side.USA].battlegrounds == 1
    # domination + Taiwan battleground + Afghanistan next to the USSR
    assert rules.region_vp(state, Region.ASIA, Side.USA) == 7 + 1 + 1


# --------------------------------------------------------------------------- #
# O8 -- realignment needs opponent influence in the target
# --------------------------------------------------------------------------- #


def test_o8_cannot_realign_a_country_with_only_your_own_influence():
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.set_inf(Side.USSR, "Angola", 2)
    assert not rules.can_realign(state, Side.USSR, "Angola")
    assert rules.can_realign(state, Side.USA, "Angola")


# --------------------------------------------------------------------------- #
# B12d -- NATO protects West Germany again once Tear Down This Wall cancels Brandt
# --------------------------------------------------------------------------- #


def test_b12d_nato_covers_west_germany_after_tear_down_this_wall():
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.add_effect("NATO", owner=Side.USA)
    state.events_resolved.add("Willy Brandt")
    state.add_effect("Willy Brandt", owner=Side.USSR)
    state.set_inf(Side.USA, "West Germany", 5)
    state.set_inf(Side.USSR, "West Germany", 1)
    assert rules.can_coup(state, Side.USSR, "West Germany")  # Brandt in effect
    run_event(game, "Tear Down This Wall", Side.USA, prefer("option:2", "pass"))
    assert not state.in_play("Willy Brandt")
    assert not rules.can_coup(state, Side.USSR, "West Germany")


# --------------------------------------------------------------------------- #
# B12c -- Brush War: NATO protection, and superpower adjacency counts
# --------------------------------------------------------------------------- #


def test_b12c_brush_war_cannot_target_nato_protected_countries():
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.set_inf(Side.USA, "Greece", 2)
    state.add_effect("NATO", owner=Side.USA)
    log = []
    run_event(game, "Brush War", Side.USSR, prefer("country:Turkey"), log)
    assert "country:Greece" not in log[0].legal_keys
    # The US playing Brush War is unaffected.
    log = []
    state.set_inf(Side.USSR, "Finland", 0)
    run_event(game, "Brush War", Side.USA, None, log)
    assert "country:Greece" in log[0].legal_keys


def test_b12c_war_penalty_counts_the_adjacent_enemy_superpower():
    """FAQ #36 / rule 2.1.5: USSR attacking Mexico subtracts 1 for the adjacent USA."""
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.set_inf(Side.USA, "Mexico", 2)
    state.rng = FixedRoll([3])  # 3 - 1 (USA) = 2: no victory on 3-6
    run_event(game, "Brush War", Side.USSR, prefer("country:Mexico"))
    assert state.inf(Side.USA, "Mexico") == 2
    assert state.inf(Side.USSR, "Mexico") == 0


# --------------------------------------------------------------------------- #
# B13 -- region-specific free coups / realignments ignore DEFCON geography
# --------------------------------------------------------------------------- #


def test_b13_tear_down_this_wall_coup_in_europe_at_defcon_four():
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.defcon = 4
    state.phasing_player = Side.USA
    state.set_inf(Side.USSR, "Poland", 2)
    state.rng = FixedRoll([6])
    log = []
    run_event(game, "Tear Down This Wall", Side.USA, prefer("option:0", "country:Poland"), log)
    labels = [o.label for d in log for o in d.options]
    assert any("coup" in lab for lab in labels)
    assert state.inf(Side.USSR, "Poland") < 2  # the coup happened
    assert state.defcon == 3  # battleground coup still degrades DEFCON
    assert state.military_ops[Side.USA] == 0  # free: no military ops


def test_b13_ordinary_operations_still_respect_defcon():
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.defcon = 4
    state.set_inf(Side.USSR, "Poland", 2)
    assert not rules.can_coup(state, Side.USA, "Poland")


# --------------------------------------------------------------------------- #
# B12a -- De-Stalinization may relocate into any non-US-controlled country
# --------------------------------------------------------------------------- #


def test_b12a_destalinization_destinations_include_ussr_controlled_countries():
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.set_inf(Side.USSR, "Poland", 6)
    state.set_inf(Side.USSR, "Cuba", 3)  # USSR-controlled
    state.set_inf(Side.USA, "Japan", 4)  # US-controlled
    log = []
    run_event(game, "De-Stalinization", Side.USSR, prefer("country:Poland", "country:Cuba"), log)
    place = [d for d in log if d.type.value == "place_influence"]
    assert place and "country:Cuba" in place[0].legal_keys
    assert "country:Poland" in place[0].legal_keys
    assert "country:Japan" not in place[0].legal_keys
    removal = [d for d in log if d.type.value == "remove_influence"]
    assert "pass" in removal[0].legal_keys  # "up to 4"


# --------------------------------------------------------------------------- #
# B12b -- Five-Year Plan only fires a US event that could be played
# --------------------------------------------------------------------------- #


def test_b12b_five_year_plan_skips_an_unplayable_us_event():
    game = Game(seed=1)
    state = game.state
    state.hands[Side.USSR][:] = ["NATO"]  # no Marshall Plan / Warsaw Pact yet
    run_event(game, "Five-Year Plan", Side.USA)
    assert not state.in_play("NATO")
    assert "NATO" in state.discard and "NATO" not in state.removed


def test_b12b_five_year_plan_kitchen_debates_needs_the_battleground_lead():
    game = Game(seed=1)
    state = game.state
    vp0 = state.vp
    state.hands[Side.USSR][:] = ["Kitchen Debates"]
    run_event(game, "Five-Year Plan", Side.USA)
    assert state.vp == vp0
    assert "Kitchen Debates" in state.discard


# --------------------------------------------------------------------------- #
# O12 -- Wargames: "may" end the game
# --------------------------------------------------------------------------- #


def test_o12_wargames_offers_three_choices_and_declining_changes_nothing():
    game = Game(seed=1)
    state = game.state
    state.defcon = 2
    log = []
    run_event(game, "Wargames", Side.USSR, prefer("option:0"), log)
    assert [d.type.value for d in log] == ["choose_option"]
    assert len(log[0].options) == 3
    assert not state.is_over and state.vp == 0


def test_o12_wargames_give_and_end():
    game = Game(seed=1)
    state = game.state
    state.defcon = 2
    state.vp = 10
    run_event(game, "Wargames", Side.USSR, prefer("option:1"))
    assert state.is_over and state.win_reason is WinReason.WARGAMES
    assert state.vp == 4 and state.winner is Side.USSR


def test_o12_wargames_give_but_go_on():
    game = Game(seed=1)
    state = game.state
    state.defcon = 2
    run_event(game, "Wargames", Side.USSR, prefer("option:2"))
    assert not state.is_over and state.vp == -6


def test_o12_wargames_exactly_level_is_a_draw():
    """User ruling 2026-11-25: 0 VP after the 6 is a draw."""
    game = Game(seed=1)
    state = game.state
    state.defcon = 2
    state.vp = 6
    run_event(game, "Wargames", Side.USSR, prefer("option:1"))
    assert state.is_over and state.winner is None
    assert state.win_reason is WinReason.WARGAMES


def test_o12_wargames_always_leaves_the_game_at_defcon_two():
    game = Game(seed=1)
    state = game.state
    for pile in (state.deck, state.hands[0], state.hands[1]):
        if "Wargames" in pile:
            pile.remove("Wargames")
    state.defcon = 2
    state.phase = Phase.ACTION_ROUND
    state.hands[Side.USSR].append("Wargames")
    drive(game.play_card("Wargames", Side.USSR), prefer("use:event", "option:0"))
    assert "Wargames" in state.removed


def test_o12_wargames_above_defcon_two_is_discarded_not_removed():
    from twilight.events import is_playable

    game = Game(seed=1)
    state = game.state
    for pile in (state.deck, state.hands[0], state.hands[1]):
        if "Wargames" in pile:
            pile.remove("Wargames")
    state.defcon = 3
    state.phase = Phase.ACTION_ROUND
    assert not is_playable(game, "Wargames", Side.USSR)
    state.hands[Side.USSR].append("Wargames")
    drive(game.play_card("Wargames", Side.USSR), prefer("use:influence"))
    assert "Wargames" in state.discard and "Wargames" not in state.removed


def test_reformer_is_four_or_six_when_ahead():
    """User ruling 2026-11-25: 4 or 6 (the USSR's choice when ahead), placed in full."""
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.vp = 3
    log = []
    run_event(game, "The Reformer", Side.USSR, prefer("option:1"), log)
    assert log[0].type.value == "choose_option" and len(log[0].options) == 2
    assert sum(state.inf(Side.USSR, n) for n in COUNTRY_ORDER) == 4
    game = Game(seed=1)
    clear_board(game)
    game.state.vp = -1
    log = []
    run_event(game, "The Reformer", Side.USSR, None, log)
    assert all(d.type.value == "place_influence" for d in log)
    assert sum(game.state.inf(Side.USSR, n) for n in COUNTRY_ORDER) == 4


# --------------------------------------------------------------------------- #
# O14 -- Flower Power ignores a war card whose event is banned
# --------------------------------------------------------------------------- #


def test_o14_flower_power_skips_arab_israeli_war_after_camp_david():
    game = Game(seed=1)
    state = game.state
    state.add_effect("Flower Power", owner=Side.USSR)
    state.add_effect("Camp David Accords", owner=Side.USA)
    vp0 = state.vp
    game._check_flower_power("Arab-Israeli War", Side.USA)
    assert state.vp == vp0
    game._check_flower_power("Korean War", Side.USA)
    assert state.vp == vp0 + 2


# --------------------------------------------------------------------------- #
# O15 -- Junta's realignments stay in one of the two regions
# --------------------------------------------------------------------------- #


def test_o15_junta_realignments_are_confined_to_one_region():
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.set_inf(Side.USA, "Mexico", 6)
    state.set_inf(Side.USA, "Brazil", 1)
    state.rng = FixedRoll([1, 1, 1, 1])  # realignments tie: nothing is removed
    log = []

    def chooser(d):
        for key in ("country:Chile", "option:1", "country:Mexico"):
            a = d.find(key)
            if a is not None:
                return a
        return d.options[0]

    run_event(game, "Junta", Side.USSR, chooser, log)
    realigns = [d for d in log if d.type.value == "realign_target"]
    assert len(realigns) == 2
    assert "country:Brazil" in realigns[0].legal_keys  # the first target is free
    assert "country:Brazil" not in realigns[1].legal_keys  # then Central America only


# --------------------------------------------------------------------------- #
# N3 / N4 and the "up to" sweep
# --------------------------------------------------------------------------- #


def test_n3_east_european_unrest_hits_exactly_three_countries_late():
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.turn = 8
    pool = ["Poland", "East Germany", "Hungary", "Romania", "Bulgaria", "Czechoslovakia"]
    for n in pool:
        state.set_inf(Side.USSR, n, 3)
    order = iter(pool)
    run_event(game, "East European Unrest", Side.USA,
              lambda d: d.find(f"country:{next(order)}") or d.options[0])
    touched = [n for n in pool if state.inf(Side.USSR, n) < 3]
    assert touched == pool[:3]
    assert all(state.inf(Side.USSR, n) == 1 for n in touched)


# --------------------------------------------------------------------------- #
# User rulings, 2026-11-25
# --------------------------------------------------------------------------- #


def _ar_game(phasing: Side, defcon: int = 5) -> Game:
    game = Game(seed=1)
    clear_board(game)
    state = game.state
    state.phase = Phase.ACTION_ROUND
    state.phasing_player = phasing
    state.player = phasing
    state.defcon = defcon
    state.action_round = 1
    for side in Side:
        state.hands[side][:] = []
    return game


@pytest.mark.parametrize(
    "card, phasing, coup_side, target",
    [("CIA Created", Side.USSR, Side.USA, "Angola"),
     ("Lone Gunman", Side.USA, Side.USSR, "Angola")],
)
def test_ruling_as_if_coup_to_defcon_one_loses_for_the_card_player(
    card, phasing, coup_side, target
):
    """The opponent's 'as if 1 Op' coup in a battleground at DEFCON 2: the phasing player
    -- who played the card for operations -- loses."""
    game = _ar_game(phasing, defcon=2)
    state = game.state
    state.set_inf(coup_side.opponent, target, 1)
    state.set_inf(phasing, "Mexico" if phasing is Side.USA else "Poland", 1)
    for pile in (state.deck, state.hands[0], state.hands[1]):
        if card in pile:
            pile.remove(card)
    state.hands[phasing].append(card)
    state.rng = FixedRoll([6])
    def chooser(d):
        if d.type.value == "card_use":
            return d.find("use:influence" if d.player is phasing else "use:coup")
        for key in ("yes", f"country:{target}", "country:Mexico", "country:Poland"):
            a = d.find(key)
            if a is not None:
                return a
        return d.options[0]

    drive(game.play_card(card, phasing), chooser)
    assert state.is_over and state.win_reason is WinReason.DEFCON
    assert state.winner is coup_side  # the phasing player lost
    assert state.military_ops[coup_side] == 1


def test_ruling_missile_envy_takes_the_highest_modified_value():
    game = _ar_game(Side.USSR)
    state = game.state
    # Containment (max 4): Duck and Cover (3) and Marshall Plan (4) are both worth 4.
    state.add_effect("Containment", owner=Side.USA, expires="end_of_turn")
    state.hands[Side.USA][:] = ["Duck and Cover", "Marshall Plan"]
    log = []
    run_event(game, "Missile Envy", Side.USSR, prefer("card:Duck and Cover"), log)
    assert set(log[0].legal_keys) == {"card:Duck and Cover", "card:Marshall Plan"}


def test_ruling_the_extra_eighth_round_may_be_forgone():
    game = _ar_game(Side.USA)
    state = game.state
    state.turn = 8
    state.action_rounds_this_turn = 7
    state.action_round = 8
    state.space_race = [0, 8]
    state.hands[Side.USA][:] = ["Duck and Cover"]
    log = []
    drive(game._take_action_round(Side.USA), prefer("pass"), log)
    assert "pass" in log[0].legal_keys
    assert state.hands[Side.USA] == ["Duck and Cover"]


def test_ruling_vietnam_revolts_applies_to_abm_treaty_ops():
    game = _vietnam_game()
    state = game.state
    before = _ussr_total(game, _SEA)
    run_event(game, "ABM Treaty", Side.USSR, _place_in("Thailand", "Laos/Cambodia"))
    assert _ussr_total(game, _SEA) - before == 5


def test_ruling_norad_does_not_fire_for_a_headline_drop():
    game = Game(seed=1)
    state = game.state
    state.add_effect("NORAD", owner=Side.USA)
    state.defcon = 3
    state.phase = Phase.HEADLINE
    state.change_defcon(-1)  # a headline takes DEFCON to 2
    assert state.defcon_moved_to_two
    # The first action round starts with the flag cleared by the turn loop.
    state.defcon_moved_to_two = False
    from twilight.events.special import norad_end_of_action_round

    log = []
    drive(norad_end_of_action_round(game, state.defcon_moved_to_two), None, log)
    assert not log


def test_ruling_chernobyl_names_only_whole_regions():
    game = Game(seed=1)
    log = []
    run_event(game, "Chernobyl", Side.USA, None, log)
    keys = set(log[0].legal_keys)
    assert "region:Western Europe" not in keys and "region:Southeast Asia" not in keys
    assert "region:Europe" in keys and "region:Asia" in keys


def test_ruling_nuclear_subs_spares_ops_coups_but_not_free_coups():
    game = _ar_game(Side.USA)
    state = game.state
    state.add_effect("Nuclear Subs", owner=Side.USA, expires="end_of_turn")
    state.set_inf(Side.USSR, "Iran", 1)
    state.set_inf(Side.USSR, "Cuba", 1)
    game.coup(Side.USA, "Iran", 2)  # an operations coup (incl. 'as if N Ops')
    assert state.defcon == 5
    game.coup(Side.USA, "Cuba", 2, free=True)  # a free coup
    assert state.defcon == 4


def test_ruling_star_wars_fetching_a_ussr_event_is_resolved_by_the_ussr():
    game = _ar_game(Side.USA)
    state = game.state
    state.space_race = [0, 2]
    state.discard[:] = ["Decolonization"]
    log = []
    run_event(game, "Star Wars", Side.USA, prefer("card:Decolonization"), log)
    placing = [d for d in log if d.type.value == "place_influence"]
    assert placing and all(d.player is Side.USSR for d in placing)


def test_ruling_grain_sales_defectors_cancels_an_unresolved_ussr_headline():
    game = Game(seed=1)
    state = game.state
    for side in Side:
        state.hands[side][:] = []
    for pile in (state.deck,):
        for c in ("Grain Sales to Soviets", "Defectors", "Fidel"):
            if c in pile:
                pile.remove(c)
    state.hands[Side.USA][:] = ["Grain Sales to Soviets"]
    state.hands[Side.USSR][:] = ["Fidel", "Defectors"]
    state.rng.choice = lambda seq: "Defectors" if "Defectors" in seq else seq[0]
    state.set_inf(Side.USA, "Cuba", 2)

    def chooser(d):
        for key in ("card:Grain Sales to Soviets", "card:Fidel", "yes", "use:event"):
            a = d.find(key)
            if a is not None:
                return a
        return d.options[0]

    drive(game._headline_phase(), chooser)
    # Grain Sales (2 ops) and Fidel (2 ops): the US resolves first on the tie, takes
    # Defectors and plays it, cancelling Fidel before it resolves.
    assert state.inf(Side.USA, "Cuba") == 2
    assert "Fidel" in state.discard

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

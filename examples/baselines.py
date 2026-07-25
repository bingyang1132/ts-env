"""Baseline agents and a head-to-head runner.

Measuring an agent needs something to measure against. Three, in increasing order of
how useful a yardstick they are:

``RandomAgent``       uniform over legal actions. The floor. Loses to itself on turn 1
                      or 2 in most games, usually by nuking the world, so it barely
                      tests an opponent.
``SafeRandomAgent``   still random, but refuses the two moves that lose immediately
                      (DEFCON 1, and being caught holding a scoring card). Games run to
                      turn 6 on average and reach final scoring, which makes this the
                      most informative baseline of the three.
``GreedyAgent``       scores actions against a hand-written positional heuristic
                      (control, battlegrounds, VP) with hard safety rules.

Measured over 40 games per pairing, 2026-07:

===================  ===================  =================  ==============
USSR                 USA                  USSR win rate      mean end turn
===================  ===================  =================  ==============
greedy               random               57%                2.1
greedy               safe_random          40%                2.7
safe_random          greedy               45%                3.6
safe_random          safe_random          57%                6.0
===================  ===================  =================  ==============

Note the honest result: **GreedyAgent does not reliably beat SafeRandomAgent.** Its
positional heuristic is real but short games are dominated by DEFCON brinkmanship, and a
scorer that has to be taught every way to lose does worse than a filter that simply
refuses to lose. Treat ``safe_random`` as the baseline to beat, and read the two
strategies as evidence that this environment rewards not-losing before it rewards
position. There is also a first-player advantage worth controlling for -- always
evaluate an agent on both sides.

Run a tournament::

    python examples/baselines.py --games 40 --ussr greedy --usa safe_random
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twilight import Game, Side, rules  # noqa: E402
from twilight.data import CARDS, COUNTRIES  # noqa: E402
from twilight.decisions import Action, ActionKind, Decision, DecisionType  # noqa: E402
from twilight.enums import AUTO_VICTORY, OpsUse  # noqa: E402
from twilight.observe import observe  # noqa: E402


class RandomAgent:
    """Uniform over legal actions."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, game: Game, decision: Decision) -> Action:
        return self.rng.choice(decision.options)


_DEFCON_LABEL = re.compile(r"\bDEFCON\s+([1-5])\b", re.IGNORECASE)


def _defcon_in_label(label: str) -> int | None:
    """The DEFCON level an option label names, if it names one."""
    match = _DEFCON_LABEL.search(label or "")
    return int(match.group(1)) if match else None


#: Effect-function names, from the game's own data, that push DEFCON downward.
_DEFCON_DEGRADING = ("DegradeDEFCONLevel", "SetDEFCONLevel")


def degrades_defcon(name: str) -> bool:
    """Whether playing *name* for its event can lower DEFCON.

    Read straight out of the extracted effect specification rather than a hand-kept
    list, so it stays correct if the database is regenerated. A crude substring scan is
    enough for a baseline agent.
    """
    spec = repr(CARDS[name].effect_spec)
    return any(fn in spec for fn in _DEFCON_DEGRADING)


class SafeRandomAgent(RandomAgent):
    """Random, but declines the two moves that lose a game outright.

    A purely random agent nukes itself almost immediately -- most games end on turn 1
    or 2 by DEFCON, which makes it a poor yardstick because an opponent barely has to
    play. This agent still chooses at random, but will not degrade DEFCON to 1 and
    will play a scoring card rather than be caught holding one. Games then run long
    enough to actually measure positional skill.
    """

    name = "safe_random"

    def act(self, game: Game, decision: Decision) -> Action:
        state = game.state
        options = list(decision.options)

        # Play a scoring card rather than lose the game holding it at turn end.
        if decision.type is DecisionType.PLAY_CARD:
            scoring = [
                a for a in options
                if a.kind is ActionKind.CARD and CARDS[a.value].is_scoring
            ]
            if scoring:
                return self.rng.choice(scoring)

        # Never take an action that would bring DEFCON to 1.
        if state.defcon <= 3:
            if decision.type is DecisionType.CARD_USE:
                risky = {OpsUse.COUP.value} if state.defcon <= 2 else set()
                if state.playing_card is not None and degrades_defcon(state.playing_card):
                    risky.add(OpsUse.EVENT.value)
                options = [a for a in options if a.value not in risky] or options
            elif decision.type is DecisionType.COUP_TARGET and state.defcon <= 2:
                safe = [
                    a for a in options
                    if a.kind is not ActionKind.COUNTRY
                    or not COUNTRIES[a.value].battleground
                ]
                options = safe or options

        return self.rng.choice(options)


class GreedyAgent:
    """Scores each legal action by the position it leads to, one ply deep.

    Cloning the game per candidate action would be O(history) each time, which is far
    too slow, so this scores actions by inspecting them directly instead of simulating.
    That makes it a heuristic rather than a true one-ply search -- adequate as a
    baseline, and it never needs the engine to be copyable.
    """

    name = "greedy"

    #: Weights for the position score, from the acting player's point of view.
    W_VP = 1.0
    W_CONTROL = 0.6
    W_BATTLEGROUND = 1.2
    W_INFLUENCE = 0.05

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    # -- scoring ---------------------------------------------------------- #

    def position_score(self, game: Game, side: Side) -> float:
        state = game.state
        score = self.W_VP * state.vp_for(side)
        for name in state.countries_with_influence(side):
            score += self.W_INFLUENCE * state.inf(side, name)
        for name, country in COUNTRIES.items():
            if country.superpower:
                continue
            owner = rules.controller(state, name)
            if owner is None:
                continue
            weight = self.W_BATTLEGROUND if country.battleground else self.W_CONTROL
            score += weight if owner is side else -weight
        return score

    # -- policy ----------------------------------------------------------- #

    def act(self, game: Game, decision: Decision) -> Action:
        side = decision.player
        scored = [(self._value(game, decision, side, a), a) for a in decision.options]
        best = max(s for s, _ in scored)
        # Break ties randomly so repeated games are not identical.
        return self.rng.choice([a for s, a in scored if s == best])

    def _value(self, game: Game, decision: Decision, side: Side, action: Action) -> float:
        state = game.state
        obs = observe(state, side, decision)

        if decision.type is DecisionType.PLAY_CARD:
            return self._value_card(obs, action)
        if decision.type is DecisionType.CARD_USE:
            return self._value_use(game, side, action)
        if action.kind is ActionKind.COUNTRY:
            return self._value_country(game, decision, side, action.value)
        if action.kind is ActionKind.PASS:
            return -0.5  # mildly prefer doing something
        # Several events offer a DEFCON level as a labelled choice, and one of those
        # labels loses the game on the spot. Prefer a high DEFCON generally.
        level = _defcon_in_label(action.label)
        if level is not None:
            return -100.0 if level == 1 else float(level)
        return 0.0

    def _value_card(self, obs, action: Action) -> float:
        name = action.value
        card = CARDS[name]
        # A scoring card must be played before the turn ends or the game is lost, so it
        # always outranks an ordinary card. The region's value only decides which
        # scoring card goes first.
        if card.is_scoring and card.scoring_region is not None:
            net = obs.region(card.scoring_region).net_vp_for_observer
            if net >= AUTO_VICTORY:
                return 1000.0
            return 100.0 + float(net)
        # Playing an opponent's card for operations still fires their event, and the
        # blame for reaching DEFCON 1 falls on whoever played the card. So a low DEFCON
        # makes the opponent's DEFCON-degrading cards unplayable, not merely bad.
        if (
            card.side is not None
            and card.side is not obs.player
            and obs.defcon <= 3
            and degrades_defcon(name)
        ):
            return -100.0

        # Prefer your own events and higher operations values.
        bonus = 2.0 if card.side is obs.player else (-2.0 if card.side is not None else 0.0)
        return card.ops + bonus

    def _value_use(self, game: Game, side: Side, action: Action) -> float:
        use = OpsUse(action.value)
        state = game.state
        if use is OpsUse.EVENT:
            # Some events degrade DEFCON by two, so anything at 3 or below can reach 1
            # and lose the game outright. Conservatively refuse them.
            card_name = state.playing_card
            if state.defcon <= 3 and card_name is not None and degrades_defcon(card_name):
                return -100.0
            return 3.0
        if use is OpsUse.COUP:
            # Never hand the opponent a win by dropping DEFCON to 1.
            return -100.0 if state.defcon <= 2 else 2.5
        if use is OpsUse.DISCARD:
            return -1.0
        if use is OpsUse.INFLUENCE:
            return 2.0
        if use is OpsUse.REALIGN:
            return 1.0
        if use is OpsUse.SPACE:
            return 1.5
        return 0.0

    def _value_country(self, game: Game, decision: Decision, side: Side, name: str) -> float:
        state = game.state
        country = COUNTRIES[name]
        weight = self.W_BATTLEGROUND if country.battleground else self.W_CONTROL

        if decision.type is DecisionType.COUP_TARGET:
            # Reward likely success against valuable targets.
            odds = max(0, min(6, 6 - 2 * country.stability + 4)) / 6.0
            if country.battleground and state.defcon <= 2:
                return -100.0  # would end the game against us
            return weight * odds * 3.0

        if decision.type in (DecisionType.PLACE_INFLUENCE, DecisionType.SETUP_INFLUENCE):
            mine, theirs = state.inf(side, name), state.inf(side.opponent, name)
            needed = country.stability + theirs - mine
            value = weight
            if needed == 1:
                value *= 3.0        # one point away from control
            elif needed <= 0:
                value *= 0.4        # already controlled, low marginal value
            if rules.influence_cost(state, side, name) == 2:
                value *= 0.5
            return value

        if decision.type is DecisionType.REMOVE_INFLUENCE:
            return weight * 2.0 if rules.controls(state, side.opponent, name) else weight

        if decision.type is DecisionType.REALIGN_TARGET:
            return weight * (1.0 + rules.realignment_modifier(state, side, name) * 0.3)

        return weight


AGENTS = {
    "random": RandomAgent,
    "safe_random": SafeRandomAgent,
    "greedy": GreedyAgent,
}


def play_game(ussr, usa, seed: int) -> dict:
    game = Game(seed=seed)
    agents = {Side.USSR: ussr, Side.USA: usa}
    steps = 0
    while game.decision is not None:
        agent = agents[game.decision.player]
        game.step(agent.act(game, game.decision))
        steps += 1
        if steps > 200_000:
            raise RuntimeError("game failed to terminate")
    return {
        "winner": game.state.winner,
        "reason": game.state.win_reason.value if game.state.win_reason else "?",
        "vp": game.state.vp,
        "turn": game.state.turn,
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--ussr", choices=sorted(AGENTS), default="greedy")
    parser.add_argument("--usa", choices=sorted(AGENTS), default="random")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    wins: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    turns: list[int] = []

    for i in range(args.games):
        seed = args.seed + i
        result = play_game(
            AGENTS[args.ussr](seed=seed), AGENTS[args.usa](seed=seed + 9999), seed
        )
        winner = result["winner"]
        wins["draw" if winner is None else winner.label] += 1
        reasons[result["reason"]] += 1
        turns.append(result["turn"])

    print(f"USSR={args.ussr}  USA={args.usa}  over {args.games} games\n")
    for label in ("USSR", "USA", "draw"):
        n = wins[label]
        print(f"  {label:<5} {n:>4}  ({n / args.games:5.1%})")
    print(f"\n  mean final turn: {sum(turns) / len(turns):.1f}")
    print("  endings:", dict(reasons.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Play the game by hand in a terminal, or watch two agents play.

    python tools/play.py                          # you are the USSR, greedy plays the US
    python tools/play.py --side usa --opponent safe_random
    python tools/play.py --side both              # hotseat, you play both sides
    python tools/play.py --side none --pause      # watch two agents, step by step
    python tools/play.py --record game.json       # save the game for tools/viz.py

At any prompt you can type a menu number, an action key, or one of:

    ?  / help     the commands
    b  / board    reprint the board
    l  / log      the full game log
    c  <card>     look up a card's text
    u  / undo     take back your last action (replays the game to get there)
    q  / quit

Append ``# why`` to any move to annotate it -- ``coup:Iran # deny them the battleground``.
The note is stored with the move and shown in the replay, which is how you build a
commented game to read back or to train on. Agents annotate their own moves the same way
(see :mod:`twilight.record`), so a recorded game shows both sides' reasoning.

Undo works by replaying the action history, which is how :meth:`Game.clone` works and is
exact -- but it costs time proportional to how far into the game you are.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twilight import Game, Side  # noqa: E402
from twilight.data import CARDS  # noqa: E402
from twilight.decisions import Decision  # noqa: E402
from twilight.observe import observe  # noqa: E402
from twilight.record import GameRecord, Step, call_agent  # noqa: E402
from twilight.render import render  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from baselines import AGENTS  # noqa: E402

HELP = """\
  <number>      choose that menu entry
  <action key>  e.g. country:Iran, use:coup, pass
  ... # why     annotate the move, e.g. `coup:Iran # deny the battleground`
  b / board     reprint the board
  l / log       the full game log
  c <card>      look up a card by name (partial match works)
  u / undo      take back your last action
  ?  / help     this list
  q  / quit     abandon the game"""


class Quit(Exception):
    """The player asked to stop."""


def show(game: Game, decision: Decision, *, full: bool = True) -> None:
    obs = observe(game.state, decision.player, decision)
    if full:
        print()
        print(render(obs))
    else:
        print()
        if obs.playing_card is not None:
            print(f"RESOLVING: {obs.playing_card}")
        print(f"DECISION ({decision.type}): {decision.prompt}")
        print(decision.menu())


def lookup_card(query: str) -> None:
    query = query.strip().lower()
    matches = [c for name, c in CARDS.items() if query in name.lower()]
    if not matches:
        print(f"  no card matching {query!r}")
        return
    for card in sorted(matches, key=lambda c: c.number)[:5]:
        side = card.side.label if card.side is not None else "neutral"
        print(f"\n  #{card.number} {card.name} -- {card.ops}op, {side}")
        if card.remove_on_event:
            print("  removed from the game if played as an event")
        for line in card.event_text.split("\n"):
            print(f"    {line}")


def prompt_human(
    game: Game, decision: Decision, *, verbose: bool
) -> tuple[str, str | None]:
    """Ask the terminal for an action, handling the meta-commands.

    Returns ``(action_key, note)``; the note is whatever followed a ``#``.
    """
    show(game, decision, full=verbose)
    while True:
        try:
            raw = input(f"\n{decision.player.label}> ").strip()
        except EOFError:
            raise Quit from None

        if not raw:
            continue

        # Everything after the first '#' is the player's own note on the move.
        note: str | None = None
        if "#" in raw:
            raw, _, rest = raw.partition("#")
            raw = raw.strip()
            note = rest.strip() or None
            if not raw:
                print("  a note needs a move in front of it")
                continue

        lowered = raw.lower()

        if lowered in ("q", "quit", "exit"):
            raise Quit
        if lowered in ("?", "h", "help"):
            print(HELP)
            continue
        if lowered in ("b", "board"):
            show(game, decision, full=True)
            continue
        if lowered in ("l", "log"):
            for entry in game.state.log:
                print(f"  {entry}")
            continue
        if lowered in ("u", "undo"):
            return "__undo__", None
        if lowered.startswith("c ") or lowered.startswith("card "):
            lookup_card(raw.split(None, 1)[1])
            continue

        # A menu index?
        if raw.isdigit():
            index = int(raw)
            if 0 <= index < len(decision.options):
                return decision.options[index].key, note
            print(f"  menu index out of range 0..{len(decision.options) - 1}")
            continue

        # An action key, exact or unique prefix.
        if decision.find(raw) is not None:
            return raw, note
        candidates = [k for k in decision.legal_keys if k.lower().startswith(lowered)]
        if len(candidates) == 1:
            print(f"  -> {candidates[0]}")
            return candidates[0], note
        if candidates:
            print(f"  ambiguous, matches: {', '.join(candidates[:8])}")
        else:
            print(f"  {raw!r} is not legal here. Type ? for help, or a menu number.")


def rebuild(seed: int | None, history: list[str], **kwargs) -> Game:
    """Replay *history* into a fresh game. Used by undo."""
    game = Game(seed, **kwargs)
    for key in history:
        if game.decision is None:
            break
        game.step(key)
    return game


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--side",
        choices=["ussr", "usa", "both", "none"],
        default="ussr",
        help="which side you play; 'both' is hotseat, 'none' watches two agents",
    )
    parser.add_argument("--opponent", choices=sorted(AGENTS), default="greedy")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--optional-cards", action="store_true")
    parser.add_argument(
        "--pause", action="store_true", help="in watch mode, wait for Enter each step"
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="show only the decision menu, not the whole board, each time",
    )
    parser.add_argument(
        "--record", type=Path, help="write the finished game to JSON for tools/viz.py"
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    options = {"optional_cards": args.optional_cards}
    game = Game(seed, **options)

    humans: set[Side] = {
        "ussr": {Side.USSR},
        "usa": {Side.USA},
        "both": {Side.USSR, Side.USA},
        "none": set(),
    }[args.side]

    agent = AGENTS[args.opponent](seed=seed ^ 0x5A5A)
    print(f"Twilight Struggle -- seed {seed}")
    if humans:
        print(f"you play {', '.join(sorted(s.label for s in humans))}; "
              f"{args.opponent} plays the rest")
    else:
        print(f"watching {args.opponent} play both sides")
    print("type ? at any prompt for help\n")

    steps: list[Step] = []
    try:
        while game.decision is not None:
            decision = game.decision
            player, dtype = decision.player.label, str(decision.type)
            extra: dict = {}

            if decision.player in humans:
                key, note = prompt_human(game, decision, verbose=not args.brief)
                if key == "__undo__":
                    # Drop back to before this player's previous action.
                    history = list(game.history)
                    while history:
                        history.pop()
                        candidate = rebuild(seed, history, **options)
                        if (
                            candidate.decision is not None
                            and candidate.decision.player in humans
                        ):
                            game = candidate
                            break
                    else:
                        game = rebuild(seed, [], **options)
                    del steps[len(game.history):]
                    print("  ...undone")
                    continue
                if note:
                    print(f'  noted: "{note}"')
                game.step(key)
            else:
                reply, note, extra = call_agent(agent, game, decision)
                key = decision.resolve(reply).key
                reason = f"  // {note}" if note else ""
                print(f"  {player} ({args.opponent}): {dtype} -> {key}{reason}")
                game.step(key)
                if args.pause and not humans:
                    try:
                        input("  [Enter] ")
                    except EOFError:
                        raise Quit from None

            steps.append(
                Step(action=key, player=player, decision_type=dtype, note=note, extra=extra)
            )
    except Quit:
        print("\nabandoned.")
        return 130
    except KeyboardInterrupt:
        print("\ninterrupted.")
        return 130

    state = game.state
    print("\n" + "=" * 70)
    winner = "a draw" if state.winner is None else f"{state.winner.label} wins"
    print(f"GAME OVER after turn {state.turn}: {winner} ({state.win_reason.value})")
    print(f"final VP {state.vp:+d} (USSR-positive), DEFCON {state.defcon}")
    print("=" * 70)
    print("\nlast few events:")
    for entry in state.log[-8:]:
        print(f"  {entry}")

    if args.record:
        record = GameRecord(
            seed=seed,
            optional_cards=game.optional_cards,
            steps=steps,
            winner=state.winner.label if state.winner is not None else "draw",
            win_reason=state.win_reason.value if state.win_reason is not None else None,
            final_vp=state.vp,
            final_turn=state.turn,
            metadata={
                "humans": sorted(s.label for s in humans),
                "opponent": args.opponent,
            },
        )
        record.save(args.record)
        annotated = len(record.annotated())
        print(
            f"\nrecorded to {args.record} ({len(record)} steps, {annotated} annotated)"
            f" -- render it with:\n  python tools/viz.py {args.record}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

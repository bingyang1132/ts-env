"""Driving the environment with a language model.

Deliberately provider-agnostic: supply any ``complete(prompt) -> str`` callable. The
parts worth copying are the retry-on-malformed-output loop and the prompt shape, not the
model plumbing.

    python examples/llm_agent.py --games 1              # runs with a stub "model"

Three things make this work in practice:

1. **The action grammar is closed and explicit.** Every legal move is a short canonical
   key (``country:Iran``, ``use:coup``). The model picks one; nothing has to be parsed
   out of prose. For post-training or constrained decoding, the same keys are the
   vocabulary.
2. **Illegal output is environment feedback, not a crash.** ``IllegalAction`` carries the
   legal set, so a malformed reply becomes another turn of conversation rather than a
   lost episode. Track the rate -- it is a useful capability metric in its own right.
3. **The view already contains the derived facts.** Region control tiers, what each
   region scores right now, control thresholds, coup odds. Models are unreliable at
   recomputing those from raw influence counts, so the environment supplies them.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twilight import Game, Side  # noqa: E402
from twilight.observe import observe  # noqa: E402
from twilight.render import render  # noqa: E402

SYSTEM_PROMPT = """\
You are playing Twilight Struggle, the Cold War board game, as {side}.

You win by reaching +20 victory points, by controlling Europe when Europe Scoring is
played, or by holding the victory point lead after turn 10.

You lose immediately if you:
  - lower DEFCON to 1 (so never coup, and never fire a DEFCON-lowering event, when
    DEFCON is already low -- note that playing an opponent's card for operations still
    triggers their event, and you carry the blame);
  - are still holding a scoring card when the turn ends.

Play to the position you are shown. Prefer taking control of battleground countries,
watch the military operations requirement, and time scoring cards for when a region
pays you.

Reply with exactly one action key from the list you are given, and nothing else."""

USER_PROMPT = """\
{view}

Reply with exactly one action key from the list above, and nothing else."""

RETRY_PROMPT = """\
{error}

Reply with exactly one action key from the list, and nothing else."""


#: Menu lines look like ``  [3] country:West Germany        USSR 3 / US 1, ...``.
#: Country and card keys contain single spaces, so the key runs until two or more.
_MENU_LINE = re.compile(r"^\s*\[\d+\]\s+(\S+(?:\s\S+)*?)(?:\s{2,}|$)", re.MULTILINE)


def stub_model(prompt: str, rng: random.Random) -> str:
    """Stand-in for a real model: picks a legal key at random out of the prompt.

    Lets the harness be exercised end to end with no API access.
    """
    keys = _MENU_LINE.findall(prompt)
    return rng.choice(keys) if keys else "pass"


def extract_key(reply: str, legal_keys: tuple[str, ...]) -> str | None:
    """Pull an action key out of a model's reply.

    Action keys contain spaces -- ``country:West Germany``, ``card:Duck and Cover`` --
    so splitting on whitespace truncates them and turns a correct answer into an
    illegal one. Match against the legal set instead, preferring an exact reply and
    then the longest key the reply contains.
    """
    cleaned = reply.strip().strip(".,`\"'")
    if cleaned in legal_keys:
        return cleaned
    for key in sorted(legal_keys, key=len, reverse=True):
        if key in reply:
            return key
    return None


def choose(game: Game, complete, *, max_retries: int = 3) -> tuple[str, int]:
    """Ask the model for one action, retrying on illegal output.

    Returns the accepted key and how many retries it took.
    """
    decision = game.decision
    assert decision is not None
    obs = observe(game.state, decision.player, decision)
    view = render(obs)

    prompt = SYSTEM_PROMPT.format(side=decision.player.label) + "\n\n"
    prompt += USER_PROMPT.format(view=view)

    for attempt in range(max_retries + 1):
        reply = complete(prompt)
        candidate = extract_key(reply, decision.legal_keys)
        if candidate is not None:
            return candidate, attempt
        if attempt == max_retries:
            break
        error = (
            f"{reply.strip()!r} is not one of the legal action keys. "
            f"Legal: {', '.join(decision.legal_keys[:12])}"
            + (f", ... ({len(decision.options)} total)" if len(decision.options) > 12 else "")
        )
        prompt += f"\n\n{RETRY_PROMPT.format(error=error)}"

    # Fall back to a legal move so a stubborn model does not end the episode.
    return decision.options[0].key, max_retries + 1


def play(seed: int, complete_ussr, complete_usa, *, verbose: bool = False) -> dict:
    game = Game(seed=seed)
    completers = {Side.USSR: complete_ussr, Side.USA: complete_usa}
    retries = Counter()
    steps = 0

    while game.decision is not None:
        side = game.decision.player
        key, used = choose(game, completers[side])
        retries[used] += 1
        if verbose:
            print(f"  {side.label:<4} {game.decision.type:<18} -> {key}")
        game.step(key)
        steps += 1

    return {
        "winner": game.state.winner,
        "reason": game.state.win_reason.value if game.state.win_reason else "?",
        "turn": game.state.turn,
        "steps": steps,
        "retries": retries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the first prompt and exit, to inspect what the model sees",
    )
    args = parser.parse_args()

    if args.show_prompt:
        game = Game(seed=args.seed)
        obs = observe(game.state, game.decision.player, game.decision)
        print(SYSTEM_PROMPT.format(side=game.decision.player.label))
        print()
        print(USER_PROMPT.format(view=render(obs)))
        return 0

    rng = random.Random(args.seed)

    def complete(prompt: str) -> str:
        # Replace with a real model call, e.g.
        #   return client.messages.create(...).content[0].text
        return stub_model(prompt, rng)

    totals: Counter[str] = Counter()
    all_retries: Counter[int] = Counter()
    for i in range(args.games):
        result = play(args.seed + i, complete, complete, verbose=args.verbose)
        winner = result["winner"]
        totals["draw" if winner is None else winner.label] += 1
        all_retries.update(result["retries"])
        print(
            f"game {i}: {'draw' if winner is None else winner.label} "
            f"by {result['reason']} on turn {result['turn']} ({result['steps']} steps)"
        )

    print(f"\nresults: {dict(totals)}")
    clean = all_retries[0]
    total = sum(all_retries.values())
    print(f"first-try legal replies: {clean}/{total} ({clean / total:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

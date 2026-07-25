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
from twilight.record import play_game  # noqa: E402
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

Reply in exactly this format and nothing else:

REASON: <one short sentence on why>
ACTION: <one action key from the list you are given>"""

USER_PROMPT = """\
{view}

Reply in exactly this format:

REASON: <one short sentence on why>
ACTION: <one action key from the list above>"""

RETRY_PROMPT = """\
{error}

Reply in exactly this format:

REASON: <one short sentence on why>
ACTION: <one action key from the list>"""

#: The reason line, if the model followed the format.
_REASON_LINE = re.compile(r"^\s*REASON\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


#: Menu lines look like ``  [3] country:West Germany        USSR 3 / US 1, ...``.
#: Country and card keys contain single spaces, so the key runs until two or more.
_MENU_LINE = re.compile(r"^\s*\[\d+\]\s+(\S+(?:\s\S+)*?)(?:\s{2,}|$)", re.MULTILINE)


def stub_model(prompt: str, rng: random.Random) -> str:
    """Stand-in for a real model: picks a legal key at random out of the prompt.

    Lets the harness be exercised end to end with no API access, in the same reply format
    a real model is asked for.
    """
    keys = _MENU_LINE.findall(prompt)
    key = rng.choice(keys) if keys else "pass"
    return f"REASON: picked at random by the stub model\nACTION: {key}"


def extract_reason(reply: str) -> str | None:
    """The model's stated reason, if it gave one in the requested format."""
    match = _REASON_LINE.search(reply)
    if match:
        return match.group(1).strip() or None
    # Some models answer with prose and then the key; keep the prose as the reason.
    stripped = reply.strip()
    if "\n" in stripped:
        head = stripped.split("\n")[0].strip()
        if head and not head.upper().startswith("ACTION"):
            return head[:300]
    return None


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


class LanguageModelAgent:
    """Wraps any ``complete(prompt) -> str`` into an agent the recorder understands.

    Exposes ``rationale`` and ``extra`` after each choice, which is how
    :func:`twilight.record.play_game` captures the model's stated reason and stores it
    beside the move. That is the whole point: a replay then shows what the model was
    thinking against the board it was looking at.
    """

    def __init__(self, complete, *, name: str = "llm", max_retries: int = 3) -> None:
        self.complete = complete
        self.name = name
        self.max_retries = max_retries
        self.rationale: str | None = None
        self.extra: dict = {}
        #: Counts of how many retries each choice needed; index 0 means first-try legal.
        self.retries: Counter[int] = Counter()

    def act(self, game: Game, decision):
        obs = observe(game.state, decision.player, decision)
        prompt = SYSTEM_PROMPT.format(side=decision.player.label) + "\n\n"
        prompt += USER_PROMPT.format(view=render(obs))

        for attempt in range(self.max_retries + 1):
            reply = self.complete(prompt)
            key = extract_key(reply, decision.legal_keys)
            if key is not None:
                self.rationale = extract_reason(reply)
                self.extra = {"retries": attempt}
                self.retries[attempt] += 1
                return key

            if attempt == self.max_retries:
                break
            legal = ", ".join(decision.legal_keys[:12])
            if len(decision.options) > 12:
                legal += f", ... ({len(decision.options)} total)"
            error = f"{reply.strip()!r} is not one of the legal action keys. Legal: {legal}"
            prompt += f"\n\n{RETRY_PROMPT.format(error=error)}"

        # Fall back to a legal move rather than abandoning the episode, and say so.
        self.retries[self.max_retries + 1] += 1
        self.rationale = (
            f"[harness] model failed to give a legal action in "
            f"{self.max_retries + 1} attempts; fell back to the first legal option"
        )
        self.extra = {"retries": self.max_retries + 1, "fallback": True}
        return decision.options[0].key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="print each move and reason")
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the first prompt and exit, to inspect what the model sees",
    )
    parser.add_argument(
        "--record",
        type=Path,
        help="write the game, with the model's reasons, for tools/viz.py",
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
        agents = {
            Side.USSR: LanguageModelAgent(complete, name="llm-ussr"),
            Side.USA: LanguageModelAgent(complete, name="llm-usa"),
        }

        def show(index, game, step):
            if args.verbose:
                reason = f"  // {step.note}" if step.note else ""
                print(f"  {step.player:<4} {step.decision_type:<18} -> {step.action}{reason}")

        record = play_game(
            agents,
            args.seed + i,
            on_step=show,
            metadata={"ussr": "llm", "usa": "llm", "stub": True},
        )

        totals[record.winner or "unfinished"] += 1
        for agent in agents.values():
            all_retries.update(agent.retries)
        print(
            f"game {i}: {record.winner} by {record.win_reason} on turn "
            f"{record.final_turn} ({len(record)} steps, "
            f"{len(record.annotated())} annotated)"
        )
        if args.record and i == 0:
            record.save(args.record)
            print(f"  recorded to {args.record} -- render it with:"
                  f"\n    python tools/viz.py {args.record}")

    print(f"\nresults: {dict(totals)}")
    clean = all_retries[0]
    total = sum(all_retries.values())
    print(f"first-try legal replies: {clean}/{total} ({clean / total:.1%})")
    if total - clean:
        print(f"retry histogram: {dict(sorted(all_retries.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

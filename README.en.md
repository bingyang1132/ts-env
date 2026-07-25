# Twilight Struggle agent environment

English | [简体中文](README.md)

A rules engine for *Twilight Struggle* (GMT Games, Deluxe edition) built to test and
train agents — reinforcement learning policies, language-model agents, and language
models being post-trained on the game.

The board and card database are **extracted from the shipped game's own Lua files**
rather than transcribed by hand, so country stability values, battleground flags,
adjacency and every card's statistics are authoritative.

```
python tools/extract_lua.py     # regenerate twilight/data/*.json from the game install
python -m pytest tests -q       # 197 tests
python tools/random_play.py --games 200
python tools/demo_views.py      # see both agent views of one position
```

## Design

### One state, three views

Everything an agent sees is a pure function of `GameState`. Information hiding happens
once, in `observe(state, player)` — so the numeric view used by an RL policy and the
text view used by a language model can never disagree about the facts or leak different
amounts.

```
GameState  ──►  observe(state, player)  ──►  Observation
                (hides opponent hand,          │
                 hides draw-pile order)         ├──►  encode(obs)  → numpy dict + action mask
                                                └──►  render(obs)  → text + numbered menu
```

### Factored action space

This is the single most important design decision. A Twilight Struggle turn is a *tree*
of micro-decisions: play a card → choose how to use it → choose targets one at a time.
If one step were a whole turn, the action space would be combinatorial (spending 4
operations points across 84 countries is millions of options).

Instead the engine yields a **stream of atomic decisions**, each with a small legal set
— typically 5–200 options. One atomic action ≈ one click in the real game.

```python
game = Game(seed=0)
while game.decision is not None:
    d = game.decision          # d.player, d.type, d.prompt, d.options
    game.step(pick(d))         # an Action, its canonical key, or a vocabulary index
```

The action vocabulary is **closed and ordered** (237 entries), so index *i* means the
same thing in every game:

| Family | Example key | Count |
|---|---|---|
| pick a card | `card:Duck and Cover` | 110 |
| pick how to use it | `use:coup` | 6 |
| pick a country | `country:West Germany` | 84 |
| pick a region | `region:Middle East` | 9 |
| pick a quantity | `number:3` | 13 |
| event-specific choice | `option:1` | 12 |
| `yes` / `no` / `pass` | `pass` | 3 |

That serves all three routes at once: RL gets a fixed head plus a boolean mask, and a
language model gets a short menu it can read and a stable grammar for constrained
decoding.

### Card events as generators

The whole game is one Python generator. Card effects `yield` decisions and receive the
choice back, so a deeply nested event needs no explicit state machine:

```python
@register("Socialist Governments", playable_if=_not_while_iron_lady)
def socialist_governments(game, ctx):
    """Remove 3 US influence from Western Europe, at most 2 per country."""
    yield from game.remove_influence(
        Side.USA, 3,
        allowed=in_region(Region.WESTERN_EUROPE),
        max_per_country=2,
        chooser=ctx.player,
    )
```

The cost of this choice: a running game holds live generator frames, which cannot be
deep-copied. `Game.clone()` therefore replays the action history against the same seed
— exact, but O(game length). Tree search over long games would want its own snapshotting.

## Using it

### Language-model agent

```python
from twilight import TwilightStruggleEnv

env = TwilightStruggleEnv(seed=0, encode_observations=False)
while env.decision is not None:
    prompt = env.text()                    # board + derived facts + numbered menu
    key = llm(prompt)                      # e.g. "country:Iran"
    env.step(key)
```

The text view deliberately precomputes what models get wrong from raw influence counts:
region control tiers, what each region would score **right now**, control thresholds per
country, coup success odds, and which regions DEFCON has closed.

```
=== You are USSR | Turn 1/10, action round 5 | action_round ===
VP +7 (you lead, +20 wins) | DEFCON 3 | space race you 1 vs 0 (attempts left 0)
military ops you 5 vs 0, need 3 by end of turn or opponent scores the shortfall
DEFCON 3 forbids coups and realignment in: Asia, Europe

BOARD (only countries with influence; ctrl = who controls)
  Europe -- you presence (1c/1bg), opponent presence (1c/0bg), 5 bg total; scoring now: +1 VP to you
    East Germany           BG stab3  USSR  4 / US  0  ctrl USSR
    UK                        stab5  USSR  0 / US  5  ctrl US
    ...

IF SCORED NOW (net VP to you)
    Europe                 +1   (you presence)
    Central America        +2   (you presence)
```

### Reinforcement learning

```python
env = TwilightStruggleEnv(seed=0, reward_mode="sparse")
obs, info = env.reset()
while True:
    logits = policy(obs["countries"], obs["global"], obs["hand"])
    action = sample(logits, mask=obs["action_mask"])
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```

The board arrives as an `(84, 27)` matrix — one row per country, fixed order — so a
network can share weights across countries and attend over them, rather than being
handed a flat blob. `flatten()` concatenates everything (2967 dims) for an MLP baseline.

| Component | Shape |
|---|---|
| `countries` | `(84, 27)` |
| `global` | `(149,)` |
| `hand` / `discard` / `removed` / `effects` / `deck_possible` | `(110,)` each |
| `action_mask` | `(237,)` |

`deck_possible` is deliberate: discard and removed piles are public in the real game, so
deck composition is inferable. Card counting is a genuine skill, and the observation
hands the agent that inference instead of making it re-derive it.

**Two-player convention.** Decisions alternate irregularly and the same side often acts
several times running, so this follows PettingZoo's turn-based convention rather than
pretending to be single-agent: `info["to_move"]` names the side to act, `step()` returns
the reward **from that side's point of view**, and `info["rewards"]` sums to zero.

**Reward modes.** `sparse` is ±1 at the end and 0 elsewhere — the honest objective.
`vp_delta` adds the victory-point change each step, scaled by 1/20. Note that VP shaping
is *not* policy-invariant here: Europe control, DEFCON 1 and a held scoring card all end
games without a matching VP move, so a policy trained only on `vp_delta` will
systematically mishandle them.

### Post-training a language model

Use `encode_observations=False` and `render()`. The text is deterministic — canonical
ordering, integers not floats, stable headers — so the same position tokenizes
identically every time. Actions are short canonical strings (`coup:Iran` style keys), so
you can constrain decoding to a grammar and assign credit at the token level.
`Decision.resolve()` accepts an action key, an `Action`, or a vocabulary index, and
raises `IllegalAction` with the legal set in the message — usable directly as
environment feedback for malformed output.

## Playing and watching

**In a terminal** — also the most useful debugging tool in the repo:

```bash
python tools/play.py                             # you are the USSR, greedy plays the US
python tools/play.py --side both                 # hotseat
python tools/play.py --side none --pause         # watch two agents, step by step
python tools/play.py --record game.json          # record a game for the visualiser
```

At any prompt: a menu number, an action key, or `b` (board), `l` (log), `c <card>` (look
up a card, partial match works), `u` (undo), `?`, `q`. Undo replays the action history, so
it is exact but costs time proportional to how far in you are.

**Graphical board with full replay:**

```bash
python tools/viz.py --open                       # play a game now and open it
python tools/viz.py game.json                    # replay a recording
python tools/viz.py --seed 7 --agents greedy safe_random
```

One self-contained HTML file, no dependencies. The layout comes from the game's own
`map_rect` coordinates, so all 84 countries sit roughly where they do on the physical
board; they are coloured by controller, show both influence totals, and battlegrounds get
a gold outline. The side panel has the VP / DEFCON / turn / military-ops / space-race
tracks, an "if scored now" preview per region, and the cards in play. A slider scrubs the
whole game (arrow keys and play/pause work too).

Every frame is a full snapshot taken by replaying the action history — the same mechanism
as `Game.clone()`, so it is exact. Frames are delta-encoded (a 1048-step game drops from
1.8 MB to about 500 KB), and `tests/test_tools.py` checks the encoding frame by frame with
an independently written decoder, so a delta bug cannot quietly draw a board that never
existed.

## Baselines

```bash
python examples/baselines.py --games 40 --ussr greedy --usa safe_random
python examples/llm_agent.py --show-prompt      # inspect what a model would see
python examples/llm_agent.py --games 3          # run the loop with a stub model
```

Three reference agents: `random` (the floor), `safe_random` (random but refuses the two
instantly-losing moves), and `greedy` (a positional heuristic).

Measured over 40 games per pairing: `greedy` beats `random` 57%, but **does not reliably
beat `safe_random`** (40–55%). That is a real result, not a bug. Short games are dominated
by DEFCON brinkmanship, and a filter that simply refuses to lose outperforms a scorer that
has to be taught every way to lose. Use `safe_random` as the baseline to beat, and always
evaluate on both sides — there is a first-player advantage.

Games between `safe_random` agents last ~6 turns and reach final scoring; games involving
plain `random` end on turn 1–2, which makes it a poor yardstick.

One parsing note worth internalising if you write your own LLM loop: **action keys contain
spaces** (`country:West Germany`). Splitting a model's reply on whitespace truncates the
key and turns a correct answer into an illegal one — this silently rejected 27% of valid
replies until fixed. Match against `decision.legal_keys` instead; see
`examples/llm_agent.py::extract_key`.

## Layout

```
twilight/
  data.py         frozen Country / Card records + indices, loaded from data/*.json
  state.py        GameState: the single source of truth
  rules.py        control, region scoring, coup, realignment, placement legality
  spacerace.py    the 8-box track (Deluxe values) and its four abilities
  decisions.py    the closed action vocabulary and Decision objects
  engine.py       the game as one generator; the API card events are written against
  events/         one handler per card, keyed by name
  observe.py      state -> what one player may know
  encode.py       Observation -> numpy arrays for learned policies
  render.py       Observation -> text for language models
  env.py          reset/step wrapper, reward modes
tools/
  extract_lua.py  regenerate the database from the game install
  dump_card_spec.py   docs/card_spec.md: every card's text + internal effect names
  random_play.py  soak test with invariant checking
  demo_views.py   show both views side by side
  play.py         interactive terminal play: human, watch, record
  viz.py          export a self-contained HTML board and game replay
examples/
  baselines.py    random / safe-random / greedy agents and a tournament runner
  llm_agent.py    prompt loop, retry-on-illegal-output, key extraction
tests/            197 tests
docs/
  card_spec.md    generated: every card's rules text and internal effect names
  known_gaps.md   what is NOT faithfully implemented, and why
```

## Rules fidelity

All 110 cards have an event implementation. Constants were verified against the 2015
GMT Deluxe rulebook, the official FAQ v5, and cross-checked against four independent
open-source implementations. Details that are commonly implemented wrongly and are
handled correctly here:

- **Space race** uses Deluxe values, not 1st/2nd edition (which swap boxes 6–8 and pay
  4/2 for Lunar Orbit). Abilities belong to the first player to a box and are cancelled
  when the opponent arrives.
- **DEFCON restrictions apply to realignment as well as coups**, and Southeast Asia
  counts as Asia. The old printed board track omitted realignment and was wrong.
- **Military operations are netted**: when both players fall short only the difference is
  scored, not each shortfall separately.
- **Domination requires more countries *and* more battlegrounds**, plus at least one
  battleground and one non-battleground. The printed scoring-summary card was incomplete.
- **Influence placement reach is a snapshot** taken at the start of the action round, so
  influence placed this round cannot be chained outward to reach further countries.
- **A losing realignment costs the attacker** influence — the roll is symmetric.
- **Free coups** ignore DEFCON geography and do not count toward required military
  operations, but still degrade DEFCON in a battleground.
- **Final scoring** scores every region, excludes Southeast Asia (already inside Asia),
  and Europe control still wins outright.
- US setup is 25 influence **including Canada 2**, which is easy to miss.

### Known gaps

Three clauses are still not faithful, and they are listed in `docs/known_gaps.md` along
with the rulings taken where sources disagree. Nothing is silently approximated —
`Game(strict_events=True)` raises on any card with no handler, and the test suite asserts
full coverage.

Cards worded "on your opponent's next action round ..." go through a general deferred
trigger mechanism (`GameState.defer` / `Game._fire_deferred`) rather than per-card hacks;
We Will Bury You's cancellable victory points and Missile Envy's forced play both use it.

## Data provenance and licence

The code is MIT licensed (see `LICENSE`).

`twilight/data/*.json` and `docs/card_spec.md` are extracted from the Lua data files in a
local retail installation of the game. They contain Twilight Struggle's card rules text
and board data, which remain **copyright GMT Games and the original authors** and are not
covered by this repository's MIT licence. They are included only to implement the rules
engine, for research use. You are expected to own a copy of the game. To remove that data,
delete both paths and regenerate with `tools/extract_lua.py` against your own install.

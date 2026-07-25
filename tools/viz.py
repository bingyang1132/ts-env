"""Render a game as a self-contained HTML board you can scrub through.

    python tools/viz.py                      # a fresh random game
    python tools/viz.py game.json            # a recording from tools/play.py --record
    python tools/viz.py --seed 7 --agents greedy safe_random --open

Writes one HTML file with no external dependencies: the board layout comes from the
game's own ``map_rect`` coordinates, so countries sit roughly where they do on the
physical board. A slider steps through every decision in the game, showing influence,
control, the tracks, and what just happened.

Every frame is a full snapshot taken by replaying the recorded action history, which is
the same mechanism as :meth:`Game.clone` and therefore exact.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twilight import Game, Side  # noqa: E402
from twilight.data import COUNTRIES, COUNTRY_ORDER  # noqa: E402
from twilight.enums import SCORING_REGIONS  # noqa: E402
from twilight import rules  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from baselines import AGENTS  # noqa: E402

#: Board pixel size the normalised map coordinates are scaled into.
BOARD_W, BOARD_H = 1180, 760


def snapshot(game: Game) -> dict:
    """Everything the page needs to draw one frame."""
    s = game.state
    decision = game.decision

    return {
        "turn": s.turn,
        "ar": s.action_round,
        "phase": s.phase.value,
        "vp": s.vp,
        "defcon": s.defcon,
        "milops": [s.military_ops[Side.USSR], s.military_ops[Side.USA]],
        "space": [s.space_race[Side.USSR], s.space_race[Side.USA]],
        "hands": [len(s.hands[Side.USSR]), len(s.hands[Side.USA])],
        "deck": len(s.deck),
        "discard": len(s.discard),
        "removed": len(s.removed),
        "china": s.china_card_owner.label,
        "chinaUp": s.china_card_face_up,
        "playing": s.playing_card,
        "effects": [
            f"{name}" + (f" ({e.owner.label})" if e.owner is not None else "")
            for name, e in sorted(s.effects.items())
        ],
        # influence[i] = [ussr, usa] in COUNTRY_ORDER order; ctrl 0=USSR 1=USA -1=none
        "inf": [[s.inf(Side.USSR, n), s.inf(Side.USA, n)] for n in COUNTRY_ORDER],
        "ctrl": [
            (-1 if rules.controller(s, n) is None else int(rules.controller(s, n)))
            for n in COUNTRY_ORDER
        ],
        "regions": {
            str(r): {
                "tiers": [
                    rules.region_status(s, r).tier(Side.USSR),
                    rules.region_status(s, r).tier(Side.USA),
                ],
                "vp": [rules.region_vp(s, r, Side.USSR), rules.region_vp(s, r, Side.USA)],
            }
            for r in SCORING_REGIONS
        },
        "toMove": decision.player.label if decision is not None else None,
        "dtype": str(decision.type) if decision is not None else None,
        "prompt": decision.prompt if decision is not None else None,
        "logLen": len(s.log),
        "winner": s.winner.label if s.winner is not None else ("draw" if s.is_over else None),
        "reason": s.win_reason.value if s.win_reason is not None else None,
    }


def record_frames(seed: int | None, history: list[str], *, optional_cards: bool) -> dict:
    """Replay *history*, capturing a snapshot before every action."""
    game = Game(seed, optional_cards=optional_cards)
    frames = [snapshot(game)]
    actions: list[str] = []
    log: list[str] = []

    for key in history:
        if game.decision is None:
            break
        before = len(game.state.log)
        actor = game.decision.player.label
        game.step(key)
        actions.append(f"{actor}: {key}")
        log.append("\n".join(str(e) for e in game.state.log[before:]))
        frames.append(snapshot(game))

    # The first frame has no action leading into it.
    actions.insert(0, "start of game")
    log.insert(0, "")
    _delta_encode(frames)
    return {"frames": frames, "actions": actions, "log": log}


#: Frame fields that usually repeat unchanged and are simply omitted when they do.
_CARRIED_FORWARD = ("regions", "effects")


def _delta_encode(frames: list[dict]) -> None:
    """Shrink the per-frame payload in place; the page expands it again on load.

    A single action usually touches one or two countries and leaves the region standings
    and cards in play alone, so storing everything in full for every frame multiplies
    the file size for no information. Board arrays become lists of changes, and the
    fields in :data:`_CARRIED_FORWARD` are dropped whenever they match the frame before.
    """
    previous_inf = frames[0]["inf"]
    previous_ctrl = frames[0]["ctrl"]
    carried = {key: frames[0][key] for key in _CARRIED_FORWARD}

    for frame in frames[1:]:
        inf, ctrl = frame["inf"], frame["ctrl"]
        frame["dinf"] = [
            [i, inf[i][0], inf[i][1]] for i in range(len(inf)) if inf[i] != previous_inf[i]
        ]
        frame["dctrl"] = [
            [i, ctrl[i]] for i in range(len(ctrl)) if ctrl[i] != previous_ctrl[i]
        ]
        previous_inf, previous_ctrl = inf, ctrl
        del frame["inf"], frame["ctrl"]

        for key in _CARRIED_FORWARD:
            if frame[key] == carried[key]:
                del frame[key]
            else:
                carried[key] = frame[key]


def play_game(seed: int, agents: tuple[str, str], *, optional_cards: bool) -> list[str]:
    game = Game(seed, optional_cards=optional_cards)
    bots = {
        Side.USSR: AGENTS[agents[0]](seed=seed),
        Side.USA: AGENTS[agents[1]](seed=seed ^ 0xFFFF),
    }
    while game.decision is not None:
        game.step(bots[game.decision.player].act(game, game.decision))
    return game.history


def country_layout() -> list[dict]:
    """Board boxes, scaled from the game's normalised coordinates."""
    boxes = []
    for name in COUNTRY_ORDER:
        c = COUNTRIES[name]
        x1, y1, x2, y2 = c.map_rect
        boxes.append(
            {
                "name": name,
                "x": round(x1 * BOARD_W, 1),
                "y": round(y1 * BOARD_H, 1),
                "w": round((x2 - x1) * BOARD_W, 1),
                "h": round((y2 - y1) * BOARD_H, 1),
                "bg": c.battleground,
                "stab": c.stability,
                "region": str(c.region),
            }
        )
    return boxes


def build_html(data: dict, title: str) -> str:
    payload = json.dumps(
        {
            "countries": country_layout(),
            "boardW": BOARD_W,
            "boardH": BOARD_H,
            **data,
        },
        separators=(",", ":"),
    )
    return _TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #14161a; --panel: #1c1f26; --line: #2c313b;
    --ink: #e6e8ec; --dim: #8b93a3;
    --ussr: #d7443e; --usa: #3f7fd0; --neutral: #4b515e;
    --bgland: #23272f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 13px/1.45 ui-sans-serif, system-ui, "Segoe UI", sans-serif;
  }
  header { padding: 12px 18px; border-bottom: 1px solid var(--line); }
  h1 { margin: 0 0 2px; font-size: 15px; font-weight: 600; }
  .sub { color: var(--dim); font-size: 12px; }
  .wrap { display: grid; grid-template-columns: 1fr 320px; gap: 14px; padding: 14px 18px 28px; }
  @media (max-width: 1100px) { .wrap { grid-template-columns: 1fr; } }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
  svg { width: 100%; height: auto; display: block; }
  .cbox { stroke: var(--line); stroke-width: 1; }
  .cname { font-size: 8.5px; fill: var(--ink); pointer-events: none; }
  .cinf  { font-size: 9px; font-weight: 700; pointer-events: none; }
  .bgmark { fill: none; stroke: #c8a24a; stroke-width: 1.6; }
  .controls { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
  input[type=range] { flex: 1; }
  button {
    background: #262b34; color: var(--ink); border: 1px solid var(--line);
    border-radius: 6px; padding: 5px 10px; cursor: pointer; font: inherit;
  }
  button:hover { background: #2f3540; }
  .tracks { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px 14px; }
  .t { border-top: 1px solid var(--line); padding-top: 6px; }
  .t b { display: block; color: var(--dim); font-weight: 500; font-size: 11px; }
  .t span { font-size: 15px; font-weight: 600; }
  .vpbar { height: 8px; background: #262b34; border-radius: 4px; position: relative; margin-top: 4px; }
  .vpbar i { position: absolute; top: 0; bottom: 0; border-radius: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  th, td { text-align: left; padding: 3px 4px; border-bottom: 1px solid var(--line); }
  th { color: var(--dim); font-weight: 500; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  pre {
    margin: 6px 0 0; white-space: pre-wrap; font: 11.5px/1.5 ui-monospace, Consolas, monospace;
    color: var(--dim); max-height: 190px; overflow: auto;
  }
  .ussr { color: var(--ussr); } .usa { color: var(--usa); }
  .action { font: 12px ui-monospace, Consolas, monospace; }
  .over { color: #e8c66a; font-weight: 600; }
  .legend { display: flex; gap: 12px; flex-wrap: wrap; color: var(--dim); font-size: 11px; margin-top: 8px; }
  .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: -1px; margin-right: 4px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub" id="sub"></div>
</header>

<div class="wrap">
  <div>
    <div class="panel">
      <svg id="board" viewBox="0 0 1180 760" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="controls">
        <button id="first">&laquo;</button>
        <button id="prev">&lsaquo;</button>
        <button id="playBtn">play</button>
        <button id="next">&rsaquo;</button>
        <button id="last">&raquo;</button>
        <input type="range" id="slider" min="0" value="0">
        <span id="pos" class="action"></span>
      </div>
      <div class="legend">
        <span><i class="sw" style="background:var(--ussr)"></i>USSR control</span>
        <span><i class="sw" style="background:var(--usa)"></i>US control</span>
        <span><i class="sw" style="background:var(--bgland)"></i>uncontrolled</span>
        <span><i class="sw" style="background:transparent;border:1.5px solid #c8a24a"></i>battleground</span>
        <span>numbers are USSR / US influence</span>
      </div>
    </div>
  </div>

  <div>
    <div class="panel">
      <div class="tracks">
        <div class="t" style="grid-column:1/-1">
          <b>victory points (USSR-positive, &plusmn;20 wins)</b>
          <span id="vp"></span>
          <div class="vpbar"><i id="vpfill"></i></div>
        </div>
        <div class="t"><b>DEFCON</b><span id="defcon"></span></div>
        <div class="t"><b>turn / action round</b><span id="turn"></span></div>
        <div class="t"><b>military ops</b><span id="milops"></span></div>
        <div class="t"><b>space race</b><span id="space"></span></div>
        <div class="t"><b>hands</b><span id="hands"></span></div>
        <div class="t"><b>deck / discard / removed</b><span id="piles"></span></div>
        <div class="t" style="grid-column:1/-1"><b>China Card</b><span id="china"></span></div>
      </div>
    </div>

    <div class="panel" style="margin-top:12px">
      <b style="color:var(--dim);font-weight:500;font-size:11px">this step</b>
      <div class="action" id="action" style="margin:4px 0 2px"></div>
      <div id="prompt" style="color:var(--dim);font-size:11.5px"></div>
      <pre id="log"></pre>
    </div>

    <div class="panel" style="margin-top:12px">
      <b style="color:var(--dim);font-weight:500;font-size:11px">if scored now</b>
      <table id="regions"></table>
    </div>

    <div class="panel" style="margin-top:12px">
      <b style="color:var(--dim);font-weight:500;font-size:11px">in play</b>
      <div id="effects" style="color:var(--dim);font-size:11.5px;margin-top:4px"></div>
    </div>
  </div>
</div>

<script>
const D = __DATA__;
const board = document.getElementById('board');
const slider = document.getElementById('slider');
slider.max = D.frames.length - 1;

// Frames after the first carry only what changed, to keep the file small. Expand them
// once here so scrubbing to any frame stays a plain array lookup.
(function expand() {
  const carried = ['regions', 'effects'];
  let inf = D.frames[0].inf, ctrl = D.frames[0].ctrl;
  const held = {};
  for (const k of carried) held[k] = D.frames[0][k];

  for (let i = 1; i < D.frames.length; i++) {
    const f = D.frames[i];
    inf = inf.slice(); ctrl = ctrl.slice();
    for (const [k, u, a] of f.dinf) inf[k] = [u, a];
    for (const [k, c] of f.dctrl) ctrl[k] = c;
    f.inf = inf; f.ctrl = ctrl;
    delete f.dinf; delete f.dctrl;
    for (const k of carried) {
      if (f[k] === undefined) f[k] = held[k]; else held[k] = f[k];
    }
  }
})();

// -- build the board once, then only update text and fills per frame ----------
const boxes = [];
for (const c of D.countries) {
  const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  r.setAttribute('x', c.x); r.setAttribute('y', c.y);
  r.setAttribute('width', c.w); r.setAttribute('height', c.h);
  r.setAttribute('rx', 2.5); r.setAttribute('class', 'cbox');
  g.appendChild(r);

  if (c.bg) {
    const b = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    b.setAttribute('x', c.x + 1.5); b.setAttribute('y', c.y + 1.5);
    b.setAttribute('width', Math.max(0, c.w - 3));
    b.setAttribute('height', Math.max(0, c.h - 3));
    b.setAttribute('rx', 1.5); b.setAttribute('class', 'bgmark');
    g.appendChild(b);
  }

  const name = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  name.setAttribute('x', c.x + c.w / 2); name.setAttribute('y', c.y + c.h * 0.42);
  name.setAttribute('text-anchor', 'middle'); name.setAttribute('class', 'cname');
  // Long names would overflow a small box; clip to something readable.
  name.textContent = c.name.length > 13 ? c.name.slice(0, 12) + '…' : c.name;
  g.appendChild(name);

  const inf = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  inf.setAttribute('x', c.x + c.w / 2); inf.setAttribute('y', c.y + c.h * 0.85);
  inf.setAttribute('text-anchor', 'middle'); inf.setAttribute('class', 'cinf');
  g.appendChild(inf);

  const tip = document.createElementNS('http://www.w3.org/2000/svg', 'title');
  g.appendChild(tip);

  board.appendChild(g);
  boxes.push({rect: r, inf, tip, c});
}

const COLOURS = ['var(--ussr)', 'var(--usa)'];

function draw(i) {
  const f = D.frames[i];

  boxes.forEach((b, k) => {
    const [u, a] = f.inf[k];
    const ctrl = f.ctrl[k];
    b.rect.setAttribute('fill',
      ctrl === 0 ? 'var(--ussr)' : ctrl === 1 ? 'var(--usa)' : 'var(--bgland)');
    b.rect.setAttribute('fill-opacity', ctrl === -1 ? (u || a ? 0.95 : 0.55) : 0.85);
    b.inf.textContent = (u || a) ? `${u}–${a}` : '';
    b.inf.setAttribute('fill', ctrl === -1 ? 'var(--dim)' : '#fff');
    b.tip.textContent =
      `${b.c.name}  |  ${b.c.region}, stability ${b.c.stab}` +
      (b.c.bg ? ', battleground' : '') +
      `\nUSSR ${u} / US ${a}` +
      (ctrl === -1 ? '\nuncontrolled' : `\n${ctrl === 0 ? 'USSR' : 'US'} controls`);
  });

  const lead = f.vp > 0 ? 'USSR' : f.vp < 0 ? 'US' : 'level';
  document.getElementById('vp').innerHTML =
    `<span class="${f.vp > 0 ? 'ussr' : f.vp < 0 ? 'usa' : ''}">${f.vp > 0 ? '+' : ''}${f.vp}</span>` +
    ` <span style="font-size:11px;color:var(--dim)">${lead}</span>`;
  const fill = document.getElementById('vpfill');
  const pct = Math.min(100, Math.abs(f.vp) / 20 * 50);
  fill.style.background = f.vp >= 0 ? 'var(--ussr)' : 'var(--usa)';
  fill.style.left = f.vp >= 0 ? '50%' : (50 - pct) + '%';
  fill.style.width = pct + '%';

  document.getElementById('defcon').textContent = f.defcon;
  document.getElementById('turn').textContent = `${f.turn} / ${f.ar}`;
  document.getElementById('milops').innerHTML =
    `<span class="ussr">${f.milops[0]}</span> : <span class="usa">${f.milops[1]}</span>` +
    ` <span style="font-size:11px;color:var(--dim)">need ${f.defcon}</span>`;
  document.getElementById('space').innerHTML =
    `<span class="ussr">${f.space[0]}</span> : <span class="usa">${f.space[1]}</span>`;
  document.getElementById('hands').innerHTML =
    `<span class="ussr">${f.hands[0]}</span> : <span class="usa">${f.hands[1]}</span>`;
  document.getElementById('piles').textContent =
    `${f.deck} / ${f.discard} / ${f.removed}`;
  document.getElementById('china').innerHTML =
    `<span class="${f.china === 'USSR' ? 'ussr' : 'usa'}">${f.china}</span>` +
    ` <span style="font-size:11px;color:var(--dim)">${f.chinaUp ? 'face up' : 'face down'}</span>`;

  document.getElementById('action').textContent = D.actions[i];
  let prompt = '';
  if (f.winner) {
    prompt = `<span class="over">game over — ${f.winner} (${f.reason})</span>`;
  } else if (f.prompt) {
    prompt = `${f.toMove} — ${f.dtype}<br>${f.prompt}`;
    if (f.playing) prompt += `<br>resolving: <b>${f.playing}</b>`;
  }
  document.getElementById('prompt').innerHTML = prompt;
  document.getElementById('log').textContent = D.log[i] || '';

  let rows = '<tr><th>region</th><th>USSR</th><th>US</th><th class="num">net</th></tr>';
  for (const [name, r] of Object.entries(f.regions)) {
    const net = r.vp[0] - r.vp[1];
    const big = Math.abs(net) >= 1000;
    const cls = net > 0 ? 'ussr' : net < 0 ? 'usa' : '';
    rows += `<tr><td>${name}</td><td>${r.tiers[0]}</td><td>${r.tiers[1]}</td>` +
            `<td class="num ${cls}">${big ? (net > 0 ? 'WIN' : 'WIN') : (net > 0 ? '+' : '') + net}</td></tr>`;
  }
  document.getElementById('regions').innerHTML = rows;
  document.getElementById('effects').textContent =
    f.effects.length ? f.effects.join(' · ') : 'nothing';

  document.getElementById('pos').textContent = `${i} / ${D.frames.length - 1}`;
  slider.value = i;
}

let at = 0, timer = null;
function go(i) { at = Math.max(0, Math.min(D.frames.length - 1, i)); draw(at); }
slider.oninput = () => go(+slider.value);
document.getElementById('first').onclick = () => go(0);
document.getElementById('prev').onclick = () => go(at - 1);
document.getElementById('next').onclick = () => go(at + 1);
document.getElementById('last').onclick = () => go(D.frames.length - 1);

const playBtn = document.getElementById('playBtn');
playBtn.onclick = () => {
  if (timer) { clearInterval(timer); timer = null; playBtn.textContent = 'play'; return; }
  playBtn.textContent = 'pause';
  timer = setInterval(() => {
    if (at >= D.frames.length - 1) { clearInterval(timer); timer = null; playBtn.textContent = 'play'; return; }
    go(at + 1);
  }, 220);
};
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') go(at + 1);
  if (e.key === 'ArrowLeft') go(at - 1);
  if (e.key === 'Home') go(0);
  if (e.key === 'End') go(D.frames.length - 1);
});

document.getElementById('sub').textContent = D.subtitle;
go(0);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording", nargs="?", type=Path, help="JSON from tools/play.py --record"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--agents",
        nargs=2,
        metavar=("USSR", "USA"),
        default=["greedy", "safe_random"],
        choices=sorted(AGENTS),
        help="agents to play a fresh game with, when no recording is given",
    )
    parser.add_argument("--optional-cards", action="store_true")
    parser.add_argument("-o", "--out", type=Path, default=Path("game.html"))
    parser.add_argument("--open", action="store_true", help="open in a browser when done")
    args = parser.parse_args()

    if args.recording:
        payload = json.loads(args.recording.read_text(encoding="utf-8"))
        seed = payload["seed"]
        history = payload["history"]
        optional = payload.get("optional_cards", False)
        subtitle = f"replay of {args.recording.name} — seed {seed}"
    else:
        seed = args.seed if args.seed is not None else random.randrange(1 << 30)
        optional = args.optional_cards
        history = play_game(seed, tuple(args.agents), optional_cards=optional)
        subtitle = f"{args.agents[0]} (USSR) vs {args.agents[1]} (US) — seed {seed}"

    data = record_frames(seed, history, optional_cards=optional)
    final = data["frames"][-1]
    outcome = final["winner"] or "unfinished"
    subtitle += (
        f" — {len(data['frames'])} steps, {outcome}"
        + (f" by {final['reason']}" if final["reason"] else "")
    )
    data["subtitle"] = subtitle

    args.out.write_text(
        build_html(data, "Twilight Struggle — game replay"), encoding="utf-8"
    )
    print(f"wrote {args.out.resolve()}  ({len(data['frames'])} frames)")
    print(f"  {subtitle}")
    if args.open:
        webbrowser.open(args.out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

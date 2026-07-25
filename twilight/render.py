"""Text rendering of an observation, for language-model agents.

Two goals shape this format:

* **Compact.** Only countries with influence are listed, grouped by region. A full
  84-country dump wastes context on empty spaces that no agent needs.
* **Deterministic.** Fixed ordering, integers rather than floats, stable section
  headers. A post-training run needs the same position to tokenize identically every
  time, and constrained decoding needs the action grammar to be exact.

The rendering deliberately includes derived facts -- region tiers, what each region
would score right now, control thresholds, coup odds. Models reliably fumble that
arithmetic from raw influence counts, and it is free for the engine to supply.
"""

from __future__ import annotations

from .data import CARDS
from .enums import AUTO_VICTORY, Region, Side
from .observe import Observation

#: Region print order: Europe first, since it decides games.
_REGION_ORDER = (
    Region.EUROPE,
    Region.MIDDLE_EAST,
    Region.ASIA,
    Region.SOUTHEAST_ASIA,
    Region.AFRICA,
    Region.CENTRAL_AMERICA,
    Region.SOUTH_AMERICA,
)

#: Regions printed as sub-groups of a parent, so countries are not listed twice.
_SUB_REGIONS = {Region.WESTERN_EUROPE, Region.EASTERN_EUROPE}


def render_board(obs: Observation) -> str:
    """The influence map, listing only countries where somebody has influence."""
    lines: list[str] = []
    shown: set[str] = set()

    for region in _REGION_ORDER:
        rows = []
        for view in obs.countries:
            if view.name in shown or region not in _region_set(obs, view.name):
                continue
            if view.ussr == 0 and view.usa == 0:
                continue
            shown.add(view.name)
            marker = (
                "USSR" if view.controller is Side.USSR
                else "US" if view.controller is Side.USA
                else "--"
            )
            bg = "BG" if view.battleground else "  "
            rows.append(
                f"    {view.name:<22} {bg} stab{view.stability}  "
                f"USSR {view.ussr:>2} / US {view.usa:>2}  ctrl {marker}"
            )
        if not rows:
            continue
        rv = obs.region(region)
        lines.append(f"  {region} -- {_region_summary(obs, rv)}")
        lines.extend(rows)
    return "\n".join(lines)


def _region_set(obs: Observation, name: str) -> set[Region]:
    from .data import COUNTRIES

    return {r for r in COUNTRIES[name].regions if r not in _SUB_REGIONS}


def _region_summary(obs: Observation, rv) -> str:
    me, them = obs.player, obs.opponent
    net = rv.net_vp_for_observer
    if abs(net) >= AUTO_VICTORY:
        swing = "AUTOMATIC VICTORY for " + ("you" if net > 0 else "opponent")
    else:
        swing = f"scoring now: {net:+d} VP to you"
    return (
        f"you {rv.tiers[me]} ({rv.countries[me]}c/{rv.battlegrounds[me]}bg), "
        f"opponent {rv.tiers[them]} ({rv.countries[them]}c/{rv.battlegrounds[them]}bg), "
        f"{rv.total_battlegrounds} bg total; {swing}"
    )


def render_scoring_preview(obs: Observation) -> str:
    """What every region would pay if its scoring card were played right now."""
    rows = []
    for region in _REGION_ORDER:
        rv = obs.region(region)
        net = rv.net_vp_for_observer
        value = (
            "AUTO-WIN" if abs(net) >= AUTO_VICTORY else f"{net:+d}"
        )
        rows.append(f"    {region!s:<16} {value:>8}   (you {rv.tiers[obs.player]})")
    return "\n".join(rows)


def render_hand(obs: Observation) -> str:
    rows = []
    for name in obs.hand:
        c = CARDS[name]
        side = c.side.label if c.side is not None else "neutral"
        tags = []
        if c.is_scoring:
            tags.append("SCORING - must be played this turn")
        if c.remove_on_event:
            tags.append("removed if played as event")
        rows.append(
            f"    #{c.number:<4} {name:<34} {c.ops}op {side:<8} {' '.join(tags)}".rstrip()
        )
    if obs.china_card_available:
        rows.append("    #6    The China Card (available)   4op neutral  +1 op if all ops in Asia")
    return "\n".join(rows) if rows else "    (empty)"


def render_card_counting(obs: Observation, *, list_limit: int = 40) -> str:
    """What is where, for counting cards.

    The discard and removed piles are public in the real game, so they are listed in
    full. Everything else is split into the observer's own hand and the *unseen* set --
    the opponent's hand plus the draw pile, which cannot be told apart.
    """
    lines = ["CARD COUNTING"]
    lines.append(
        f"    draw pile {obs.deck_size} cards, opponent holds {obs.opponent_hand_size}; "
        f"{len(obs.unseen)} cards unseen, each ~{obs.in_deck_odds:.0%} likely to be in "
        "the draw pile rather than their hand"
    )

    scoring = obs.unseen_scoring_cards()
    if scoring:
        lines.append(
            f"    scoring cards still unseen ({len(scoring)}): "
            + ", ".join(str(CARDS[n].scoring_region) for n in scoring)
        )
    else:
        lines.append("    no scoring cards unseen -- all are accounted for")

    by_side = obs.unseen_by_side()
    lines.append(
        "    unseen events by side: "
        f"{by_side['USSR']} USSR, {by_side['USA']} US, {by_side['Neutral']} neutral"
    )

    lines.append(f"    discarded ({len(obs.discard)}): " + _card_list(obs.discard, list_limit))
    lines.append(
        f"    removed from the game ({len(obs.removed)}): "
        + _card_list(obs.removed, list_limit)
    )
    lines.append(f"    unseen ({len(obs.unseen)}): " + _card_list(obs.unseen, list_limit))
    return "\n".join(lines)


def _card_list(names: tuple[str, ...], limit: int) -> str:
    if not names:
        return "none"
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += f", ... (+{len(names) - limit} more)"
    return shown


def render(obs: Observation, *, include_log: bool = True) -> str:
    """Full text view of the position and the pending decision."""
    parts: list[str] = []

    if obs.winner is not None:
        result = f"{obs.winner.label} wins ({obs.win_reason})"
        outcome = "YOU WIN" if obs.winner is obs.player else "YOU LOSE"
        parts.append(f"=== GAME OVER: {result} -- {outcome} ===")

    lead = "you lead" if obs.vp > 0 else "opponent leads" if obs.vp < 0 else "level"
    parts.append(
        f"=== You are {obs.player.label} | Turn {obs.turn}/10, action round "
        f"{obs.action_round} | {obs.phase.value} ==="
    )
    parts.append(
        f"VP {obs.vp:+d} ({lead}, +20 wins) | DEFCON {obs.defcon} | "
        f"space race you {obs.space_race[obs.player]} vs {obs.space_race[obs.opponent]}"
        f" (attempts left {obs.space_attempts_left})"
    )
    parts.append(
        f"military ops you {obs.military_ops[obs.player]} vs "
        f"{obs.military_ops[obs.opponent]}, need {obs.military_ops_required} "
        f"by end of turn or opponent scores the shortfall"
    )

    blocked = _defcon_note(obs)
    if blocked:
        parts.append(blocked)

    parts.append("")
    parts.append("BOARD (only countries with influence; ctrl = who controls)")
    parts.append(render_board(obs))

    parts.append("")
    parts.append("IF SCORED NOW (net VP to you)")
    parts.append(render_scoring_preview(obs))

    parts.append("")
    parts.append(f"YOUR HAND ({len(obs.hand)} cards; opponent holds {obs.opponent_hand_size})")
    parts.append(render_hand(obs))

    if obs.opponent_hand_revealed is not None:
        parts.append("")
        parts.append("OPPONENT HAND (revealed to you)")
        parts.append("    " + ", ".join(obs.opponent_hand_revealed))

    if obs.effects:
        parts.append("")
        parts.append("IN PLAY")
        for name, owner in obs.effects:
            parts.append(f"    {name}" + (f" ({owner})" if owner else ""))

    china = (
        "you hold it"
        if obs.china_card_owner is obs.player
        else "opponent holds it"
    )
    face = "face up" if obs.china_card_face_up else "face down, unusable this turn"
    parts.append("")
    parts.append(f"China Card: {china}, {face}")

    parts.append("")
    parts.append(render_card_counting(obs))

    if include_log and obs.log_tail:
        parts.append("")
        parts.append("RECENT EVENTS")
        for entry in obs.log_tail:
            parts.append(f"    {entry}")

    if obs.decision is not None:
        parts.append("")
        if obs.playing_card is not None:
            card = CARDS[obs.playing_card]
            parts.append(
                f"RESOLVING: {obs.playing_card} ({card.ops}op)"
            )
            if card.event_text:
                for line in card.event_text.split("\n"):
                    parts.append(f"    {line}")
        parts.append(f"DECISION ({obs.decision.type}): {obs.decision.prompt}")
        parts.append("Choose exactly one action key from this list:")
        parts.append(obs.decision.menu())

    return "\n".join(parts)


def _defcon_note(obs: Observation) -> str:
    from . import rules

    blocked = rules.DEFCON_RESTRICTED[obs.defcon]
    if not blocked:
        return ""
    names = ", ".join(sorted(str(r) for r in blocked))
    return f"DEFCON {obs.defcon} forbids coups and realignment in: {names}"


__all__ = [
    "render",
    "render_board",
    "render_card_counting",
    "render_hand",
    "render_scoring_preview",
]

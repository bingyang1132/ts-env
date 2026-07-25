"""Shared building blocks for card events.

These wrap patterns that recur across dozens of cards, so each event implementation
stays close to the wording on the physical card.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Iterator

from .. import rules
from ..data import COUNTRIES, COUNTRY_ORDER, REGION_COUNTRIES
from ..enums import Region, Side

if TYPE_CHECKING:  # pragma: no cover
    from ..decisions import Decision
    from ..engine import Game
    from ..state import GameState


def nothing() -> Iterator["Decision"]:
    """A handler body for events that ask nothing and are resolved immediately."""
    return
    yield  # pragma: no cover - makes this a generator


def ensure_control(state: "GameState", side: Side, name: str) -> int:
    """Add just enough influence for *side* to control *name*. Returns points added."""
    needed = (
        COUNTRIES[name].stability
        + state.inf(side.opponent, name)
        - state.inf(side, name)
    )
    if needed > 0:
        state.add_inf(side, name, needed)
        return needed
    return 0


def in_region(region: Region) -> tuple[str, ...]:
    return REGION_COUNTRIES[region]


def controlled_by(state: "GameState", side: Side, names: Iterable[str] | None = None) -> list[str]:
    pool = COUNTRY_ORDER if names is None else names
    return [n for n in pool if rules.controls(state, side, n)]


def uncontrolled(state: "GameState", names: Iterable[str]) -> list[str]:
    """Countries nobody controls."""
    return [n for n in names if rules.controller(state, n) is None]


def with_influence(state: "GameState", side: Side, names: Iterable[str] | None = None) -> list[str]:
    pool = COUNTRY_ORDER if names is None else names
    return [n for n in pool if state.inf(side, n) > 0]


def empty_countries(state: "GameState", names: Iterable[str] | None = None) -> list[str]:
    """Countries with no influence from either side."""
    pool = COUNTRY_ORDER if names is None else names
    return [
        n for n in pool if state.inf(Side.USSR, n) == 0 and state.inf(Side.USA, n) == 0
    ]


def adjacent_to(name: str) -> list[str]:
    """Neighbours of *name*, excluding the superpower spaces."""
    return [n for n in COUNTRIES[name].adjacent if not COUNTRIES[n].superpower]


def war(
    game: "Game",
    ctx_player: Side,
    *,
    target: str,
    penalty_regions: Iterable[str] | None = None,
    victory_from: int,
    vp: int,
    military_ops: int,
) -> Iterator["Decision"]:
    """Resolve one of the "war" cards.

    Roll a die, subtract one for every opponent-controlled country adjacent to
    *target*, and on *victory_from* or higher the attacker scores *vp* victory points
    and replaces all of the defender's influence in *target* with its own. Military
    operations are credited either way.

    *penalty_regions* is unused for now but kept so the signature matches the handful
    of war cards that count adjacency differently.
    """
    del penalty_regions
    state = game.state
    opponent = ctx_player.opponent

    # Only countries adjacent to the target count; the target itself never does.
    penalty = sum(
        1 for n in adjacent_to(target) if rules.controls(state, opponent, n)
    )
    roll = game.roll()
    modified = roll - penalty
    state.military_ops[ctx_player] = min(5, state.military_ops[ctx_player] + military_ops)

    if modified >= victory_from:
        state.award_vp(ctx_player, vp)
        removed = state.clear_inf(opponent, target)
        state.add_inf(ctx_player, target, removed)
        state.note(
            f"war in {target}: rolled {roll} -{penalty} = {modified}, victory; "
            f"+{vp} VP and {removed} influence flipped",
            ctx_player,
        )
    else:
        state.note(
            f"war in {target}: rolled {roll} -{penalty} = {modified}, no victory",
            ctx_player,
        )
    return
    yield  # pragma: no cover


def free_coup(
    game: "Game",
    side: Side,
    ops: int,
    *,
    allowed: Iterable[str],
    prompt: str = "Choose a country to coup",
) -> Iterator["Decision"]:
    """Let *side* make a coup attempt granted by an event."""
    pool = [n for n in allowed if rules.can_coup(game.state, side, n)]
    name = yield from game.choose_country(
        side, pool, prompt, allow_pass=True,
        labels={n: game.coup_label(n, ops) for n in pool},
    )
    if name is not None:
        game.coup(side, name, ops, free=True)


__all__ = [
    "adjacent_to",
    "controlled_by",
    "empty_countries",
    "ensure_control",
    "free_coup",
    "in_region",
    "nothing",
    "uncontrolled",
    "war",
    "with_influence",
]

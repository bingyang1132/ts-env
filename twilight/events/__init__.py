"""Card event implementations, keyed by card name.

Each handler is a generator written against the :class:`~twilight.engine.Game` API::

    @register("Fidel")
    def fidel(game, ctx):
        game.state.clear_inf(Side.USA, "Cuba")
        ensure_control(game.state, Side.USSR, "Cuba")
        return
        yield                      # keeps it a generator when it asks nothing

Handlers may yield decisions freely; ``yield from game.choose_country(...)`` and the
other helpers on :class:`~twilight.engine.Game` do the asking.

Some events are unplayable under certain conditions -- "Socialist Governments" does
nothing while The Iron Lady is in effect, for instance. Register those with
``playable_if`` so the engine can leave the *event* option off the menu entirely
rather than offering a choice that does nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterator

from ..data import CARDS
from ..enums import Side

if TYPE_CHECKING:  # pragma: no cover
    from ..decisions import Decision
    from ..engine import EventContext, Game

    from ..state import DeferredTrigger

    EventHandler = Callable[[Game, EventContext], Iterator[Decision]]
    Playability = Callable[[Game, Side], bool]
    DeferredHandler = Callable[[Game, DeferredTrigger], Iterator[Decision]]
else:  # pragma: no cover
    EventHandler = Callable
    Playability = Callable
    DeferredHandler = Callable


#: card name -> handler
EVENTS: dict[str, "EventHandler"] = {}
#: card name -> predicate deciding whether the event may currently fire
PLAYABILITY: dict[str, "Playability"] = {}
#: kind -> handler, for effects a card scheduled for a later action round. Registered
#: separately from EVENTS because these fire between action rounds rather than when a
#: card is played. See :meth:`twilight.state.GameState.defer`.
DEFERRED: dict[str, "DeferredHandler"] = {}


def register(
    *names: str, playable_if: "Playability | None" = None
) -> Callable[["EventHandler"], "EventHandler"]:
    """Register a handler for one or more cards.

    Raises if a name is not a real card or already has a handler, so typos and
    accidental duplicates fail at import time rather than mid-game.
    """

    def decorate(fn: "EventHandler") -> "EventHandler":
        for name in names:
            if name not in CARDS:
                raise KeyError(f"cannot register event for unknown card {name!r}")
            if name in EVENTS:
                raise ValueError(f"{name!r} already has an event handler")
            EVENTS[name] = fn
            if playable_if is not None:
                PLAYABILITY[name] = playable_if
        return fn

    return decorate


def register_deferred(kind: str) -> Callable[["DeferredHandler"], "DeferredHandler"]:
    """Register the handler for a deferred trigger *kind*."""

    def decorate(fn: "DeferredHandler") -> "DeferredHandler":
        if kind in DEFERRED:
            raise ValueError(f"deferred kind {kind!r} already has a handler")
        DEFERRED[kind] = fn
        return fn

    return decorate


def deferred_handler_for(kind: str) -> "DeferredHandler":
    try:
        return DEFERRED[kind]
    except KeyError:
        raise KeyError(f"no deferred handler registered for {kind!r}") from None


def handler_for(name: str) -> "EventHandler | None":
    return EVENTS.get(name)


def is_playable(game: "Game", name: str, side: Side) -> bool:
    """Whether *name*'s event can currently do something for *side*.

    Cards with no handler yet report unplayable, which keeps a half-finished registry
    from offering event choices that would silently do nothing.
    """
    if name not in EVENTS:
        return False
    predicate = PLAYABILITY.get(name)
    return True if predicate is None else predicate(game, side)


def missing_handlers() -> list[str]:
    """Cards in the deck with no event implementation, in printed order."""
    return sorted(
        (name for name in CARDS if name not in EVENTS),
        key=lambda n: CARDS[n].number,
    )


def coverage() -> tuple[int, int]:
    """``(implemented, total)`` card counts."""
    return len(EVENTS), len(CARDS)


# Importing the submodules is what populates the registry. Keep this at the bottom so
# the helpers above already exist when they run.
from . import scoring  # noqa: E402,F401
from . import early_war  # noqa: E402,F401
from . import mid_war  # noqa: E402,F401
from . import mid_war2  # noqa: E402,F401
from . import late_war  # noqa: E402,F401
from . import special  # noqa: E402,F401

__all__ = [
    "DEFERRED",
    "EVENTS",
    "PLAYABILITY",
    "coverage",
    "deferred_handler_for",
    "handler_for",
    "is_playable",
    "missing_handlers",
    "register",
    "register_deferred",
]

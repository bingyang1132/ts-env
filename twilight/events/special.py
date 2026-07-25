"""Events needing engine-level cooperation, plus the seven optional cards.

The China Card, UN Intervention and Olympic Games all reach into card play itself
rather than just moving influence around, so they live here alongside the optional
cards rather than with their printed stage.

Four effects in this module cannot be finished from inside an event handler, because
they fire at moments the engine never routes through :meth:`Game._resolve_event`. Each
is implemented as a plain generator/function here and marked ``ENGINE HOOK`` so the
call site in :mod:`twilight.engine` is a one-liner once someone adds it:

* :func:`china_card_ops_bonus` -- the China Card's +1 Ops when every point went to Asia,
  which is only knowable after the operations have been spent;
* :func:`china_card_cancels_formosan_resolution` -- the China Card played by the US
  removes Formosan Resolution from play;
* :func:`norad_end_of_action_round` -- NORAD's bonus influence at the end of an action
  round in which DEFCON dropped to 2;
* :func:`yuri_and_samantha_on_coup` -- 1 VP to the USSR per US coup attempt.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from .. import rules
from ..data import CARDS, CHINA_CARD, COUNTRIES, COUNTRY_ORDER, REGION_COUNTRIES
from ..decisions import Decision, DecisionType, use_action
from ..enums import OpsUse, Phase, Region, Side
from . import register
from .helpers import adjacent_to, in_region, nothing, with_influence

# --------------------------------------------------------------------------- #
# Local plumbing
# --------------------------------------------------------------------------- #


def _conduct_operations(
    game,
    side: Side,
    ops: int,
    *,
    prompt: str,
) -> Iterator[Decision]:
    """Let *side* spend *ops* points as influence, a coup, or realignments.

    This is :meth:`Game.free_operations` with ``free=False``: the three cards here
    ("use its Operations value to Conduct Operations", "as if they played a 4 Ops
    card") spend a real card's operations, so a coup taken with them counts toward the
    required military operations. ``Game.free_operations`` marks its coups free, which
    is right for events that *grant* a coup outside of card play but not for these.
    """
    state = game.state
    available = []
    if rules.placeable_countries(state, side):
        available.append(use_action(OpsUse.INFLUENCE, f"place {ops} influence"))
    if any(rules.can_coup(state, side, n) for n in COUNTRY_ORDER):
        available.append(use_action(OpsUse.COUP, f"coup with {ops} ops"))
    if any(rules.can_realign(state, side, n) for n in COUNTRY_ORDER):
        available.append(use_action(OpsUse.REALIGN, f"{ops} realignment rolls"))

    if not available:
        state.note(f"{ops} operations point(s) wasted: nothing legal", side)
        return

    if len(available) == 1:
        use = OpsUse(available[0].value)
    else:
        action = yield from game.ask(
            Decision(
                DecisionType.CARD_USE,
                side,
                prompt,
                tuple(available),
                {"ops": ops},
            )
        )
        use = OpsUse(action.value)

    yield from game.conduct_operations(side, ops, use)


def _card_label(name: str, side: Side) -> str:
    """Short annotation for a card offered outside of a normal action round."""
    c = CARDS[name]
    bits = ["scoring" if c.is_scoring else f"{c.ops}op"]
    if c.side is None:
        bits.append("neutral")
    elif c.side is side:
        bits.append("yours")
    else:
        bits.append(f"{c.side.label} event")
    return ", ".join(bits)


# --------------------------------------------------------------------------- #
# #6 The China Card
# --------------------------------------------------------------------------- #


def _china_card_never_an_event(game, side) -> bool:
    # The China Card has no event: it is only ever played for its operations. It is
    # registered so the coverage report does not list it as missing, and reports
    # unplayable so the engine never offers "resolve the event" as a use.
    return False


@register(CHINA_CARD, playable_if=_china_card_never_an_event)
def the_china_card(game, ctx):
    """No event; the card is pure operations plus the bonuses handled by the engine.

    Of the four lines of text, two are already live elsewhere -- the hand-off after
    play (``Game._retire_card``) and the +1 VP at the end of turn 10 (``Game._turn_end``)
    -- and two still need engine hooks, see :func:`china_card_ops_bonus` and
    :func:`china_card_cancels_formosan_resolution`.
    """
    return (yield from nothing())


def all_points_in_asia(names: Iterable[str]) -> bool:
    """Whether every country in *names* is in Asia (Southeast Asia counts as Asia)."""
    names = list(names)
    return bool(names) and all(
        Region.ASIA in COUNTRIES[n].regions for n in names
    )


def china_card_ops_bonus(card_name: str, touched: Iterable[str]) -> int:
    """ENGINE HOOK: the China Card's "+1 Operations value when all points are in Asia".

    *touched* is every country the operations affected. Not called anywhere yet: the
    bonus cannot be applied by ``Game.effective_ops``, which runs before the player has
    chosen where the points go. See the module docstring for what the engine needs.
    """
    if card_name != CHINA_CARD:
        return 0
    return 1 if all_points_in_asia(touched) else 0


def china_card_cancels_formosan_resolution(game, side: Side) -> None:
    """ENGINE HOOK: the US playing the China Card removes Formosan Resolution.

    Belongs in ``Game._retire_card`` (or ``play_card``) next to the hand-off, since the
    China Card has no event handler to run. Not called anywhere yet.
    """
    if side is Side.USA and game.state.in_play("Formosan Resolution"):
        game.state.remove_effect("Formosan Resolution")
        game.state.note("the China Card cancels Formosan Resolution", side)


# --------------------------------------------------------------------------- #
# #20 Olympic Games
# --------------------------------------------------------------------------- #


@register("Olympic Games")
def olympic_games(game, ctx):
    """The opponent either competes (duelling dice, sponsor +2) or boycotts."""
    state = game.state
    sponsor, opponent = ctx.player, ctx.player.opponent

    choice = yield from game.choose_option(
        opponent,
        f"Olympic Games: {sponsor.label} sponsors the games",
        ["participate: both roll, the sponsor adds 2, high roll scores 2 VP",
         "boycott: degrade DEFCON one level, the sponsor gets 4 ops"],
    )

    if choice == 0:
        # Ties are re-rolled, so this always resolves to a winner.
        while True:
            sponsor_roll = game.roll()
            opponent_roll = game.roll()
            if sponsor_roll + 2 != opponent_roll:
                break
            state.note(
                f"Olympic Games: {sponsor_roll}+2 vs {opponent_roll}, tied, re-rolling",
                ctx.player,
            )
        winner = sponsor if sponsor_roll + 2 > opponent_roll else opponent
        state.award_vp(winner, 2)
        state.note(
            f"Olympic Games: {sponsor.label} {sponsor_roll}+2 vs {opponent.label} "
            f"{opponent_roll}, {winner.label} +2 VP",
            ctx.player,
        )
        return

    # A boycott. The DEFCON degradation is attributed to the sponsor, who is the
    # phasing player, even though the opponent chose to boycott.
    state.note(f"Olympic Games: {opponent.label} boycotts", ctx.player)
    game.degrade_defcon(1, sponsor)
    if state.is_over:
        return
    yield from _conduct_operations(
        game, sponsor, 4, prompt="Olympic Games: conduct operations as if a 4 Op card"
    )


# --------------------------------------------------------------------------- #
# #32 UN Intervention
# --------------------------------------------------------------------------- #


def _un_intervention_candidates(game, side: Side) -> list[str]:
    """Cards in *side*'s hand carrying the opponent's event and some ops value."""
    return [
        n
        for n in game.state.hand(side)
        if CARDS[n].side is side.opponent and CARDS[n].ops > 0
    ]


def _un_intervention_playable(game, side) -> bool:
    # Needs an opponent-associated card to cancel, and never in the headline phase.
    if game.state.phase is Phase.HEADLINE:
        return False
    return bool(_un_intervention_candidates(game, side))


@register("UN Intervention", playable_if=_un_intervention_playable)
def un_intervention(game, ctx):
    """Cancel an opponent event in hand and spend that card's ops instead."""
    state = game.state
    candidates = _un_intervention_candidates(game, ctx.player)
    chosen = yield from game.choose_card(
        ctx.player,
        candidates,
        "UN Intervention: cancel which opponent event and use its operations?",
        labels={
            n: f"{game.effective_ops(n, ctx.player)}op, {CARDS[n].side.label} event"
            for n in candidates
        },
    )
    if chosen is None:
        return

    ops = game.effective_ops(chosen, ctx.player)
    state.hand(ctx.player).remove(chosen)
    # The cancelled event never fires, so the card is discarded rather than resolved:
    # it does not count as played for its event and is not removed from the game.
    state.discard_card(chosen)
    state.note(
        f"UN Intervention cancels {chosen} and uses its {ops} ops", ctx.player
    )

    # U2 Incident pays the USSR one further VP if UN Intervention is played as an
    # event while it is in effect, and is then spent.
    if state.in_play("U2 Incident"):
        state.award_vp(Side.USSR, 1)
        state.note("U2 Incident: UN Intervention played as an event, USSR +1 VP", ctx.player)
        state.remove_effect("U2 Incident")

    # The US playing this as an event heads off We Will Bury You's victory points.
    if ctx.player is Side.USA and state.cancel_deferred(card="We Will Bury You"):
        state.note(
            "UN Intervention cancels the victory points from We Will Bury You", ctx.player
        )

    yield from _conduct_operations(
        game, ctx.player, ops, prompt=f"UN Intervention: spend {chosen}'s {ops} ops"
    )


# --------------------------------------------------------------------------- #
# #104 The Cambridge Five
# --------------------------------------------------------------------------- #


def _cambridge_five_playable(game, side) -> bool:
    # No effect in the Late War, which begins on turn 8.
    return game.state.turn < 8


@register("The Cambridge Five", playable_if=_cambridge_five_playable)
def the_cambridge_five(game, ctx):
    """The US reveals its scoring cards; the USSR adds 1 influence in a named region."""
    state = game.state
    revealed = [n for n in state.hand(Side.USA) if CARDS[n].is_scoring]
    if not revealed:
        state.note("The Cambridge Five: the US holds no scoring cards", ctx.player)
        return (yield from nothing())

    state.note(
        "The Cambridge Five reveals " + ", ".join(sorted(revealed)), ctx.player
    )
    regions = [CARDS[n].scoring_region for n in revealed]
    pool = [n for region in regions for n in REGION_COUNTRIES[region]]
    yield from game.add_influence(
        Side.USSR,
        1,
        allowed=pool,
        prompt=(
            "The Cambridge Five: 1 USSR influence in a region named on a revealed "
            "scoring card (" + ", ".join(str(r) for r in dict.fromkeys(regions)) + ")"
        ),
    )


# --------------------------------------------------------------------------- #
# #105 Special Relationship
# --------------------------------------------------------------------------- #


def _special_relationship_playable(game, side) -> bool:
    # Both halves of the card require US control of the UK.
    return rules.controls(game.state, Side.USA, "UK")


@register("Special Relationship", playable_if=_special_relationship_playable)
def special_relationship(game, ctx):
    """With the UK controlled: 1 influence next door, or 2 VP and 2 in Western Europe."""
    state = game.state
    if not state.in_play("NATO"):
        yield from game.add_influence(
            Side.USA,
            1,
            allowed=adjacent_to("UK"),
            prompt="Special Relationship: 1 US influence in a country adjacent to the UK",
        )
        return

    name = yield from game.choose_country(
        Side.USA,
        in_region(Region.WESTERN_EUROPE),
        "Special Relationship: 2 US influence in one Western European country",
    )
    if name is not None:
        state.add_inf(Side.USA, name, 2)
        state.note(f"Special Relationship: 2 US influence in {name}", ctx.player)
    state.award_vp(Side.USA, 2)


# --------------------------------------------------------------------------- #
# #106 NORAD
# --------------------------------------------------------------------------- #


@register("NORAD")
def norad(game, ctx):
    """Stays in play; see :func:`norad_end_of_action_round` for what it then does."""
    game.state.add_effect(ctx.card, owner=Side.USA)
    return (yield from nothing())


def norad_end_of_action_round(game, defcon_dropped_to_2: bool) -> Iterator[Decision]:
    """ENGINE HOOK: NORAD's bonus influence at the end of an action round.

    While NORAD is in play and the US controls Canada, an action round in which the
    DEFCON marker moved to the 2 box lets the US add 1 influence to any country that
    already contains US influence. Not called anywhere yet -- the engine's action-round
    loop would have to remember the DEFCON level at the start of the round and call
    this at the end of it.
    """
    state = game.state
    if not defcon_dropped_to_2 or not state.in_play("NORAD", owner=Side.USA):
        return
    if not rules.controls(state, Side.USA, "Canada"):
        return
    pool = with_influence(state, Side.USA)
    if not pool:
        return
    yield from game.add_influence(
        Side.USA,
        1,
        allowed=pool,
        prompt="NORAD: 1 US influence in a country already containing US influence",
    )


# --------------------------------------------------------------------------- #
# #107 Che
# --------------------------------------------------------------------------- #

#: Che may only strike non-battlegrounds in these three regions.
_CHE_REGIONS = (Region.CENTRAL_AMERICA, Region.SOUTH_AMERICA, Region.AFRICA)


def _che_targets(game, exclude: str | None = None) -> list[str]:
    return [
        n
        for region in _CHE_REGIONS
        for n in in_region(region)
        if not COUNTRIES[n].battleground
        and n != exclude
        and rules.can_coup(game.state, Side.USSR, n)
    ]


@register("Che")
def che(game, ctx):
    """A coup in a Latin American or African non-battleground, twice if the first bit."""
    state = game.state
    ops = game.effective_ops(ctx.card, Side.USSR)

    pool = _che_targets(game)
    first = yield from game.choose_country(
        Side.USSR,
        pool,
        f"Che: coup a non-battleground in Latin America or Africa ({ops} ops)",
        dtype=DecisionType.COUP_TARGET,
        allow_pass=True,
        labels={n: game.coup_label(n, ops) for n in pool},
        ops=ops,
    )
    if first is None:
        return

    result = game.coup(Side.USSR, first, ops)
    # A second attempt only if the first actually took US influence off the board.
    if result.removed == 0 or state.is_over:
        return

    pool = _che_targets(game, exclude=first)
    again = yield from game.choose_country(
        Side.USSR,
        pool,
        f"Che: the coup removed US influence, so coup a second, different country ({ops} ops)",
        dtype=DecisionType.COUP_TARGET,
        allow_pass=True,
        labels={n: game.coup_label(n, ops) for n in pool},
        ops=ops,
    )
    if again is not None:
        game.coup(Side.USSR, again, ops)


# --------------------------------------------------------------------------- #
# #108 Our Man In Tehran
# --------------------------------------------------------------------------- #


def _our_man_in_tehran_playable(game, side) -> bool:
    # Requires at least one US-controlled Middle East country.
    return any(
        rules.controls(game.state, Side.USA, n)
        for n in in_region(Region.MIDDLE_EAST)
    )


@register("Our Man In Tehran", playable_if=_our_man_in_tehran_playable)
def our_man_in_tehran(game, ctx):
    """Look at the top 5 cards, discard any of them, reshuffle the rest back in."""
    state = game.state
    drawn: list[str] = []
    for _ in range(5):
        name = state.draw_card()
        if name is None:
            break
        drawn.append(name)

    if not drawn:
        state.note("Our Man In Tehran: the draw pile is empty", ctx.player)
        return

    # Off the deck but not yet anywhere else, for as long as the choices take.
    state.hold(*drawn)
    state.note("Our Man In Tehran reveals " + ", ".join(drawn), ctx.player)

    while drawn:
        chosen = yield from game.choose_card(
            Side.USA,
            drawn,
            "Our Man In Tehran: discard a revealed card, or pass to keep the rest "
            "in the deck",
            dtype=DecisionType.DISCARD_CARD,
            allow_pass=True,
            labels={n: _card_label(n, Side.USA) for n in drawn},
        )
        if chosen is None:
            break
        drawn.remove(chosen)
        # Discarded without triggering the event.
        state.discard_card(chosen)
        state.release(chosen)
        state.note(f"Our Man In Tehran discards {chosen}", ctx.player)

    if drawn:
        state.shuffle_into_deck(drawn)
        state.release(*drawn)
        state.note(
            f"Our Man In Tehran returns {len(drawn)} card(s) and reshuffles the deck",
            ctx.player,
        )


# --------------------------------------------------------------------------- #
# #109 Yuri and Samantha
# --------------------------------------------------------------------------- #


@register("Yuri and Samantha")
def yuri_and_samantha(game, ctx):
    """1 VP to the USSR per US coup attempt for the rest of the turn."""
    game.state.add_effect(ctx.card, owner=Side.USSR, expires="end_of_turn")
    return (yield from nothing())


def yuri_and_samantha_on_coup(game, side: Side) -> None:
    """ENGINE HOOK: score Yuri and Samantha when the US attempts a coup.

    Belongs in ``Game.coup``, which every coup goes through. Not called anywhere yet,
    so the effect currently sits in play without paying out.
    """
    state = game.state
    if side is Side.USA and state.in_play("Yuri and Samantha", owner=Side.USSR):
        state.award_vp(Side.USSR, 1)
        state.note("Yuri and Samantha: US coup attempt, USSR +1 VP", side)


# --------------------------------------------------------------------------- #
# #110 AWACS Sale to Saudis
# --------------------------------------------------------------------------- #


@register("AWACS Sale to Saudis")
def awacs_sale_to_saudis(game, ctx):
    """2 US influence in Saudi Arabia; Muslim Revolution is dead as an event."""
    game.state.add_inf(Side.USA, "Saudi Arabia", 2)
    # The effect key is the card name, so Muslim Revolution's own predicate can read
    # state.in_play("AWACS Sale to Saudis").
    game.state.add_effect(ctx.card, owner=Side.USA)
    return (yield from nothing())

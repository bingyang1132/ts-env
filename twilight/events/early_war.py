"""Early War events, cards 1-35.

This module doubles as the reference for how events are written. The patterns:

* a handler is a generator taking ``(game, ctx)``; ``ctx.player`` is the side whose
  event it is and who makes its choices;
* anything that asks the player something uses ``yield from game.<helper>(...)``;
* a handler that asks nothing still has to be a generator, hence
  ``return (yield from nothing())``;
* a card that stays on the table calls ``state.add_effect(ctx.card, ...)``; the engine
  then leaves it in play instead of discarding it, and the effect name must equal the
  card name so ``state.in_play("NATO")`` works;
* conditional playability goes in ``register(playable_if=...)``, so the engine can omit
  the event option entirely rather than offering a no-op.
"""

from __future__ import annotations

from .. import rules
from ..data import CARDS, COUNTRY_ORDER
from ..enums import Phase, Region, Side
from . import register
from .helpers import (
    adjacent_to,
    ensure_control,
    in_region,
    nothing,
    uncontrolled,
    war,
    with_influence,
)

# --------------------------------------------------------------------------- #
# #4 Duck and Cover
# --------------------------------------------------------------------------- #


@register("Duck and Cover")
def duck_and_cover(game, ctx):
    """Degrade DEFCON one level, then the US scores 5 minus the new DEFCON level."""
    game.degrade_defcon(1, ctx.player)
    if not game.state.is_over:
        game.state.award_vp(Side.USA, 5 - game.state.defcon)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #5 Five-Year Plan
# --------------------------------------------------------------------------- #


@register("Five-Year Plan")
def five_year_plan(game, ctx):
    """The USSR discards a random card; if it is a US event, that event fires."""
    state = game.state
    hand = state.hand(Side.USSR)
    if not hand:
        state.note("Five-Year Plan: USSR hand is empty", ctx.player)
        return (yield from nothing())

    name = state.rng.choice(hand)
    hand.remove(name)
    discarded = CARDS[name]
    state.note(f"Five-Year Plan discards {name} at random", ctx.player)

    if discarded.side is Side.USA:
        # A US event fires immediately, resolved by the US player.
        yield from game._resolve_event(name, Side.USA)
        game._retire_card(name, Side.USA, as_event=True)
    else:
        state.discard_card(name)


# --------------------------------------------------------------------------- #
# #7 Socialist Governments
# --------------------------------------------------------------------------- #


def _socialist_governments_playable(game, side) -> bool:
    # Unplayable while The Iron Lady is in effect.
    return not game.state.has_resolved("The Iron Lady")


@register("Socialist Governments", playable_if=_socialist_governments_playable)
def socialist_governments(game, ctx):
    """Remove 3 US influence from Western Europe, at most 2 per country."""
    yield from game.remove_influence(
        Side.USA,
        3,
        allowed=in_region(Region.WESTERN_EUROPE),
        max_per_country=2,
        chooser=ctx.player,
        prompt="Socialist Governments: remove US influence in Western Europe (max 2 per country)",
    )


# --------------------------------------------------------------------------- #
# #8 Fidel
# --------------------------------------------------------------------------- #


@register("Fidel")
def fidel(game, ctx):
    """Remove all US influence in Cuba; the USSR gains control there."""
    game.state.clear_inf(Side.USA, "Cuba")
    ensure_control(game.state, Side.USSR, "Cuba")
    game.state.note("Fidel: USSR controls Cuba", ctx.player)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #9 Vietnam Revolts
# --------------------------------------------------------------------------- #


@register("Vietnam Revolts")
def vietnam_revolts(game, ctx):
    """2 USSR influence in Vietnam, and +1 ops on all-Southeast-Asia plays this turn."""
    game.state.add_inf(Side.USSR, "Vietnam", 2)
    game.state.add_effect(ctx.card, owner=Side.USSR, expires="end_of_turn")
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #10 Blockade
# --------------------------------------------------------------------------- #


@register("Blockade")
def blockade(game, ctx):
    """The US discards a card of 3+ ops or loses all influence in West Germany."""
    state = game.state
    candidates = [n for n in state.hand(Side.USA) if game.effective_ops(n, Side.USA) >= 3]

    if candidates:
        chosen = yield from game.choose_card(
            Side.USA,
            candidates,
            "Blockade: discard a card with 3 or more operations points, or pass to "
            "lose all US influence in West Germany",
            allow_pass=True,
        )
        if chosen is not None:
            state.hand(Side.USA).remove(chosen)
            state.discard_card(chosen)
            state.note(f"Blockade: US discards {chosen}", Side.USA)
            return

    removed = state.clear_inf(Side.USA, "West Germany")
    state.note(f"Blockade: US loses {removed} influence in West Germany", ctx.player)


# --------------------------------------------------------------------------- #
# #11 Korean War
# --------------------------------------------------------------------------- #


@register("Korean War")
def korean_war(game, ctx):
    """USSR invades South Korea: victory on 4-6, worth 2 VP and 2 military ops."""
    yield from war(
        game,
        Side.USSR,
        target="South Korea",
        victory_from=4,
        vp=2,
        military_ops=2,
    )


# --------------------------------------------------------------------------- #
# #12 Romanian Abdication
# --------------------------------------------------------------------------- #


@register("Romanian Abdication")
def romanian_abdication(game, ctx):
    """Remove all US influence in Romania; the USSR gains control there."""
    game.state.clear_inf(Side.USA, "Romania")
    ensure_control(game.state, Side.USSR, "Romania")
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #13 Arab-Israeli War
# --------------------------------------------------------------------------- #


def _arab_israeli_playable(game, side) -> bool:
    # Camp David Accords makes this card unplayable as an event.
    return not game.state.in_play("Camp David Accords")


@register("Arab-Israeli War", playable_if=_arab_israeli_playable)
def arab_israeli_war(game, ctx):
    """A pan-Arab coalition attacks Israel. Israel itself counts against the USSR."""
    state = game.state
    # Unusually, US control of Israel is itself a penalty, on top of adjacency.
    penalty = sum(1 for n in adjacent_to("Israel") if rules.controls(state, Side.USA, n))
    if rules.controls(state, Side.USA, "Israel"):
        penalty += 1

    roll = game.roll()
    modified = roll - penalty
    state.military_ops[Side.USSR] = min(5, state.military_ops[Side.USSR] + 2)

    if modified >= 4:
        state.award_vp(Side.USSR, 2)
        moved = state.clear_inf(Side.USA, "Israel")
        state.add_inf(Side.USSR, "Israel", moved)
        state.note(
            f"Arab-Israeli War: rolled {roll} -{penalty} = {modified}, USSR victory",
            ctx.player,
        )
    else:
        state.note(
            f"Arab-Israeli War: rolled {roll} -{penalty} = {modified}, no victory",
            ctx.player,
        )
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #14 COMECON
# --------------------------------------------------------------------------- #


@register("COMECON")
def comecon(game, ctx):
    """1 USSR influence in each of four non-US-controlled Eastern European countries."""
    pool = [
        n
        for n in in_region(Region.EASTERN_EUROPE)
        if not rules.controls(game.state, Side.USA, n)
    ]
    yield from game.add_influence(
        Side.USSR,
        4,
        allowed=pool,
        max_per_country=1,
        prompt="COMECON: 1 USSR influence in four different Eastern European countries",
    )


# --------------------------------------------------------------------------- #
# #15 Nasser
# --------------------------------------------------------------------------- #


@register("Nasser")
def nasser(game, ctx):
    """2 USSR influence in Egypt; the US loses half its influence there, rounded up."""
    state = game.state
    state.add_inf(Side.USSR, "Egypt", 2)
    present = state.inf(Side.USA, "Egypt")
    state.remove_inf(Side.USA, "Egypt", (present + 1) // 2)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #16 Warsaw Pact Formed
# --------------------------------------------------------------------------- #


@register("Warsaw Pact Formed")
def warsaw_pact_formed(game, ctx):
    """Either strip US influence from Eastern Europe, or add USSR influence there."""
    state = game.state
    choice = yield from game.choose_option(
        ctx.player,
        "Warsaw Pact Formed:",
        [
            "remove all US influence from four Eastern European countries",
            "add 5 USSR influence in Eastern Europe, at most 2 per country",
        ],
    )

    if choice == 0:
        pool = with_influence(state, Side.USA, in_region(Region.EASTERN_EUROPE))
        for remaining in range(4, 0, -1):
            pool = with_influence(state, Side.USA, in_region(Region.EASTERN_EUROPE))
            if not pool:
                break
            name = yield from game.choose_country(
                ctx.player,
                pool,
                f"Remove all US influence from an Eastern European country "
                f"({remaining} left)",
            )
            if name is None:
                break
            state.clear_inf(Side.USA, name)
    else:
        yield from game.add_influence(
            Side.USSR,
            5,
            allowed=in_region(Region.EASTERN_EUROPE),
            max_per_country=2,
            prompt="Warsaw Pact Formed: add USSR influence in Eastern Europe (max 2 each)",
        )

    # Either way the card stays in play, enabling NATO.
    state.add_effect(ctx.card, owner=Side.USSR)


# --------------------------------------------------------------------------- #
# #17 De Gaulle Leads France
# --------------------------------------------------------------------------- #


@register("De Gaulle Leads France")
def de_gaulle(game, ctx):
    """Remove 2 US influence in France, add 1 USSR; cancels NATO for France."""
    game.state.remove_inf(Side.USA, "France", 2)
    game.state.add_inf(Side.USSR, "France", 1)
    game.state.add_effect(ctx.card, owner=Side.USSR)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #18 Captured Nazi Scientist
# --------------------------------------------------------------------------- #


@register("Captured Nazi Scientist")
def captured_nazi_scientist(game, ctx):
    """Advance the player's space race marker one box."""
    from .. import spacerace

    state = game.state
    box = spacerace.next_box(state.space_race[ctx.player])
    if box is None:
        state.note("Captured Nazi Scientist: already at the end of the track", ctx.player)
        return (yield from nothing())

    gained = spacerace.victory_points(state.space_race, ctx.player, box.number)
    state.space_race[ctx.player] = box.number
    state.award_vp(ctx.player, gained)
    state.note(f"Captured Nazi Scientist: advanced to {box.name}, +{gained} VP", ctx.player)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #19 Truman Doctrine
# --------------------------------------------------------------------------- #


@register("Truman Doctrine")
def truman_doctrine(game, ctx):
    """Remove all USSR influence from one uncontrolled European country."""
    state = game.state
    pool = [
        n
        for n in uncontrolled(state, in_region(Region.EUROPE))
        if state.inf(Side.USSR, n) > 0
    ]
    name = yield from game.choose_country(
        ctx.player,
        pool,
        "Truman Doctrine: remove all USSR influence from an uncontrolled European country",
    )
    if name is not None:
        removed = state.clear_inf(Side.USSR, name)
        state.note(f"Truman Doctrine: {name} loses {removed} USSR influence", ctx.player)


# --------------------------------------------------------------------------- #
# #21 NATO
# --------------------------------------------------------------------------- #


def _nato_playable(game, side) -> bool:
    # Requires the Marshall Plan or the Warsaw Pact to have been played.
    return game.state.has_resolved("Marshall Plan") or game.state.has_resolved(
        "Warsaw Pact Formed"
    )


@register("NATO", playable_if=_nato_playable)
def nato(game, ctx):
    """US-controlled European countries become immune to coups and realignment."""
    game.state.add_effect(ctx.card, owner=Side.USA)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #22 Independent Reds
# --------------------------------------------------------------------------- #


@register("Independent Reds")
def independent_reds(game, ctx):
    """Match USSR influence in one of the Yugoslav-bloc countries."""
    state = game.state
    pool = ["Yugoslavia", "Romania", "Bulgaria", "Hungary", "Czechoslovakia"]
    name = yield from game.choose_country(
        ctx.player,
        [n for n in pool if state.inf(Side.USSR, n) > state.inf(Side.USA, n)],
        "Independent Reds: match USSR influence in one country",
    )
    if name is not None:
        state.set_inf(Side.USA, name, state.inf(Side.USSR, name))
        state.note(f"Independent Reds: US matches USSR influence in {name}", ctx.player)


# --------------------------------------------------------------------------- #
# #23 Marshall Plan
# --------------------------------------------------------------------------- #


@register("Marshall Plan")
def marshall_plan(game, ctx):
    """1 US influence in each of seven non-USSR-controlled Western European countries."""
    pool = [
        n
        for n in in_region(Region.WESTERN_EUROPE)
        if not rules.controls(game.state, Side.USSR, n)
    ]
    yield from game.add_influence(
        Side.USA,
        7,
        allowed=pool,
        max_per_country=1,
        prompt="Marshall Plan: 1 US influence in seven different Western European countries",
    )
    game.state.add_effect(ctx.card, owner=Side.USA)


# --------------------------------------------------------------------------- #
# #24 Indo-Pakistani War
# --------------------------------------------------------------------------- #


@register("Indo-Pakistani War")
def indo_pakistani_war(game, ctx):
    """Invade India or Pakistan; victory on 4-6 for 2 VP and 2 military ops."""
    target = yield from game.choose_country(
        ctx.player, ["India", "Pakistan"], "Indo-Pakistani War: choose the country invaded"
    )
    yield from war(
        game, ctx.player, target=target, victory_from=4, vp=2, military_ops=2
    )


# --------------------------------------------------------------------------- #
# #25 Containment
# --------------------------------------------------------------------------- #


@register("Containment")
def containment(game, ctx):
    """All US cards get +1 operations point for the rest of the turn."""
    game.state.add_effect(ctx.card, owner=Side.USA, expires="end_of_turn")
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #26 CIA Created
# --------------------------------------------------------------------------- #


@register("CIA Created")
def cia_created(game, ctx):
    """Reveal the USSR hand, then the US conducts operations as if playing a 1-op card."""
    state = game.state
    state.note(
        "CIA Created: USSR hand revealed -- " + ", ".join(sorted(state.hand(Side.USSR))),
        ctx.player,
    )
    state.add_effect(ctx.card, owner=Side.USA, expires="end_of_turn", reveal=True)
    yield from game.free_operations(Side.USA, 1)


# --------------------------------------------------------------------------- #
# #27 US/Japan Mutual Defense Pact
# --------------------------------------------------------------------------- #


@register("US/Japan Mutual Defense Pact")
def us_japan_pact(game, ctx):
    """The US gains control of Japan, which the USSR may never coup or realign."""
    ensure_control(game.state, Side.USA, "Japan")
    game.state.add_effect(ctx.card, owner=Side.USA)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #28 Suez Crisis
# --------------------------------------------------------------------------- #


@register("Suez Crisis")
def suez_crisis(game, ctx):
    """Remove 4 US influence from France, the UK and Israel, at most 2 per country."""
    yield from game.remove_influence(
        Side.USA,
        4,
        allowed=["France", "UK", "Israel"],
        max_per_country=2,
        chooser=ctx.player,
        prompt="Suez Crisis: remove US influence from France, the UK or Israel (max 2 each)",
    )


# --------------------------------------------------------------------------- #
# #29 East European Unrest
# --------------------------------------------------------------------------- #


@register("East European Unrest")
def east_european_unrest(game, ctx):
    """Remove USSR influence from three Eastern European countries.

    One point each in the Early and Mid War, two each in the Late War.
    """
    per_country = 2 if game.state.turn >= 8 else 1
    yield from game.remove_influence(
        Side.USSR,
        3 * per_country,
        allowed=in_region(Region.EASTERN_EUROPE),
        max_per_country=per_country,
        chooser=ctx.player,
        prompt=(
            f"East European Unrest: remove {per_country} USSR influence from each of "
            "three Eastern European countries"
        ),
    )


# --------------------------------------------------------------------------- #
# #30 Decolonization
# --------------------------------------------------------------------------- #


@register("Decolonization")
def decolonization(game, ctx):
    """1 USSR influence in each of four African or Southeast Asian countries."""
    pool = list(in_region(Region.AFRICA)) + list(in_region(Region.SOUTHEAST_ASIA))
    yield from game.add_influence(
        Side.USSR,
        4,
        allowed=pool,
        max_per_country=1,
        prompt="Decolonization: 1 USSR influence in four different African or SE Asian countries",
    )


# --------------------------------------------------------------------------- #
# #31 Red Scare/Purge
# --------------------------------------------------------------------------- #


@register("Red Scare/Purge")
def red_scare_purge(game, ctx):
    """All of the opponent's cards are worth one less operations point this turn."""
    game.state.add_effect(ctx.card, owner=ctx.player, expires="end_of_turn")
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #33 De-Stalinization
# --------------------------------------------------------------------------- #


@register("De-Stalinization")
def destalinization(game, ctx):
    """Relocate up to 4 USSR influence to countries nobody controls, max 2 per country."""
    state = game.state
    sources = with_influence(state, Side.USSR)
    taken = yield from game.remove_influence(
        Side.USSR,
        4,
        allowed=sources,
        chooser=ctx.player,
        prompt="De-Stalinization: remove USSR influence to relocate (up to 4)",
    )
    moved = sum(taken.values())
    if not moved:
        return

    yield from game.add_influence(
        Side.USSR,
        moved,
        allowed=uncontrolled(state, COUNTRY_ORDER),
        max_per_country=2,
        prompt=f"De-Stalinization: place {moved} influence in uncontrolled countries (max 2 each)",
    )


# --------------------------------------------------------------------------- #
# #34 Nuclear Test Ban
# --------------------------------------------------------------------------- #


@register("Nuclear Test Ban")
def nuclear_test_ban(game, ctx):
    """Score DEFCON minus 2 in VP, then degrade DEFCON two levels."""
    state = game.state
    state.award_vp(ctx.player, state.defcon - 2)
    game.degrade_defcon(2, ctx.player)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #35 Formosan Resolution
# --------------------------------------------------------------------------- #


@register("Formosan Resolution")
def formosan_resolution(game, ctx):
    """Taiwan counts as a battleground for Asia scoring while the US controls it."""
    game.state.add_effect(ctx.card, owner=Side.USA)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #103 Defectors
# --------------------------------------------------------------------------- #


@register("Defectors")
def defectors(game, ctx):
    """Cancels the USSR headline; played in an action round it costs the USSR 1 VP.

    The headline cancellation is handled in the headline phase itself, since it has to
    pre-empt the USSR event rather than follow it.
    """
    state = game.state
    if state.phase is Phase.HEADLINE:
        return (yield from nothing())
    if ctx.triggered_by_opponent:
        # The USSR played it for operations: the US scores 1 VP.
        state.award_vp(Side.USA, 1)
        state.note("Defectors played by the USSR for operations: US +1 VP", ctx.player)
    return (yield from nothing())

"""Late War events, cards 82-102.

Conventions are the ones documented in :mod:`twilight.events.early_war`.

Two clauses in this range need engine support that does not exist yet and are
therefore recorded in the effect payload but not enforced:

* North Sea Oil's extra action round -- ``action_rounds_this_turn`` is read once per
  turn and is shared by both players, so there is no per-side hook to raise;
* Chernobyl's placement ban -- :meth:`twilight.engine.Game.spend_operations_influence`
  does not consult the effect.
"""

from __future__ import annotations

from .. import rules
from ..data import CARDS
from ..decisions import Decision, DecisionType, region_action
from ..enums import OpsUse, Region, Side, WinReason
from . import is_playable, register
from .helpers import adjacent_to, free_coup, in_region, nothing, war

#: The regions a player may name for Chernobyl: the six that scoring cards use.
_NAMEABLE_REGIONS = (
    Region.EUROPE,
    Region.ASIA,
    Region.MIDDLE_EAST,
    Region.AFRICA,
    Region.CENTRAL_AMERICA,
    Region.SOUTH_AMERICA,
)


def _choose_region(game, side, prompt, regions=_NAMEABLE_REGIONS):
    """Ask *side* to name one region. Local to this module; there is no shared helper."""
    action = yield from game.ask(
        Decision(
            DecisionType.CHOOSE_REGION,
            side,
            prompt,
            tuple(region_action(r, r.value) for r in regions),
        )
    )
    return Region(action.value)


def _free_realignments(game, side, rolls, *, allowed, prompt):
    """Realignment rolls granted by an event, restricted to *allowed* countries."""
    pool = list(allowed)
    for remaining in range(rolls, 0, -1):
        targets = [n for n in pool if rules.can_realign(game.state, side, n)]
        if not targets:
            return
        name = yield from game.choose_country(
            side,
            targets,
            f"{prompt} ({remaining} roll(s) left)",
            dtype=DecisionType.REALIGN_TARGET,
            allow_pass=True,
            remaining=remaining,
        )
        if name is None:
            return
        rules.resolve_realignment(game.state, side, name)


def _event_side(name: str, default: Side) -> Side:
    """Who resolves *name*'s event: its owner, or *default* for a neutral card."""
    owner = CARDS[name].side
    return default if owner is None else owner


# --------------------------------------------------------------------------- #
# #82 Iranian Hostage Crisis
# --------------------------------------------------------------------------- #


@register("Iranian Hostage Crisis")
def iranian_hostage_crisis(game, ctx):
    """Remove all US influence in Iran, add 2 USSR; doubles Terrorism against the US."""
    game.state.clear_inf(Side.USA, "Iran")
    game.state.add_inf(Side.USSR, "Iran", 2)
    game.state.add_effect(ctx.card, owner=Side.USSR)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #83 The Iron Lady
# --------------------------------------------------------------------------- #


@register("The Iron Lady")
def the_iron_lady(game, ctx):
    """US +1 VP, 1 USSR influence in Argentina, USSR swept out of the UK."""
    state = game.state
    state.award_vp(Side.USA, 1)
    state.add_inf(Side.USSR, "Argentina", 1)
    state.clear_inf(Side.USSR, "UK")
    # Stays in play: Socialist Governments can no longer be played as an event.
    state.add_effect(ctx.card, owner=Side.USA)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #84 Reagan Bombs Libya
# --------------------------------------------------------------------------- #


@register("Reagan Bombs Libya")
def reagan_bombs_libya(game, ctx):
    """The US scores 1 VP for every 2 USSR influence in Libya."""
    state = game.state
    gained = state.inf(Side.USSR, "Libya") // 2
    state.award_vp(Side.USA, gained)
    state.note(f"Reagan Bombs Libya: US +{gained} VP", ctx.player)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #85 Star Wars
# --------------------------------------------------------------------------- #


def _star_wars_targets(game, side) -> list[str]:
    # Star Wars itself is excluded: played for operations it lands in the discard, and
    # looking at its own playability from here would recurse.
    return [
        name
        for name in game.state.discard
        if name != "Star Wars"
        and not CARDS[name].is_scoring
        and is_playable(game, name, _event_side(name, side))
    ]


def _star_wars_playable(game, side) -> bool:
    # Only while strictly ahead on the space race, and only with something to take.
    state = game.state
    if state.space_race[side] <= state.space_race[side.opponent]:
        return False
    return bool(_star_wars_targets(game, side))


@register("Star Wars", playable_if=_star_wars_playable)
def star_wars(game, ctx):
    """Take a non-scoring card out of the discard pile; its event occurs at once."""
    state = game.state
    # Headlines bypass playable_if, so the space race lead is re-checked here.
    candidates = (
        _star_wars_targets(game, ctx.player)
        if _star_wars_playable(game, ctx.player)
        else []
    )
    chosen = yield from game.choose_card(
        ctx.player,
        candidates,
        "Star Wars: choose a non-scoring card from the discard pile to resolve",
        labels={
            n: f"{CARDS[n].ops}op, {_event_side(n, ctx.player).label} event"
            for n in candidates
        },
    )
    if chosen is None:
        return

    state.discard.remove(chosen)
    actor = _event_side(chosen, ctx.player)
    state.note(f"Star Wars resolves {chosen} from the discard pile", ctx.player)
    yield from game._resolve_event(chosen, actor)
    game._retire_card(chosen, actor, as_event=True)


# --------------------------------------------------------------------------- #
# #86 North Sea Oil
# --------------------------------------------------------------------------- #


@register("North Sea Oil")
def north_sea_oil(game, ctx):
    """OPEC is no longer playable, and the US takes an eighth action round this turn."""
    state = game.state
    # The extra action round is recorded for the turn it was played; the engine has no
    # per-side action-round hook, so only the OPEC clause is actually enforced.
    state.add_effect(ctx.card, owner=Side.USA, extra_action_round_turn=state.turn)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #87 The Reformer
# --------------------------------------------------------------------------- #


@register("The Reformer")
def the_reformer(game, ctx):
    """4 USSR influence in Europe, 6 while ahead on VP; no more USSR coups in Europe."""
    state = game.state
    points = 6 if state.vp_for(ctx.player) > 0 else 4
    yield from game.add_influence(
        Side.USSR,
        points,
        allowed=in_region(Region.EUROPE),
        max_per_country=2,
        prompt=f"The Reformer: add {points} USSR influence in Europe (max 2 per country)",
    )
    state.add_effect(ctx.card, owner=Side.USSR)


# --------------------------------------------------------------------------- #
# #88 Marine Barracks Bombing
# --------------------------------------------------------------------------- #


@register("Marine Barracks Bombing")
def marine_barracks_bombing(game, ctx):
    """All US influence in Lebanon, plus 2 more from anywhere in the Middle East."""
    state = game.state
    removed = state.clear_inf(Side.USA, "Lebanon")
    state.note(f"Marine Barracks Bombing: Lebanon loses {removed} US influence", ctx.player)
    yield from game.remove_influence(
        Side.USA,
        2,
        allowed=in_region(Region.MIDDLE_EAST),
        chooser=ctx.player,
        prompt="Marine Barracks Bombing: remove 2 more US influence in the Middle East",
    )


# --------------------------------------------------------------------------- #
# #89 Soviets Shoot Down KAL-007
# --------------------------------------------------------------------------- #


@register("Soviets Shoot Down KAL-007")
def kal_007(game, ctx):
    """DEFCON down one, US +2 VP, plus 4 ops of operations if it holds South Korea."""
    state = game.state
    game.degrade_defcon(1, ctx.player)
    if state.is_over:
        return
    state.award_vp(Side.USA, 2)

    if rules.controls(state, Side.USA, "South Korea"):
        yield from game.free_operations(
            Side.USA,
            4,
            uses=(OpsUse.INFLUENCE, OpsUse.REALIGN),
            prompt="Soviets Shoot Down KAL-007: place influence or realign as a 4 Op card",
        )


# --------------------------------------------------------------------------- #
# #90 Glasnost
# --------------------------------------------------------------------------- #


@register("Glasnost")
def glasnost(game, ctx):
    """USSR +2 VP and DEFCON up one; with The Reformer out, also 4 ops of operations."""
    state = game.state
    state.award_vp(Side.USSR, 2)
    game.improve_defcon(1)

    if state.in_play("The Reformer"):
        yield from game.free_operations(
            Side.USSR,
            4,
            uses=(OpsUse.INFLUENCE, OpsUse.REALIGN),
            prompt="Glasnost: place influence or realign as a 4 Op card",
        )


# --------------------------------------------------------------------------- #
# #91 Ortega Elected in Nicaragua
# --------------------------------------------------------------------------- #


@register("Ortega Elected in Nicaragua")
def ortega_elected(game, ctx):
    """Clear the US out of Nicaragua, then coup a neighbour of it for free."""
    state = game.state
    removed = state.clear_inf(Side.USA, "Nicaragua")
    state.note(f"Ortega Elected: Nicaragua loses {removed} US influence", ctx.player)

    ops = game.effective_ops(ctx.card, Side.USSR)
    yield from free_coup(
        game,
        Side.USSR,
        ops,
        allowed=adjacent_to("Nicaragua"),
        prompt=f"Ortega Elected: free coup with {ops} ops adjacent to Nicaragua",
    )


# --------------------------------------------------------------------------- #
# #92 Terrorism
# --------------------------------------------------------------------------- #


@register("Terrorism")
def terrorism(game, ctx):
    """The opponent discards one card at random, or two against a US hand after #82."""
    state = game.state
    victim = ctx.player.opponent
    doubled = ctx.player is Side.USSR and state.has_resolved("Iranian Hostage Crisis")
    count = 2 if doubled else 1

    for _ in range(count):
        hand = state.hand(victim)
        if not hand:
            state.note(f"Terrorism: {victim.label} hand is empty", ctx.player)
            break
        name = state.rng.choice(hand)
        hand.remove(name)
        state.discard_card(name)
        # The event on a discarded card does not occur.
        state.note(f"Terrorism: {victim.label} discards {name} at random", ctx.player)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #93 Iran-Contra Scandal
# --------------------------------------------------------------------------- #


@register("Iran-Contra Scandal")
def iran_contra_scandal(game, ctx):
    """US realignment rolls are at -1 for the rest of the turn."""
    game.state.add_effect(ctx.card, owner=Side.USSR, expires="end_of_turn")
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #94 Chernobyl
# --------------------------------------------------------------------------- #


@register("Chernobyl")
def chernobyl(game, ctx):
    """The US names a region the USSR may not place operations influence into."""
    state = game.state
    region = yield from _choose_region(
        game,
        Side.USA,
        "Chernobyl: name the region the USSR may not place influence in this turn",
    )
    state.add_effect(
        ctx.card, owner=Side.USA, expires="end_of_turn", region=region.value
    )
    state.note(f"Chernobyl: USSR barred from placing influence in {region}", ctx.player)


# --------------------------------------------------------------------------- #
# #95 Latin American Debt Crisis
# --------------------------------------------------------------------------- #


@register("Latin American Debt Crisis")
def latin_american_debt_crisis(game, ctx):
    """The US discards a 3+ ops card, or the USSR doubles up in two South American ones."""
    state = game.state
    candidates = [n for n in state.hand(Side.USA) if game.effective_ops(n, Side.USA) >= 3]

    if candidates:
        chosen = yield from game.choose_card(
            Side.USA,
            candidates,
            "Latin American Debt Crisis: discard a card with 3 or more operations "
            "points, or pass to let the USSR double its South American influence",
            allow_pass=True,
        )
        if chosen is not None:
            state.hand(Side.USA).remove(chosen)
            state.discard_card(chosen)
            state.note(f"Latin American Debt Crisis: US discards {chosen}", Side.USA)
            return

    doubled: list[str] = []
    for remaining in (2, 1):
        pool = [
            n
            for n in in_region(Region.SOUTH_AMERICA)
            if state.inf(Side.USSR, n) > 0 and n not in doubled
        ]
        name = yield from game.choose_country(
            ctx.player,
            pool,
            f"Latin American Debt Crisis: double USSR influence ({remaining} left)",
        )
        if name is None:
            break
        state.add_inf(Side.USSR, name, state.inf(Side.USSR, name))
        doubled.append(name)

    if doubled:
        state.note(
            "Latin American Debt Crisis doubles USSR influence in " + ", ".join(doubled),
            ctx.player,
        )


# --------------------------------------------------------------------------- #
# #96 Tear Down This Wall
# --------------------------------------------------------------------------- #


@register("Tear Down This Wall")
def tear_down_this_wall(game, ctx):
    """Cancel Willy Brandt, 3 US influence in East Germany, then free ops in Europe."""
    state = game.state
    if state.remove_effect("Willy Brandt") is not None:
        state.note("Tear Down This Wall cancels Willy Brandt", ctx.player)
    state.add_inf(Side.USA, "East Germany", 3)
    state.add_effect(ctx.card, owner=Side.USA)

    ops = game.effective_ops(ctx.card, Side.USA)
    europe = in_region(Region.EUROPE)
    choices: list[tuple[str, str]] = []
    if any(rules.can_coup(state, Side.USA, n) for n in europe):
        choices.append(("coup", f"free coup attempt in Europe with {ops} ops"))
    if any(rules.can_realign(state, Side.USA, n) for n in europe):
        choices.append(("realign", f"{ops} realignment rolls in Europe"))
    choices.append(("none", "no coup or realignment"))

    pick = yield from game.choose_option(
        Side.USA, "Tear Down This Wall:", [label for _, label in choices]
    )
    kind = choices[pick][0]

    if kind == "coup":
        yield from free_coup(
            game,
            Side.USA,
            ops,
            allowed=europe,
            prompt=f"Tear Down This Wall: free coup with {ops} ops in Europe",
        )
    elif kind == "realign":
        yield from _free_realignments(
            game,
            Side.USA,
            ops,
            allowed=europe,
            prompt="Tear Down This Wall: realign in Europe",
        )


# --------------------------------------------------------------------------- #
# #97 An Evil Empire
# --------------------------------------------------------------------------- #


@register("An Evil Empire")
def an_evil_empire(game, ctx):
    """Cancel Flower Power and score the US 1 VP."""
    state = game.state
    if state.remove_effect("Flower Power") is not None:
        state.note("An Evil Empire cancels Flower Power", ctx.player)
    state.award_vp(Side.USA, 1)
    state.add_effect(ctx.card, owner=Side.USA)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #98 Aldrich Ames Remix
# --------------------------------------------------------------------------- #


@register("Aldrich Ames Remix")
def aldrich_ames_remix(game, ctx):
    """The US hand is exposed for the turn; the USSR picks one card to be discarded."""
    state = game.state
    hand = state.hand(Side.USA)
    state.note(
        "Aldrich Ames Remix: US hand revealed -- " + ", ".join(sorted(hand)), ctx.player
    )
    state.add_effect(ctx.card, owner=Side.USSR, expires="end_of_turn", reveal=True)

    chosen = yield from game.choose_card(
        Side.USSR,
        list(hand),
        "Aldrich Ames Remix: choose a card from the US hand to discard",
        dtype=DecisionType.DISCARD_CARD,
        labels={n: game._card_label(n, Side.USA) for n in hand},
    )
    if chosen is None:
        return
    hand.remove(chosen)
    state.discard_card(chosen)
    state.note(f"Aldrich Ames Remix: US discards {chosen}", ctx.player)


# --------------------------------------------------------------------------- #
# #99 Pershing II Deployed
# --------------------------------------------------------------------------- #


@register("Pershing II Deployed")
def pershing_ii_deployed(game, ctx):
    """USSR +1 VP, and 1 US influence off each of up to three Western European countries."""
    game.state.award_vp(Side.USSR, 1)
    yield from game.remove_influence(
        Side.USA,
        3,
        allowed=in_region(Region.WESTERN_EUROPE),
        max_per_country=1,
        chooser=ctx.player,
        prompt=(
            "Pershing II Deployed: remove 1 US influence from each of up to three "
            "Western European countries"
        ),
    )


# --------------------------------------------------------------------------- #
# #100 Wargames
# --------------------------------------------------------------------------- #


def _wargames_playable(game, side) -> bool:
    # Only at DEFCON 2, the brink of war.
    del side
    return game.state.defcon == 2


@register("Wargames", playable_if=_wargames_playable)
def wargames(game, ctx):
    """Hand the opponent 6 VP and end the game at once, with no final scoring."""
    state = game.state
    # The headline phase does not consult playable_if, so re-check the condition here
    # rather than end the game from a headline played at DEFCON 3 or better.
    if not _wargames_playable(game, ctx.player):
        state.note("Wargames: not at DEFCON 2, no effect", ctx.player)
        return (yield from nothing())

    state.award_vp(ctx.player.opponent, 6)

    if state.vp > 0:
        winner: Side | None = Side.USSR
    elif state.vp < 0:
        winner = Side.USA
    else:
        winner = None
    state.note(f"Wargames: {ctx.player.opponent.label} +6 VP, the game ends", ctx.player)
    state.finish(winner, WinReason.WARGAMES)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #101 Solidarity
# --------------------------------------------------------------------------- #


def _solidarity_playable(game, side) -> bool:
    # Needs John Paul II Elected Pope to have been played.
    del side
    return game.state.has_resolved("John Paul II Elected Pope")


@register("Solidarity", playable_if=_solidarity_playable)
def solidarity(game, ctx):
    """3 US influence in Poland, spending John Paul II Elected Pope."""
    state = game.state
    # Headlines bypass playable_if, so the precondition is re-checked here.
    if not _solidarity_playable(game, ctx.player):
        state.note("Solidarity: John Paul II Elected Pope has not been played", ctx.player)
        return (yield from nothing())

    state.add_inf(Side.USA, "Poland", 3)
    state.remove_effect("John Paul II Elected Pope")
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #102 Iran-Iraq War
# --------------------------------------------------------------------------- #


@register("Iran-Iraq War")
def iran_iraq_war(game, ctx):
    """Iran or Iraq invades the other: victory on 4-6 for 2 VP and 2 military ops."""
    target = yield from game.choose_country(
        ctx.player, ["Iran", "Iraq"], "Iran-Iraq War: choose the country invaded"
    )
    yield from war(
        game, ctx.player, target=target, victory_from=4, vp=2, military_ops=2
    )

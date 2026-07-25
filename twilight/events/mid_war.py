"""Mid War events, cards 36-58."""

from __future__ import annotations

from .. import rules
from ..data import CARDS, COUNTRIES
from ..decisions import DecisionType
from ..enums import Region, Side
from . import register, register_deferred
from .helpers import adjacent_to, free_coup, in_region, nothing, war

#: Regions a Summit roll counts: one bonus each for domination or control. The
#: sub-regions are excluded, since only these six have their own presence tiers.
_SUMMIT_REGIONS = (
    Region.EUROPE,
    Region.ASIA,
    Region.MIDDLE_EAST,
    Region.AFRICA,
    Region.CENTRAL_AMERICA,
    Region.SOUTH_AMERICA,
)

#: Every battleground on the board, for Kitchen Debates.
_BATTLEGROUNDS = tuple(name for name, c in COUNTRIES.items() if c.battleground)

#: The eight countries Muslim Revolution may strip of US influence.
_MUSLIM_REVOLUTION = (
    "Sudan",
    "Iran",
    "Iraq",
    "Egypt",
    "Libya",
    "Saudi Arabia",
    "Syria",
    "Jordan",
)

#: Central and South America together, the reach of Junta.
_LATIN_AMERICA = tuple(in_region(Region.CENTRAL_AMERICA)) + tuple(
    in_region(Region.SOUTH_AMERICA)
)


def _free_realignments(game, side, ops, *, allowed, prompt):
    """Realignment rolls granted by an event, restricted to *allowed*.

    The engine's own realignment loop covers the whole board, so the region-limited
    cards need their own.
    """
    for remaining in range(ops, 0, -1):
        targets = [n for n in allowed if rules.can_realign(game.state, side, n)]
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


def _controlled_battlegrounds(state, side) -> int:
    return sum(1 for n in _BATTLEGROUNDS if rules.controls(state, side, n))


# --------------------------------------------------------------------------- #
# #36 Brush War
# --------------------------------------------------------------------------- #


@register("Brush War")
def brush_war(game, ctx):
    """Attack a stability 1-2 country: victory on 3-6 for 1 VP and 3 military ops."""
    pool = [
        n
        for n in COUNTRIES
        if not COUNTRIES[n].superpower and COUNTRIES[n].stability <= 2
    ]
    target = yield from game.choose_country(
        ctx.player, pool, "Brush War: choose a country with stability 1 or 2 to attack"
    )
    yield from war(
        game,
        ctx.player,
        target=target,
        victory_from=3,
        vp=1,
        military_ops=3,
    )


# --------------------------------------------------------------------------- #
# #39 Arms Race
# --------------------------------------------------------------------------- #


@register("Arms Race")
def arms_race(game, ctx):
    """1 VP for leading the military operations track, 3 if the requirement is met."""
    state = game.state
    mine = state.military_ops[ctx.player]
    theirs = state.military_ops[ctx.player.opponent]

    if mine <= theirs:
        state.note(f"Arms Race: military ops {mine} vs {theirs}, no VP", ctx.player)
        return (yield from nothing())

    # The required amount is the current DEFCON level.
    gained = 3 if mine >= state.defcon else 1
    state.award_vp(ctx.player, gained)
    state.note(
        f"Arms Race: military ops {mine} vs {theirs}, required {state.defcon}, "
        f"+{gained} VP",
        ctx.player,
    )
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #40 Cuban Missile Crisis
# --------------------------------------------------------------------------- #


@register("Cuban Missile Crisis")
def cuban_missile_crisis(game, ctx):
    """DEFCON drops to 2; any further coup by the opponent this turn loses them the game.

    The opponent may cancel it by removing 2 influence from Cuba (USSR) or from West
    Germany or Turkey (US). Both halves need engine hooks that do not exist yet, so the
    effect is only recorded here.
    """
    state = game.state
    game.degrade_defcon(state.defcon - 2, ctx.player)
    if state.is_over:
        return (yield from nothing())

    state.add_effect(ctx.card, owner=ctx.player, expires="end_of_turn")
    state.note(
        "Cuban Missile Crisis: DEFCON 2, any further coup by the opponent loses them "
        "the game unless they cancel it",
        ctx.player,
    )
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #41 Nuclear Subs
# --------------------------------------------------------------------------- #


@register("Nuclear Subs")
def nuclear_subs(game, ctx):
    """US coups in battlegrounds do not degrade DEFCON for the rest of the turn."""
    game.state.add_effect(ctx.card, owner=Side.USA, expires="end_of_turn")
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #42 Quagmire
# --------------------------------------------------------------------------- #


@register("Quagmire")
def quagmire(game, ctx):
    """The US must discard a 2+ ops card and roll 1-4 each round to escape.

    The discard-and-escape loop lives in the engine's action round; this only puts the
    trap on the US player and cancels NORAD.
    """
    state = game.state
    state.add_effect(ctx.card, owner=Side.USA)
    if state.remove_effect("NORAD") is not None:
        state.note("Quagmire: NORAD leaves play", ctx.player)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #43 Salt Negotiations
# --------------------------------------------------------------------------- #


@register("Salt Negotiations")
def salt_negotiations(game, ctx):
    """Improve DEFCON two levels, coups at -1 this turn, and reclaim a discarded card."""
    state = game.state
    game.improve_defcon(2)

    pool = [n for n in state.discard if not CARDS[n].is_scoring]
    chosen = yield from game.choose_card(
        ctx.player,
        pool,
        "Salt Negotiations: reclaim one non-scoring card from the discard pile",
        allow_pass=True,
    )
    if chosen is not None:
        state.discard.remove(chosen)
        state.hand(ctx.player).append(chosen)
        state.note(f"Salt Negotiations: {chosen} reclaimed from the discard pile", ctx.player)

    state.add_effect(ctx.card, owner=ctx.player, expires="end_of_turn")


# --------------------------------------------------------------------------- #
# #44 Bear Trap
# --------------------------------------------------------------------------- #


@register("Bear Trap")
def bear_trap(game, ctx):
    """The USSR must discard a 2+ ops card and roll 1-4 each round to escape."""
    game.state.add_effect(ctx.card, owner=Side.USSR)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #45 Summit
# --------------------------------------------------------------------------- #


@register("Summit")
def summit(game, ctx):
    """Both roll, +1 per region dominated or controlled; the winner takes 2 VP."""
    state = game.state
    rolls: dict[Side, int] = {}
    totals: dict[Side, int] = {}

    for side in Side:
        bonus = sum(
            1
            for region in _SUMMIT_REGIONS
            if rules.region_status(state, region).tier(side) in ("domination", "control")
        )
        rolls[side] = game.roll()
        totals[side] = rolls[side] + bonus

    detail = " vs ".join(
        f"{s.label} {rolls[s]}+{totals[s] - rolls[s]}={totals[s]}" for s in Side
    )
    if totals[Side.USSR] == totals[Side.USA]:
        state.note(f"Summit: {detail}, tied and nothing happens", ctx.player)
        return

    winner = Side.USSR if totals[Side.USSR] > totals[Side.USA] else Side.USA
    state.award_vp(winner, 2)
    state.note(f"Summit: {detail}, {winner.label} wins and gains 2 VP", ctx.player)

    labels = ["leave DEFCON alone", "degrade DEFCON one level"]
    if state.defcon < 5:
        labels.append("improve DEFCON one level")
    choice = yield from game.choose_option(winner, "Summit: the winner may move DEFCON", labels)
    if choice == 1:
        game.degrade_defcon(1, winner)
    elif choice == 2:
        game.improve_defcon(1)


# --------------------------------------------------------------------------- #
# #46 How I Learned to Stop Worrying
# --------------------------------------------------------------------------- #


@register("How I Learned to Stop Worrying")
def how_i_learned_to_stop_worrying(game, ctx):
    """Set DEFCON to any level, and count 5 military operations.

    Choosing DEFCON 1 is legal and loses the game for the player who does it.
    """
    state = game.state
    choice = yield from game.choose_option(
        ctx.player,
        "How I Learned to Stop Worrying: set DEFCON to",
        [f"DEFCON {level}" for level in range(1, 6)],
    )
    level = choice + 1

    if level < state.defcon:
        game.degrade_defcon(state.defcon - level, ctx.player)
    else:
        game.improve_defcon(level - state.defcon)

    state.military_ops[ctx.player] = min(5, state.military_ops[ctx.player] + 5)
    state.note("How I Learned to Stop Worrying counts as 5 military operations", ctx.player)


# --------------------------------------------------------------------------- #
# #47 Junta
# --------------------------------------------------------------------------- #


@register("Junta")
def junta(game, ctx):
    """2 influence in one Latin American country, then a free coup or realignment there."""
    state = game.state
    name = yield from game.choose_country(
        ctx.player,
        _LATIN_AMERICA,
        "Junta: place 2 influence in one Central or South American country",
    )
    if name is not None:
        state.add_inf(ctx.player, name, 2)
        state.note(f"Junta: {name} +2 {ctx.player.label} influence", ctx.player)

    ops = game.effective_ops(ctx.card, ctx.player)
    labels: list[str] = []
    kinds: list[str | None] = []
    if any(rules.can_coup(state, ctx.player, n) for n in _LATIN_AMERICA):
        labels.append(f"free coup attempt with {ops} ops")
        kinds.append("coup")
    if any(rules.can_realign(state, ctx.player, n) for n in _LATIN_AMERICA):
        labels.append(f"{ops} free realignment rolls")
        kinds.append("realign")
    labels.append("neither")
    kinds.append(None)

    choice = yield from game.choose_option(
        ctx.player, "Junta: free operations in Central or South America", labels
    )
    if kinds[choice] == "coup":
        yield from free_coup(
            game,
            ctx.player,
            ops,
            allowed=_LATIN_AMERICA,
            prompt="Junta: free coup attempt in Central or South America",
        )
    elif kinds[choice] == "realign":
        yield from _free_realignments(
            game,
            ctx.player,
            ops,
            allowed=_LATIN_AMERICA,
            prompt="Junta: free realignment in Central or South America",
        )


# --------------------------------------------------------------------------- #
# #48 Kitchen Debates
# --------------------------------------------------------------------------- #


def _kitchen_debates_playable(game, side) -> bool:
    # Only worth anything while the US leads on controlled battlegrounds.
    state = game.state
    return _controlled_battlegrounds(state, Side.USA) > _controlled_battlegrounds(
        state, Side.USSR
    )


@register("Kitchen Debates", playable_if=_kitchen_debates_playable)
def kitchen_debates(game, ctx):
    """The US scores 2 VP for controlling more battlegrounds than the USSR."""
    state = game.state
    state.award_vp(Side.USA, 2)
    state.note(
        "Kitchen Debates: US leads on controlled battlegrounds, +2 VP", ctx.player
    )
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #49 Missile Envy
# --------------------------------------------------------------------------- #


@register("Missile Envy")
def missile_envy(game, ctx):
    """Take the opponent's highest-ops card; this card passes to them.

    A card carrying the player's own event, or a neutral one, fires at once; a card
    carrying the opponent's event is used for its operations instead.
    """
    state = game.state
    opponent = ctx.player.opponent

    pool = [n for n in state.hand(opponent) if CARDS[n].ops > 0]
    if not pool:
        state.note("Missile Envy: the opponent holds no operations card", ctx.player)
        return (yield from nothing())

    # Printed operations value decides which card is highest; the opponent breaks ties.
    best = max(CARDS[n].ops for n in pool)
    tied = [n for n in pool if CARDS[n].ops == best]
    given = yield from game.choose_card(
        opponent, tied, f"Missile Envy: hand over one of your {best} ops cards"
    )
    assert given is not None
    state.hand(opponent).remove(given)
    # Held outside every pile until it is used and retired below.
    state.hold(given)

    # Missile Envy itself changes hands rather than being discarded.
    state.hand(opponent).append(ctx.card)
    state.add_effect(ctx.card, owner=opponent, expires="end_of_turn", do_not_discard=True)
    state.note(f"Missile Envy: exchanged for {given}", ctx.player)

    # The opponent is compelled to spend Missile Envy itself for operations on their
    # next action round.
    state.must_play[int(opponent)] = ctx.card

    card = CARDS[given]
    if card.event_belongs_to(ctx.player) and game._event_is_playable(given, ctx.player):
        yield from game._resolve_event(given, ctx.player)
        game._retire_card(given, ctx.player, as_event=True)
    else:
        ops = game.effective_ops(given, ctx.player)
        yield from game.free_operations(
            ctx.player, ops, prompt=f"Missile Envy: use {given} for {ops} ops"
        )
        game._retire_card(given, ctx.player, as_event=False)
    # Released on the normal path only, so collecting an abandoned game cannot mutate
    # a state somebody is still reading.
    state.release(given)


# --------------------------------------------------------------------------- #
# #50 We Will Bury You
# --------------------------------------------------------------------------- #


#: Deferred-trigger kind for We Will Bury You's victory points.
WE_WILL_BURY_YOU = "we_will_bury_you_vp"


@register("We Will Bury You")
def we_will_bury_you(game, ctx):
    """Degrade DEFCON one level; the USSR scores 3 VP unless the US plays UN Intervention.

    The victory points are scheduled for the end of the US player's next action round
    rather than awarded now, so that playing UN Intervention as an event in between
    cancels them -- which is what the card says.
    """
    game.degrade_defcon(1, ctx.player)
    if game.state.is_over:
        return (yield from nothing())

    game.state.defer(
        ctx.card, WE_WILL_BURY_YOU, player=Side.USA, when="end", vp=3
    )
    game.state.note(
        "We Will Bury You: USSR scores 3 VP at the end of the next US action round "
        "unless UN Intervention is played as an event first",
        ctx.player,
    )
    return (yield from nothing())


@register_deferred(WE_WILL_BURY_YOU)
def _we_will_bury_you_vp(game, trigger):
    """Award the victory points the US failed to head off."""
    vp = trigger.data.get("vp", 3)
    game.state.award_vp(Side.USSR, vp)
    game.state.note(f"We Will Bury You resolves: USSR +{vp} VP")
    game._check_victory_points()
    return (yield from nothing())
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #51 Brezhnev Doctrine
# --------------------------------------------------------------------------- #


@register("Brezhnev Doctrine")
def brezhnev_doctrine(game, ctx):
    """All USSR cards get +1 operations point for the rest of the turn."""
    game.state.add_effect(ctx.card, owner=Side.USSR, expires="end_of_turn")
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #52 Portuguese Empire Crumbles
# --------------------------------------------------------------------------- #


@register("Portuguese Empire Crumbles")
def portuguese_empire_crumbles(game, ctx):
    """2 USSR influence in both SE African States and Angola."""
    game.state.add_inf(Side.USSR, "SE African States", 2)
    game.state.add_inf(Side.USSR, "Angola", 2)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #53 South African Unrest
# --------------------------------------------------------------------------- #


@register("South African Unrest")
def south_african_unrest(game, ctx):
    """2 USSR influence in South Africa, or 1 there and 2 among its neighbours."""
    state = game.state
    choice = yield from game.choose_option(
        Side.USSR,
        "South African Unrest:",
        [
            "add 2 USSR influence in South Africa",
            "add 1 USSR influence in South Africa and 2 in adjacent countries",
        ],
    )

    if choice == 0:
        state.add_inf(Side.USSR, "South Africa", 2)
        return

    state.add_inf(Side.USSR, "South Africa", 1)
    yield from game.add_influence(
        Side.USSR,
        2,
        allowed=adjacent_to("South Africa"),
        prompt="South African Unrest: 2 USSR influence in countries adjacent to South Africa",
    )


# --------------------------------------------------------------------------- #
# #54 Allende
# --------------------------------------------------------------------------- #


@register("Allende")
def allende(game, ctx):
    """2 USSR influence in Chile."""
    game.state.add_inf(Side.USSR, "Chile", 2)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #55 Willy Brandt
# --------------------------------------------------------------------------- #


def _willy_brandt_playable(game, side) -> bool:
    # Tear Down This Wall makes this event unplayable.
    return not game.state.has_resolved("Tear Down This Wall")


@register("Willy Brandt", playable_if=_willy_brandt_playable)
def willy_brandt(game, ctx):
    """USSR scores 1 VP and 1 influence in West Germany; cancels NATO there."""
    state = game.state
    state.award_vp(Side.USSR, 1)
    state.add_inf(Side.USSR, "West Germany", 1)
    state.add_effect(ctx.card, owner=Side.USSR)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #56 Muslim Revolution
# --------------------------------------------------------------------------- #


def _muslim_revolution_playable(game, side) -> bool:
    # AWACS Sale to Saudis makes this event unplayable.
    return not game.state.has_resolved("AWACS Sale to Saudis")


@register("Muslim Revolution", playable_if=_muslim_revolution_playable)
def muslim_revolution(game, ctx):
    """Remove all US influence in two of the eight named Muslim countries."""
    state = game.state
    for remaining in (2, 1):
        pool = [n for n in _MUSLIM_REVOLUTION if state.inf(Side.USA, n) > 0]
        name = yield from game.choose_country(
            ctx.player,
            pool,
            f"Muslim Revolution: remove all US influence from a country "
            f"({remaining} left)",
        )
        if name is None:
            break
        removed = state.clear_inf(Side.USA, name)
        state.note(f"Muslim Revolution: {name} loses {removed} US influence", ctx.player)


# --------------------------------------------------------------------------- #
# #57 ABM Treaty
# --------------------------------------------------------------------------- #


@register("ABM Treaty")
def abm_treaty(game, ctx):
    """Improve DEFCON one level, then conduct operations as if playing a 4 ops card."""
    game.improve_defcon(1)
    yield from game.free_operations(
        ctx.player, 4, prompt="ABM Treaty: conduct operations as if playing a 4 Op card"
    )


# --------------------------------------------------------------------------- #
# #58 Cultural Revolution
# --------------------------------------------------------------------------- #


@register("Cultural Revolution")
def cultural_revolution(game, ctx):
    """The USSR claims the China Card face up, or scores 1 VP if it already holds it."""
    state = game.state
    if state.china_card_owner is Side.USSR:
        state.award_vp(Side.USSR, 1)
        state.note("Cultural Revolution: the USSR already holds the China Card, +1 VP", ctx.player)
    else:
        state.china_card_owner = Side.USSR
        state.china_card_face_up = True
        state.note("Cultural Revolution: the USSR claims the China Card face up", ctx.player)
    return (yield from nothing())

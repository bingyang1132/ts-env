"""Mid War events, cards 59-81.

Conventions are the ones set out in :mod:`twilight.events.early_war`.

Three cards here need a hook the engine does not yet offer, and are implemented as
far as they can be from inside this module -- the effect is placed correctly, but
nothing consults it yet. Each is marked with a ``HOOK NEEDED`` comment:

* Flower Power, which watches every subsequent US card play;
* U2 Incident, which watches for UN Intervention;
* Shuttle Diplomacy, which alters Asia / Middle East scoring.
"""

from __future__ import annotations

from .. import rules, spacerace
from ..data import CARDS, COUNTRIES, COUNTRY_ORDER
from ..decisions import Decision, DecisionType, use_action
from ..enums import OpsUse, Region, Side
from . import register
from .helpers import empty_countries, in_region, nothing

#: The five cards Flower Power scores against. Kept here rather than in the shared
#: data because Flower Power is the only card that cares.
WAR_CARDS = frozenset(
    {
        "Arab-Israeli War",
        "Korean War",
        "Brush War",
        "Indo-Pakistani War",
        "Iran-Iraq War",
    }
)

#: The seven countries OPEC pays for.
_OPEC_COUNTRIES = (
    "Egypt",
    "Iran",
    "Libya",
    "Saudi Arabia",
    "Iraq",
    "Gulf States",
    "Venezuela",
)


def _hand_label(name: str) -> str:
    """Ops value and event ownership, for a card offered out of a hand."""
    c = CARDS[name]
    return (f"{c.ops}op" if c.ops else "scoring") + (
        f", {c.side.label} event" if c.side is not None else ", neutral"
    )


def _operations_with_card(game, side: Side, ops: int, prompt: str):
    """Conduct operations with the card being played, rather than a free grant.

    ``Game.free_operations`` exists for events worded "as if they played an N Op
    card", and marks any coup free so it does not count toward the required military
    operations. Grain Sales to Soviets instead says "use this card to conduct
    Operations normally", so the coup has to be a paid one.
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


# --------------------------------------------------------------------------- #
# #59 Flower Power
# --------------------------------------------------------------------------- #


def _flower_power_playable(game, side) -> bool:
    # An Evil Empire cancels and pre-empts this event.
    return not game.state.has_resolved("An Evil Empire")


@register("Flower Power", playable_if=_flower_power_playable)
def flower_power(game, ctx):
    """The USSR scores 2 VP for every war card the US subsequently plays."""
    # HOOK NEEDED: the engine has to award 2 VP to the USSR whenever the US plays
    # one of WAR_CARDS for its event or for operations -- but not on the space race
    # -- while this effect is in play. The natural place is Game._perform_use.
    game.state.add_effect(ctx.card, owner=Side.USSR)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #60 U2 Incident
# --------------------------------------------------------------------------- #


@register("U2 Incident")
def u2_incident(game, ctx):
    """1 VP to the USSR, and 1 more if UN Intervention is evented later this turn."""
    # HOOK NEEDED: UN Intervention's handler has to award the USSR 1 further VP and
    # remove this effect when it fires as an event while this card is in play.
    game.state.award_vp(Side.USSR, 1)
    game.state.add_effect(ctx.card, owner=Side.USSR, expires="end_of_turn")
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #61 OPEC
# --------------------------------------------------------------------------- #


def _opec_playable(game, side) -> bool:
    # North Sea Oil makes this event unplayable for the rest of the game.
    return not game.state.has_resolved("North Sea Oil")


@register("OPEC", playable_if=_opec_playable)
def opec(game, ctx):
    """1 VP to the USSR for each of the seven oil countries it controls."""
    state = game.state
    controlled = [n for n in _OPEC_COUNTRIES if rules.controls(state, Side.USSR, n)]
    state.award_vp(Side.USSR, len(controlled))
    state.note(
        f"OPEC: USSR controls {len(controlled)} oil countries "
        + (f"({', '.join(controlled)})" if controlled else ""),
        ctx.player,
    )
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #62 Lone Gunman
# --------------------------------------------------------------------------- #


@register("Lone Gunman")
def lone_gunman(game, ctx):
    """Reveal the US hand, then the USSR conducts operations as if playing 1 op."""
    state = game.state
    state.note(
        "Lone Gunman: US hand revealed -- " + ", ".join(sorted(state.hand(Side.USA))),
        ctx.player,
    )
    yield from game.free_operations(Side.USSR, 1)


# --------------------------------------------------------------------------- #
# #63 Colonial Rear Guards
# --------------------------------------------------------------------------- #


@register("Colonial Rear Guards")
def colonial_rear_guards(game, ctx):
    """1 US influence in each of four African or Southeast Asian countries."""
    pool = list(in_region(Region.AFRICA)) + list(in_region(Region.SOUTHEAST_ASIA))
    yield from game.add_influence(
        Side.USA,
        4,
        allowed=pool,
        max_per_country=1,
        prompt=(
            "Colonial Rear Guards: 1 US influence in four different African or "
            "SE Asian countries"
        ),
    )


# --------------------------------------------------------------------------- #
# #64 Panama Canal Returned
# --------------------------------------------------------------------------- #


@register("Panama Canal Returned")
def panama_canal_returned(game, ctx):
    """1 US influence in Panama, Costa Rica and Venezuela."""
    for name in ("Panama", "Costa Rica", "Venezuela"):
        game.state.add_inf(Side.USA, name, 1)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #65 Camp David Accords
# --------------------------------------------------------------------------- #


@register("Camp David Accords")
def camp_david_accords(game, ctx):
    """1 VP and 1 influence each in Israel, Jordan and Egypt; kills Arab-Israeli War."""
    state = game.state
    state.award_vp(Side.USA, 1)
    for name in ("Israel", "Jordan", "Egypt"):
        state.add_inf(Side.USA, name, 1)
    state.add_effect(ctx.card, owner=Side.USA)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #66 Puppet Governments
# --------------------------------------------------------------------------- #


@register("Puppet Governments")
def puppet_governments(game, ctx):
    """1 US influence in each of three countries with no influence from either side."""
    yield from game.add_influence(
        Side.USA,
        3,
        allowed=empty_countries(game.state),
        max_per_country=1,
        prompt="Puppet Governments: 1 US influence in three currently empty countries",
    )


# --------------------------------------------------------------------------- #
# #67 Grain Sales to Soviets
# --------------------------------------------------------------------------- #


@register("Grain Sales to Soviets")
def grain_sales_to_soviets(game, ctx):
    """Take a card at random from the USSR hand; play it, or return it and use this one."""
    state = game.state
    hand = state.hand(Side.USSR)

    if hand:
        stolen = state.rng.choice(hand)
        hand.remove(stolen)
        # In neither hand nor pile while the US decides what to do with it.
        state.hold(stolen)
        state.note(f"Grain Sales to Soviets takes {stolen} at random", ctx.player)
        keep = yield from game.confirm(
            Side.USA,
            f"Grain Sales to Soviets: play {stolen}? Declining returns it to the "
            "USSR hand and you use Grain Sales for its operations instead.",
            card=stolen,
            event_text=CARDS[stolen].event_text,
        )
        if keep:
            # Played as if it were a US card: rule 7.4.1 keeps the operations
            # modifiers with the player, not the card, and play_card recomputes them
            # for the US.
            state.release(stolen)
            yield from game.play_card(stolen, Side.USA)
            return
        hand.append(stolen)
        state.release(stolen)
        state.note(f"Grain Sales to Soviets: {stolen} returned", ctx.player)
    else:
        state.note("Grain Sales to Soviets: USSR hand is empty", ctx.player)

    ops = game.effective_ops(ctx.card, Side.USA)
    yield from _operations_with_card(
        game, Side.USA, ops, f"Grain Sales to Soviets: conduct {ops} operations"
    )


# --------------------------------------------------------------------------- #
# #68 John Paul II Elected Pope
# --------------------------------------------------------------------------- #


@register("John Paul II Elected Pope")
def john_paul_ii_elected_pope(game, ctx):
    """Remove 2 USSR influence in Poland, add 1 US; allows Solidarity."""
    state = game.state
    state.remove_inf(Side.USSR, "Poland", 2)
    state.add_inf(Side.USA, "Poland", 1)
    state.add_effect(ctx.card, owner=Side.USA)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #69 Latin American Death Squads
# --------------------------------------------------------------------------- #


@register("Latin American Death Squads")
def latin_american_death_squads(game, ctx):
    """+1 to the player's coups and -1 to the opponent's in the Americas this turn."""
    # rules.coup_modifier reads the effect's owner and applies the sign itself.
    game.state.add_effect(ctx.card, owner=ctx.player, expires="end_of_turn")
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #70 OAS Founded
# --------------------------------------------------------------------------- #


@register("OAS Founded")
def oas_founded(game, ctx):
    """2 US influence anywhere in Central or South America."""
    pool = list(in_region(Region.CENTRAL_AMERICA)) + list(in_region(Region.SOUTH_AMERICA))
    yield from game.add_influence(
        Side.USA,
        2,
        allowed=pool,
        prompt="OAS Founded: place 2 US influence in Central or South America",
    )


# --------------------------------------------------------------------------- #
# #71 Nixon Plays The China Card
# --------------------------------------------------------------------------- #


@register("Nixon Plays The China Card")
def nixon_plays_the_china_card(game, ctx):
    """2 VP if the US already holds the China Card, otherwise it takes it face down."""
    state = game.state
    if state.china_card_owner is Side.USA:
        state.award_vp(Side.USA, 2)
        state.note("Nixon Plays The China Card: US already holds it, +2 VP", ctx.player)
    else:
        state.china_card_owner = Side.USA
        state.china_card_face_up = False
        state.note(
            "Nixon Plays The China Card: the US takes the China Card face down",
            ctx.player,
        )
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #72 Sadat Expels Soviets
# --------------------------------------------------------------------------- #


@register("Sadat Expels Soviets")
def sadat_expels_soviets(game, ctx):
    """Remove all USSR influence in Egypt and add 1 US influence there."""
    state = game.state
    removed = state.clear_inf(Side.USSR, "Egypt")
    state.add_inf(Side.USA, "Egypt", 1)
    state.note(f"Sadat Expels Soviets: Egypt loses {removed} USSR influence", ctx.player)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #73 Shuttle Diplomacy
# --------------------------------------------------------------------------- #


@register("Shuttle Diplomacy")
def shuttle_diplomacy(game, ctx):
    """The next Asia or Middle East scoring ignores one USSR battleground."""
    # HOOK NEEDED: rules.region_status has to subtract one from the USSR battleground
    # count when scoring Asia or the Middle East while this effect is in play, and
    # Game.score_region then has to remove the effect and discard the card. Neither
    # may apply during the turn-10 final scoring.
    game.state.add_effect(ctx.card, owner=Side.USA)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #74 The Voice Of America
# --------------------------------------------------------------------------- #


@register("The Voice Of America")
def the_voice_of_america(game, ctx):
    """Remove 4 USSR influence from non-European countries, at most 2 per country."""
    pool = [n for n in COUNTRY_ORDER if Region.EUROPE not in COUNTRIES[n].regions]
    yield from game.remove_influence(
        Side.USSR,
        4,
        allowed=pool,
        max_per_country=2,
        chooser=ctx.player,
        prompt=(
            "The Voice Of America: remove USSR influence outside Europe (max 2 per "
            "country)"
        ),
    )


# --------------------------------------------------------------------------- #
# #75 Liberation Theology
# --------------------------------------------------------------------------- #


@register("Liberation Theology")
def liberation_theology(game, ctx):
    """3 USSR influence in Central America, at most 2 per country."""
    yield from game.add_influence(
        Side.USSR,
        3,
        allowed=in_region(Region.CENTRAL_AMERICA),
        max_per_country=2,
        prompt="Liberation Theology: 3 USSR influence in Central America (max 2 each)",
    )


# --------------------------------------------------------------------------- #
# #76 Ussuri River Skirmish
# --------------------------------------------------------------------------- #


@register("Ussuri River Skirmish")
def ussuri_river_skirmish(game, ctx):
    """Take the China Card face up, or add 4 US influence in Asia if already held."""
    state = game.state
    if state.china_card_owner is Side.USA:
        yield from game.add_influence(
            Side.USA,
            4,
            allowed=in_region(Region.ASIA),
            max_per_country=2,
            prompt="Ussuri River Skirmish: 4 US influence in Asia (max 2 per country)",
        )
        return

    state.china_card_owner = Side.USA
    state.china_card_face_up = True
    state.note(
        "Ussuri River Skirmish: the US claims the China Card face up", ctx.player
    )


# --------------------------------------------------------------------------- #
# #77 Ask Not What Your Country Can Do For You...
# --------------------------------------------------------------------------- #


def _ask_not_playable(game, side) -> bool:
    # The card has already left the hand by the time this is asked, so an empty hand
    # here means there is nothing to exchange.
    return bool(game.state.hand(Side.USA))


@register(
    "Ask Not What Your Country Can Do For You...", playable_if=_ask_not_playable
)
def ask_not_what_your_country_can_do_for_you(game, ctx):
    """The US discards any number of cards from hand and draws that many back."""
    state = game.state
    hand = state.hand(Side.USA)
    chosen: list[str] = []

    # All the discards are chosen before anything is drawn, as the card requires.
    while True:
        remaining = [n for n in hand if n not in chosen]
        if not remaining:
            break
        picked = yield from game.choose_card(
            Side.USA,
            remaining,
            f"Ask Not: choose a card to discard ({len(chosen)} chosen so far), "
            "or pass to stop",
            dtype=DecisionType.DISCARD_CARD,
            allow_pass=True,
            labels={n: _hand_label(n) for n in remaining},
        )
        if picked is None:
            break
        chosen.append(picked)

    if not chosen:
        state.note("Ask Not: the US discards nothing", ctx.player)
        return

    for name in chosen:
        hand.remove(name)
        state.discard_card(name)

    drawn: list[str] = []
    for _ in chosen:
        card = state.draw_card()
        if card is None:
            break
        hand.append(card)
        drawn.append(card)

    state.note(
        f"Ask Not: US discards {len(chosen)} card(s) and draws {len(drawn)}",
        ctx.player,
    )


# --------------------------------------------------------------------------- #
# #78 Alliance for Progress
# --------------------------------------------------------------------------- #


@register("Alliance for Progress")
def alliance_for_progress(game, ctx):
    """1 VP per US-controlled battleground in Central and South America."""
    state = game.state
    pool = list(in_region(Region.CENTRAL_AMERICA)) + list(in_region(Region.SOUTH_AMERICA))
    count = sum(
        1
        for n in pool
        if COUNTRIES[n].battleground and rules.controls(state, Side.USA, n)
    )
    state.award_vp(Side.USA, count)
    state.note(f"Alliance for Progress: {count} US battlegrounds, +{count} VP", ctx.player)
    return (yield from nothing())


# --------------------------------------------------------------------------- #
# #80 One Small Step...
# --------------------------------------------------------------------------- #


def _one_small_step_playable(game, side) -> bool:
    return game.state.space_race[side] < game.state.space_race[side.opponent]


@register("One Small Step...", playable_if=_one_small_step_playable)
def one_small_step(game, ctx):
    """Advance two space race boxes, scoring only the second box's victory points."""
    state = game.state
    first = spacerace.next_box(state.space_race[ctx.player])
    if first is None:
        state.note("One Small Step: already at the end of the track", ctx.player)
        return (yield from nothing())

    # The first box is passed over without paying out; only the landing box scores.
    second = spacerace.next_box(first.number)
    landing = first if second is None else second

    gained = spacerace.victory_points(state.space_race, ctx.player, landing.number)
    state.space_race[ctx.player] = landing.number
    state.award_vp(ctx.player, gained)
    state.note(
        f"One Small Step: advanced to {landing.name}, +{gained} VP", ctx.player
    )
    return (yield from nothing())

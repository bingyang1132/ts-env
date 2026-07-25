"""Turning game state into what one player is allowed to know.

Information hiding happens here and only here. :func:`observe` is the single funnel
that every agent -- numeric policy, language model, scripted bot -- looks through, so
the numeric encoding in :mod:`twilight.encode` and the text rendering in
:mod:`twilight.render` cannot disagree about the facts or leak different amounts.

What is hidden: the opponent's hand, and the order of the draw pile. What is *not*
hidden, because it is public in the real game: every influence marker, both military
operations totals, both space race positions, the discard and removed piles, and every
card in play. Deck composition is therefore inferable, which is deliberate -- card
counting is a real skill in Twilight Struggle, so :attr:`Observation.unseen` and the
helpers beside it hand the agent that inference rather than making it re-derive it.

The one exception is ``reveal_opponent_hand``, which exposes the opponent's actual hand.
It exists for analysis and for training a value network on complete information, and it
is off by default: a policy trained with it on has learned a game nobody can play.

The observation also precomputes the derived quantities agents habitually get wrong:
region control tiers, what each region would score right now, and coup thresholds.
Those are cheap for the engine and expensive (or error-prone) for a policy to infer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import rules, spacerace
from .data import CARDS, CARD_ORDER, CHINA_CARD, COUNTRIES, COUNTRY_ORDER
from .decisions import Decision
from .enums import AUTO_VICTORY, SCORING_REGIONS, Phase, Region, Side
from .state import GameState


@dataclass(frozen=True, slots=True)
class CountryView:
    """Everything public about one country, plus the observer's perspective on it."""

    name: str
    region: Region
    stability: int
    battleground: bool
    ussr: int
    usa: int
    controller: Side | None
    #: Influence the observer would need to add to take control.
    to_control: int
    #: Whether the observer may place operations influence here this action round.
    can_place: bool
    #: Cost per point of influence for the observer: 1, or 2 into enemy-controlled.
    place_cost: int
    can_coup: bool
    can_realign: bool
    adjacent_to_enemy_superpower: bool

    def mine(self, player: Side) -> int:
        return self.ussr if player is Side.USSR else self.usa

    def theirs(self, player: Side) -> int:
        return self.usa if player is Side.USSR else self.ussr


@dataclass(frozen=True, slots=True)
class RegionView:
    """A region's standing, from both sides."""

    region: Region
    #: ``"control"`` / ``"domination"`` / ``"presence"`` / ``"none"`` per side.
    tiers: tuple[str, str]
    countries: tuple[int, int]
    battlegrounds: tuple[int, int]
    total_battlegrounds: int
    #: VP each side would score if this region were scored right now.
    vp: tuple[int, int]
    #: Net VP swing to the observer if scored now; negative favours the opponent.
    net_vp_for_observer: int


@dataclass(frozen=True, slots=True)
class Observation:
    """One player's complete, legal view of the game."""

    player: Side
    turn: int
    action_round: int
    phase: Phase
    to_move: Side | None

    # -- tracks (all public) ------------------------------------------------ #
    #: Victory points from the observer's perspective: positive means they lead.
    vp: int
    defcon: int
    space_race: tuple[int, int]
    space_attempts_left: int
    military_ops: tuple[int, int]
    military_ops_required: int

    # -- board -------------------------------------------------------------- #
    countries: tuple[CountryView, ...]
    regions: tuple[RegionView, ...]

    # -- cards -------------------------------------------------------------- #
    hand: tuple[str, ...]
    opponent_hand_size: int
    #: The opponent's actual hand. Normally ``None``; set when a card in play reveals it
    #: (CIA Created, The Cambridge Five, Aldrich Ames Remix) or when the observation was
    #: taken with ``reveal_opponent_hand=True`` for analysis.
    opponent_hand_revealed: tuple[str, ...] | None
    #: The card currently being resolved, if any. Public: cards are played face up.
    playing_card: str | None
    china_card_owner: Side
    china_card_face_up: bool
    china_card_available: bool
    deck_size: int
    #: Cards whose location is unknown: the opponent's hand plus the draw pile. This is
    #: the raw material for card counting; see :meth:`Observation.unseen_scoring_cards`
    #: and :attr:`in_deck_odds`.
    unseen: tuple[str, ...]
    #: Chance a given unseen card is in the draw pile rather than the opponent's hand,
    #: assuming nothing else is known about it.
    in_deck_odds: float
    discard: tuple[str, ...]
    removed: tuple[str, ...]
    effects: tuple[tuple[str, str | None], ...]

    # -- decision ----------------------------------------------------------- #
    decision: Decision | None
    #: Set once the game is over.
    winner: Side | None = None
    win_reason: str | None = None
    log_tail: tuple[str, ...] = field(default=())

    # -- convenience -------------------------------------------------------- #

    @property
    def is_over(self) -> bool:
        return self.winner is not None or self.decision is None

    @property
    def opponent(self) -> Side:
        return self.player.opponent

    def country(self, name: str) -> CountryView:
        for view in self.countries:
            if view.name == name:
                return view
        raise KeyError(name)

    def region(self, region: Region) -> RegionView:
        for view in self.regions:
            if view.region is region:
                return view
        raise KeyError(region)

    @property
    def legal_actions(self) -> tuple[str, ...]:
        return () if self.decision is None else self.decision.legal_keys

    # -- card counting ---------------------------------------------------- #

    def unseen_scoring_cards(self) -> tuple[str, ...]:
        """Scoring cards still unaccounted for, in printed order.

        The single most valuable card-counting fact in the game: a scoring card must be
        played the turn it is drawn, so knowing whether Europe Scoring is still out there
        changes how much a region is worth defending.
        """
        return tuple(n for n in self.unseen if CARDS[n].is_scoring)

    def unseen_by_side(self) -> dict[str, int]:
        """How many unseen cards carry each side's event."""
        counts = {"USSR": 0, "USA": 0, "Neutral": 0}
        for name in self.unseen:
            side = CARDS[name].side
            counts[side.label if side is not None else "Neutral"] += 1
        return counts

    def cards_seen(self) -> tuple[str, ...]:
        """Every card whose location the observer knows, in printed order."""
        known = set(self.hand) | set(self.discard) | set(self.removed)
        known |= {name for name, _ in self.effects}
        return tuple(n for n in CARD_ORDER if n in known)


#: Cards that, while in play, let their owner see the opponent's hand.
_HAND_REVEALING = ("CIA Created", "The Cambridge Five", "Aldrich Ames Remix")


def _opponent_hand_revealed(state: GameState, player: Side) -> tuple[str, ...] | None:
    for name in _HAND_REVEALING:
        if state.in_play(name, owner=player):
            return tuple(sorted(state.hand(player.opponent)))
    return None


def _unseen_cards(state: GameState, player: Side) -> tuple[str, ...]:
    """Cards in this game whose location the observer does not know.

    That is the opponent's hand plus the draw pile: everything except the observer's own
    hand, the discard and removed piles, and the cards face up in play -- all of which
    are public or theirs.

    Two exclusions matter and are easy to get wrong. Cards from a stage that has not been
    shuffled in yet **cannot** be drawn (no Mid War card exists before turn 4), and the
    seven optional cards are absent entirely unless the game was set up with them. Listing
    either as unseen would tell an agent to count cards that are not in the game.
    """
    accounted = set(state.hand(player)) | set(state.discard) | set(state.removed)
    accounted |= set(state.effects) | set(state.transit)
    accounted.add(CHINA_CARD)
    if state.playing_card is not None:
        accounted.add(state.playing_card)

    return tuple(
        name
        for name in CARD_ORDER
        if name not in accounted
        and CARDS[name].stage in state.stages_in_deck
        and (state.optional_cards or not CARDS[name].optional)
    )


def observe(
    state: GameState,
    player: Side,
    decision: Decision | None = None,
    *,
    log_tail: int = 12,
    reveal_opponent_hand: bool = False,
) -> Observation:
    """Build *player*'s view of *state*.

    *decision* is the pending question, if any; it is included verbatim only when it
    belongs to *player*, so an agent never sees the menu it is not being asked.

    *reveal_opponent_hand* breaks the game's hidden information on purpose. Use it for
    analysis, replays and complete-information value networks -- never for a policy that
    has to play for real.
    """
    reachable = rules.reachable_countries(state, player)

    country_views = []
    for name in COUNTRY_ORDER:
        c = COUNTRIES[name]
        ussr = state.inf(Side.USSR, name)
        usa = state.inf(Side.USA, name)
        mine = ussr if player is Side.USSR else usa
        theirs = usa if player is Side.USSR else ussr
        country_views.append(
            CountryView(
                name=name,
                region=c.region,
                stability=c.stability,
                battleground=c.battleground,
                ussr=ussr,
                usa=usa,
                controller=rules.controller(state, name),
                to_control=max(0, c.stability + theirs - mine),
                can_place=name in reachable,
                place_cost=rules.influence_cost(state, player, name),
                can_coup=rules.can_coup(state, player, name),
                can_realign=rules.can_realign(state, player, name),
                adjacent_to_enemy_superpower=name
                in COUNTRIES[("USA" if player is Side.USSR else "USSR")].adjacent,
            )
        )

    region_views = []
    for region in SCORING_REGIONS:
        status = rules.region_status(state, region)
        vp = tuple(rules.region_vp(state, region, s) for s in Side)
        # Europe control is an automatic win rather than a VP amount; report the swing
        # as the VP cap so it still reads as overwhelmingly good.
        capped = tuple(min(v, AUTO_VICTORY) for v in vp)
        net = capped[player] - capped[player.opponent]
        region_views.append(
            RegionView(
                region=region,
                tiers=(status.tier(Side.USSR), status.tier(Side.USA)),
                countries=(
                    status.sides[Side.USSR].countries,
                    status.sides[Side.USA].countries,
                ),
                battlegrounds=(
                    status.sides[Side.USSR].battlegrounds,
                    status.sides[Side.USA].battlegrounds,
                ),
                total_battlegrounds=status.total_battlegrounds,
                vp=(vp[Side.USSR], vp[Side.USA]),
                net_vp_for_observer=net,
            )
        )

    china_available = state.china_card_owner is player and state.china_card_face_up

    unseen = _unseen_cards(state, player)
    hidden_elsewhere = len(state.deck) + len(state.hand(player.opponent))
    in_deck_odds = len(state.deck) / hidden_elsewhere if hidden_elsewhere else 0.0

    return Observation(
        player=player,
        turn=state.turn,
        action_round=state.action_round,
        phase=state.phase,
        to_move=None if decision is None else decision.player,
        vp=state.vp_for(player),
        defcon=state.defcon,
        space_race=(state.space_race[Side.USSR], state.space_race[Side.USA]),
        space_attempts_left=max(
            0,
            spacerace.attempts_allowed(state.space_race, player)
            - state.space_attempts[player],
        ),
        military_ops=(state.military_ops[Side.USSR], state.military_ops[Side.USA]),
        military_ops_required=state.defcon,
        countries=tuple(country_views),
        regions=tuple(region_views),
        hand=tuple(sorted(state.hand(player), key=lambda n: CARDS[n].number)),
        opponent_hand_size=len(state.hand(player.opponent)),
        opponent_hand_revealed=(
            tuple(sorted(state.hand(player.opponent), key=lambda n: CARDS[n].number))
            if reveal_opponent_hand
            else _opponent_hand_revealed(state, player)
        ),
        playing_card=state.playing_card,
        china_card_owner=state.china_card_owner,
        china_card_face_up=state.china_card_face_up,
        china_card_available=china_available,
        deck_size=len(state.deck),
        unseen=unseen,
        in_deck_odds=in_deck_odds,
        discard=tuple(sorted(state.discard, key=lambda n: CARDS[n].number)),
        removed=tuple(sorted(state.removed, key=lambda n: CARDS[n].number)),
        effects=tuple(
            (name, effect.owner.label if effect.owner is not None else None)
            for name, effect in sorted(state.effects.items())
        ),
        decision=decision if (decision is not None and decision.player is player) else None,
        winner=state.winner,
        win_reason=state.win_reason.value if state.win_reason is not None else None,
        log_tail=tuple(str(e) for e in state.log[-log_tail:]) if log_tail else (),
    )


__all__ = ["CountryView", "Observation", "RegionView", "observe"]

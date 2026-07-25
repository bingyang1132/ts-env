"""The action vocabulary and decision points.

The engine never asks for a whole turn at once. It asks a stream of *atomic* questions
-- "which card?", "which country?", "how many?" -- each with a small legal set. One
atomic action corresponds to roughly one click in the real game.

This factoring is what makes the same engine usable by every kind of agent:

* a reinforcement-learning policy gets a fixed ``ACTION_VOCAB`` and a boolean mask,
  never a combinatorial blowup (placing 4 influence across 84 countries is 4 separate
  choices, not one of ~2 million);
* a language model gets a short numbered menu it can actually read, and a stable
  canonical string per action for constrained decoding and token-level credit.

The vocabulary is closed and ordered, so index *i* means the same thing in every game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from .data import CARD_ORDER, COUNTRY_ORDER
from .enums import OpsUse, Region, Side


class ActionKind(str, Enum):
    """The families of atomic action."""

    CARD = "card"        # pick a card out of a hand or pile
    USE = "use"          # pick how to use the card just chosen
    COUNTRY = "country"  # pick a country as a target
    REGION = "region"    # pick a region
    NUMBER = "number"    # pick a quantity or a die/track value
    OPTION = "option"    # pick from an event-specific list of choices
    YES = "yes"
    NO = "no"
    PASS = "pass"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Action:
    """One atomic action.

    ``key`` is the canonical, stable string form -- the thing a language model emits
    and the thing that indexes into :data:`ACTION_VOCAB`. ``label`` is free to be more
    descriptive for prompts and logs.
    """

    kind: ActionKind
    value: Any = None
    label: str = ""

    @property
    def key(self) -> str:
        if self.value is None:
            return str(self.kind)
        return f"{self.kind}:{self.value}"

    @property
    def index(self) -> int:
        return ACTION_INDEX[self.key]

    def display(self) -> str:
        return self.label or self.key

    def __str__(self) -> str:
        return self.key


# --------------------------------------------------------------------------- #
# The closed vocabulary
# --------------------------------------------------------------------------- #

#: Largest quantity any decision ever asks for (influence into one country, ops to
#: spend, a die face, a space-race box).
MAX_NUMBER = 12
#: Largest number of event-specific choices any single card offers.
MAX_OPTIONS = 12


def _build_vocab() -> tuple[tuple[str, ...], dict[str, int]]:
    keys: list[str] = []
    keys += [f"{ActionKind.CARD}:{name}" for name in CARD_ORDER]
    keys += [f"{ActionKind.USE}:{use.value}" for use in OpsUse]
    keys += [f"{ActionKind.COUNTRY}:{name}" for name in COUNTRY_ORDER]
    keys += [f"{ActionKind.REGION}:{region.value}" for region in Region]
    keys += [f"{ActionKind.NUMBER}:{n}" for n in range(MAX_NUMBER + 1)]
    keys += [f"{ActionKind.OPTION}:{n}" for n in range(MAX_OPTIONS)]
    keys += [str(ActionKind.YES), str(ActionKind.NO), str(ActionKind.PASS)]
    return tuple(keys), {key: i for i, key in enumerate(keys)}


ACTION_VOCAB, ACTION_INDEX = _build_vocab()
NUM_ACTIONS = len(ACTION_VOCAB)


# -- constructors, so callers never hand-build key strings ------------------- #


def card_action(name: str, label: str = "") -> Action:
    return Action(ActionKind.CARD, name, label)


def use_action(use: OpsUse, label: str = "") -> Action:
    return Action(ActionKind.USE, use.value, label)


def country_action(name: str, label: str = "") -> Action:
    return Action(ActionKind.COUNTRY, name, label)


def region_action(region: Region, label: str = "") -> Action:
    return Action(ActionKind.REGION, region.value, label)


def number_action(value: int, label: str = "") -> Action:
    if not 0 <= value <= MAX_NUMBER:
        raise ValueError(f"number action out of range: {value}")
    return Action(ActionKind.NUMBER, value, label)


def option_action(index: int, label: str = "") -> Action:
    if not 0 <= index < MAX_OPTIONS:
        raise ValueError(f"option index out of range: {index}")
    return Action(ActionKind.OPTION, index, label)


YES = Action(ActionKind.YES, None, "yes")
NO = Action(ActionKind.NO, None, "no")
PASS = Action(ActionKind.PASS, None, "pass")


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #


class DecisionType(str, Enum):
    """What the engine is asking. Useful as a feature and for prompt templating."""

    SETUP_INFLUENCE = "setup_influence"
    HEADLINE = "headline"
    PLAY_CARD = "play_card"
    CARD_USE = "card_use"
    PLACE_INFLUENCE = "place_influence"
    REMOVE_INFLUENCE = "remove_influence"
    COUP_TARGET = "coup_target"
    REALIGN_TARGET = "realign_target"
    REALIGN_CONTINUE = "realign_continue"
    DISCARD_CARD = "discard_card"
    CHOOSE_CARD = "choose_card"
    CHOOSE_COUNTRY = "choose_country"
    CHOOSE_REGION = "choose_region"
    CHOOSE_NUMBER = "choose_number"
    CHOOSE_OPTION = "choose_option"
    CONFIRM = "confirm"
    SPACE_RACE = "space_race"

    def __str__(self) -> str:
        return self.value


@dataclass(slots=True)
class Decision:
    """A single question put to one player."""

    type: DecisionType
    player: Side
    prompt: str
    options: tuple[Action, ...]
    #: Extra machine-readable context for encoders and prompt templates, e.g. the
    #: card being resolved or how many operations points remain.
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(f"decision {self.type} for {self.player.label} has no legal options")
        seen: set[str] = set()
        for action in self.options:
            if action.key in seen:
                raise ValueError(f"duplicate option {action.key} in {self.type} decision")
            seen.add(action.key)

    # -- lookup ---------------------------------------------------------- #

    def find(self, key: str) -> Action | None:
        for action in self.options:
            if action.key == key:
                return action
        return None

    def resolve(self, choice: "Action | str | int") -> Action:
        """Interpret an agent's choice, accepting an Action, a canonical key, or an index.

        An integer is read as an index into the global vocabulary, not into
        ``options`` -- reinforcement-learning policies emit vocabulary indices, and
        making that the only integer meaning avoids a silent off-by-one class of bug.
        """
        if isinstance(choice, Action):
            match = self.find(choice.key)
            if match is None:
                raise IllegalAction(self, choice.key)
            return match
        if isinstance(choice, str):
            match = self.find(choice)
            if match is None:
                raise IllegalAction(self, choice)
            return match
        if isinstance(choice, int):
            if not 0 <= choice < NUM_ACTIONS:
                raise IllegalAction(self, f"index {choice}")
            match = self.find(ACTION_VOCAB[choice])
            if match is None:
                raise IllegalAction(self, ACTION_VOCAB[choice])
            return match
        raise TypeError(f"cannot interpret choice of type {type(choice).__name__}")

    # -- encodings ------------------------------------------------------- #

    @property
    def legal_keys(self) -> tuple[str, ...]:
        return tuple(a.key for a in self.options)

    @property
    def legal_indices(self) -> tuple[int, ...]:
        return tuple(a.index for a in self.options)

    def mask(self) -> list[bool]:
        """Boolean mask over the whole vocabulary."""
        out = [False] * NUM_ACTIONS
        for action in self.options:
            out[action.index] = True
        return out

    def menu(self) -> str:
        """Numbered, human-readable option list for language-model prompts."""
        return "\n".join(
            f"  [{i}] {a.key:<28} {a.label}".rstrip()
            for i, a in enumerate(self.options)
        )


class IllegalAction(ValueError):
    """Raised when an agent picks something the current decision does not allow."""

    def __init__(self, decision: Decision, attempted: str) -> None:
        self.decision = decision
        self.attempted = attempted
        legal = ", ".join(decision.legal_keys[:12])
        if len(decision.options) > 12:
            legal += f", ... ({len(decision.options)} total)"
        super().__init__(
            f"{attempted!r} is not legal for {decision.type} "
            f"({decision.player.label}). Legal: {legal}"
        )


def countries_decision(
    dtype: DecisionType,
    player: Side,
    prompt: str,
    names: Sequence[str],
    *,
    labels: dict[str, str] | None = None,
    allow_pass: bool = False,
    **context: Any,
) -> Decision:
    """Build a decision over a list of countries, in canonical board order."""
    labels = labels or {}
    ordered = [n for n in COUNTRY_ORDER if n in set(names)]
    options = [country_action(n, labels.get(n, "")) for n in ordered]
    if allow_pass:
        options.append(PASS)
    return Decision(dtype, player, prompt, tuple(options), context)


def cards_decision(
    dtype: DecisionType,
    player: Side,
    prompt: str,
    names: Sequence[str],
    *,
    labels: dict[str, str] | None = None,
    allow_pass: bool = False,
    **context: Any,
) -> Decision:
    """Build a decision over a list of cards, in printed-number order."""
    labels = labels or {}
    wanted = set(names)
    ordered = [n for n in CARD_ORDER if n in wanted]
    options = [card_action(n, labels.get(n, "")) for n in ordered]
    if allow_pass:
        options.append(PASS)
    return Decision(dtype, player, prompt, tuple(options), context)


def options_decision(
    player: Side,
    prompt: str,
    labels: Sequence[str],
    *,
    dtype: DecisionType = DecisionType.CHOOSE_OPTION,
    **context: Any,
) -> Decision:
    """Build a decision over an event-specific list of labelled choices."""
    if len(labels) > MAX_OPTIONS:
        raise ValueError(f"{len(labels)} options exceeds MAX_OPTIONS={MAX_OPTIONS}")
    options = tuple(option_action(i, label) for i, label in enumerate(labels))
    return Decision(dtype, player, prompt, options, context)


def confirm_decision(player: Side, prompt: str, **context: Any) -> Decision:
    return Decision(DecisionType.CONFIRM, player, prompt, (YES, NO), context)


__all__ = [
    "ACTION_INDEX",
    "ACTION_VOCAB",
    "Action",
    "ActionKind",
    "Decision",
    "DecisionType",
    "IllegalAction",
    "MAX_NUMBER",
    "MAX_OPTIONS",
    "NO",
    "NUM_ACTIONS",
    "PASS",
    "YES",
    "card_action",
    "cards_decision",
    "confirm_decision",
    "countries_decision",
    "country_action",
    "number_action",
    "option_action",
    "options_decision",
    "region_action",
    "use_action",
]

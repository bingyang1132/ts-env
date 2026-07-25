"""The Space Race track.

Values are the Deluxe / 2015 edition ones. The pre-Deluxe track differs -- it swaps
the box 6/7/8 names and abilities and pays 4/2 for Lunar Orbit -- so do not mix the
editions.

Each box needs a minimum operations value to attempt and succeeds on a die roll at or
below its threshold. Four boxes pay victory points; the other four instead grant a
lasting ability, which belongs to the *first* player to arrive and is cancelled the
moment the opponent reaches the same box (rule 6.4.4).
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import Side


@dataclass(frozen=True, slots=True)
class SpaceBox:
    number: int
    name: str
    #: Minimum operations value of the card discarded to attempt this box.
    ops_required: int
    #: The attempt succeeds on a die roll of this value or lower.
    max_roll: int
    vp_first: int
    vp_second: int
    #: Ability granted to the first player to reach this box, if any.
    ability: str | None = None


#: Ability identifiers.
TWO_ATTEMPTS = "two_attempts"
OPPONENT_HEADLINE_FIRST = "opponent_headline_first"
DISCARD_HELD_CARD = "discard_held_card"
EIGHT_ACTION_ROUNDS = "eight_action_rounds"

#: Box 0 is the starting position, so index N is box N.
SPACE_TRACK: tuple[SpaceBox, ...] = (
    SpaceBox(0, "Start", 0, 0, 0, 0),
    SpaceBox(1, "Earth Satellite", 2, 3, 2, 1),
    SpaceBox(2, "Animal in Space", 2, 4, 0, 0, ability=TWO_ATTEMPTS),
    SpaceBox(3, "Man in Space", 2, 3, 2, 0),
    SpaceBox(4, "Man in Earth Orbit", 2, 4, 0, 0, ability=OPPONENT_HEADLINE_FIRST),
    SpaceBox(5, "Lunar Orbit", 3, 3, 3, 1),
    SpaceBox(6, "Eagle/Bear has Landed", 3, 4, 0, 0, ability=DISCARD_HELD_CARD),
    SpaceBox(7, "Space Shuttle", 3, 3, 4, 2),
    SpaceBox(8, "Space Station", 4, 2, 2, 0, ability=EIGHT_ACTION_ROUNDS),
)

MAX_BOX = len(SPACE_TRACK) - 1

ABILITY_BOX: dict[str, int] = {
    box.ability: box.number for box in SPACE_TRACK if box.ability is not None
}


def next_box(position: int) -> SpaceBox | None:
    """The box a player at *position* would attempt next, or ``None`` if finished."""
    if position >= MAX_BOX:
        return None
    return SPACE_TRACK[position + 1]


def has_ability(space_race: list[int], side: Side, ability: str) -> bool:
    """Whether *side* currently holds the ability granted by a space race box.

    Held only while *side* has reached the box and the opponent has not.
    """
    box = ABILITY_BOX[ability]
    return space_race[side] >= box > space_race[side.opponent]


def attempts_allowed(space_race: list[int], side: Side) -> int:
    """How many space race attempts *side* may make this turn.

    One normally; two while holding the Animal in Space ability. An attempt is spent
    whether it succeeds or fails.
    """
    return 2 if has_ability(space_race, side, TWO_ATTEMPTS) else 1


def action_rounds_for(space_race: list[int], side: Side, base: int) -> int:
    """Action rounds *side* may take this turn, allowing for the Space Station."""
    return base + 1 if has_ability(space_race, side, EIGHT_ACTION_ROUNDS) else base


def victory_points(space_race: list[int], side: Side, box_number: int) -> int:
    """VP for *side* newly reaching *box_number*.

    The first player to a box scores the higher amount, the second the lower.
    """
    box = SPACE_TRACK[box_number]
    opponent_already_there = space_race[side.opponent] >= box_number
    return box.vp_second if opponent_already_there else box.vp_first


__all__ = [
    "ABILITY_BOX",
    "DISCARD_HELD_CARD",
    "EIGHT_ACTION_ROUNDS",
    "MAX_BOX",
    "OPPONENT_HEADLINE_FIRST",
    "SPACE_TRACK",
    "SpaceBox",
    "TWO_ATTEMPTS",
    "action_rounds_for",
    "attempts_allowed",
    "has_ability",
    "next_box",
    "victory_points",
]

"""The interactive CLI and the HTML visualiser.

The delta encoding is the part worth testing hard: if it drifts, the visualiser shows a
board that never existed, and nothing else would notice.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "examples"))

from twilight import Game  # noqa: E402
from twilight.record import GameRecord, record_from_game  # noqa: E402

import play  # noqa: E402
import viz  # noqa: E402


def expand(frames: list[dict]) -> list[dict]:
    """Reference implementation of what the page does on load.

    Deliberately written independently of the encoder so the two have to agree.
    """
    out = [dict(frames[0])]
    inf = list(frames[0]["inf"])
    ctrl = list(frames[0]["ctrl"])
    carried = {key: frames[0][key] for key in viz._CARRIED_FORWARD}

    for frame in frames[1:]:
        inf = list(inf)
        ctrl = list(ctrl)
        for index, ussr, usa in frame["dinf"]:
            inf[index] = [ussr, usa]
        for index, value in frame["dctrl"]:
            ctrl[index] = value
        restored = dict(frame)
        restored.pop("dinf")
        restored.pop("dctrl")
        restored["inf"] = inf
        restored["ctrl"] = ctrl
        for key in viz._CARRIED_FORWARD:
            if key in restored:
                carried[key] = restored[key]
            else:
                restored[key] = carried[key]
        out.append(restored)
    return out


# --------------------------------------------------------------------------- #
# Delta encoding
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 5, 11])
def test_delta_encoding_round_trips_exactly(seed):
    """Every frame the visualiser draws must match the state it was taken from."""
    record = viz.play_game(seed, ("safe_random", "safe_random"), optional_cards=False)

    # Capture the truth without any encoding.
    truth = []
    game = Game(seed)
    truth.append(viz.snapshot(game))
    for key in record.history:
        if game.decision is None:
            break
        game.step(key)
        truth.append(viz.snapshot(game))

    encoded = viz.record_frames(record)["frames"]
    assert len(encoded) == len(truth)

    for i, (got, want) in enumerate(zip(expand(encoded), truth, strict=True)):
        assert got["inf"] == want["inf"], f"influence wrong at frame {i}"
        assert got["ctrl"] == want["ctrl"], f"control wrong at frame {i}"
        assert got["regions"] == want["regions"], f"regions wrong at frame {i}"
        assert got["effects"] == want["effects"], f"effects wrong at frame {i}"
        assert got["vp"] == want["vp"] and got["defcon"] == want["defcon"]


def test_delta_encoding_actually_shrinks_the_payload():
    record = viz.play_game(3, ("safe_random", "safe_random"), optional_cards=False)
    encoded = viz.record_frames(record)["frames"]

    # Only the first frame carries full arrays.
    assert "inf" in encoded[0] and "ctrl" in encoded[0]
    assert all("inf" not in f and "dinf" in f for f in encoded[1:])

    # A single action rarely touches more than a couple of countries.
    changes = [len(f["dinf"]) for f in encoded[1:]]
    assert max(changes) <= 84
    assert sum(changes) / len(changes) < 8, "deltas are not actually small"


def test_every_playable_country_has_a_board_rectangle():
    layout = viz.country_layout()
    assert len(layout) == 84
    for box in layout:
        assert 0 <= box["x"] <= viz.BOARD_W
        assert 0 <= box["y"] <= viz.BOARD_H
        assert box["w"] > 0 and box["h"] > 0


def test_board_boxes_do_not_all_collapse_to_one_spot():
    """A coordinate mix-up would stack every country on top of each other."""
    layout = viz.country_layout()
    xs = {box["x"] for box in layout}
    ys = {box["y"] for box in layout}
    assert len(xs) > 40 and len(ys) > 25


# --------------------------------------------------------------------------- #
# Regions on the board
# --------------------------------------------------------------------------- #


def test_region_bands_report_the_engines_scoring_membership():
    """The chips must agree with the rules layer, not with the drawn box."""
    from twilight.data import REGION_COUNTRIES
    from twilight.enums import Region

    bands = {b["name"]: b for b in viz.region_bands()}
    for name, band in bands.items():
        expected = set(REGION_COUNTRIES[Region(name)])
        assert set(band["members"]) == expected, name

    # Austria and Finland are in both halves, so both halves must count them.
    assert len(bands["Eastern Europe"]["members"]) == 9
    assert len(bands["Western Europe"]["members"]) == 14
    # Southeast Asia scores inside Asia too.
    assert len(bands["Asia"]["members"]) == 15


def test_region_bands_are_plausible_boxes():
    bands = viz.region_bands()
    assert len(bands) == 8
    board = viz.BOARD_W * viz.BOARD_H
    for band in bands:
        assert band["w"] > 0 and band["h"] > 0
        # No band may swallow the board; the largest is Asia at about a fifth.
        assert (band["w"] * band["h"]) / board < 0.30, band["name"]


def test_both_europes_are_austria_and_finland():
    """Found by dual membership: their primary region resolves to Western Europe."""
    assert set(viz.BOTH_EUROPES) == {"Austria", "Finland"}


def test_card_reference_covers_every_card_with_its_text():
    from twilight.data import CARDS

    reference = viz.card_reference()
    assert set(reference) == set(CARDS)
    nato = reference["NATO"]
    assert nato["n"] == 21 and nato["side"] == "USA" and nato["star"] is True
    assert reference["Europe Scoring"]["scoring"] is True
    assert all(entry["text"] for entry in reference.values())


# --------------------------------------------------------------------------- #
# Hands in the replay
# --------------------------------------------------------------------------- #


def test_frames_carry_both_hands_and_the_unseen_counts():
    record = viz.play_game(5, ("safe_random", "safe_random"), optional_cards=False)
    frames = expand(viz.record_frames(record)["frames"])

    for i, frame in enumerate(frames):
        assert len(frame["handCards"]) == 2
        assert len(frame["handCards"][0]) == frame["hands"][0], f"frame {i}"
        assert len(frame["handCards"][1]) == frame["hands"][1], f"frame {i}"
        # Each side's unseen count excludes their own hand, so it cannot exceed the rest.
        assert 0 <= frame["unseen"][0] <= 110
        assert 0 <= frame["unseen"][1] <= 110

    # Hands should actually change over a game, or the carry-forward is broken.
    distinct = {tuple(f["handCards"][0]) for f in frames}
    assert len(distinct) > 5


# --------------------------------------------------------------------------- #
# HTML output
# --------------------------------------------------------------------------- #


def test_visualiser_writes_self_contained_html(tmp_path: Path):
    record = viz.play_game(7, ("safe_random", "safe_random"), optional_cards=False)
    data = viz.record_frames(record)
    data["subtitle"] = "test"
    page = viz.build_html(data, "Test title")

    assert page.startswith("<!DOCTYPE html>")
    assert "__DATA__" not in page and "__TITLE__" not in page
    assert "Test title" in page
    # Nothing may be fetched at view time. The SVG namespace URI is not a request, so
    # look for actual resource loads rather than for the string "http".
    for pattern in (
        "<script src=",
        "<link ",
        "@import",
        "fetch(",
        "XMLHttpRequest",
        'src="http',
        'href="http',
        "url(http",
    ):
        assert pattern not in page, f"page is not self-contained: {pattern}"

    out = tmp_path / "game.html"
    out.write_text(page, encoding="utf-8")
    assert out.stat().st_size > 10_000


def test_visualiser_carries_rationales_through_to_the_page():
    """The greedy baseline explains itself, so a replay of it must show reasons."""
    record = viz.play_game(4, ("greedy", "greedy"), optional_cards=False)
    data = viz.record_frames(record)

    assert len(data["notes"]) == len(data["frames"])
    assert data["notes"][0] is None, "the opening frame has no action behind it"
    annotated = [n for n in data["notes"] if n]
    assert annotated, "greedy should annotate its moves"
    assert any("scored" in n for n in annotated)

    page = viz.build_html({**data, "subtitle": "t"}, "t")
    assert "WHY —" in page and "prev reason" in page


def test_visualiser_handles_a_recording_with_no_rationales():
    record = viz.play_game(9, ("safe_random", "safe_random"), optional_cards=False)
    data = viz.record_frames(record)
    assert all(n is None for n in data["notes"])
    # The page must still build, and say so rather than showing an empty panel.
    page = viz.build_html({**data, "subtitle": "t"}, "t")
    assert "no rationales in this recording" in page


def test_visualiser_frame_count_matches_the_action_count():
    record = viz.play_game(9, ("safe_random", "safe_random"), optional_cards=False)
    data = viz.record_frames(record)
    # One frame before the first action, plus one after each.
    assert len(data["frames"]) == len(record) + 1
    assert len(data["actions"]) == len(data["frames"])
    assert len(data["log"]) == len(data["frames"])
    assert len(data["notes"]) == len(data["frames"])
    assert data["actions"][0] == "start of game"


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


def test_recording_round_trips_through_the_visualiser(tmp_path: Path):
    game = Game(seed=13)
    rng = random.Random(13)
    while game.decision is not None:
        game.step(rng.choice(game.decision.options))

    path = tmp_path / "rec.json"
    record_from_game(game, seed=13).save(path)
    reloaded = GameRecord.load(path)

    assert reloaded.seed == 13
    assert reloaded.history == game.history
    # Replaying the recording must reproduce the same ending.
    frames = viz.record_frames(reloaded)["frames"]
    assert frames[-1]["winner"] == reloaded.winner
    assert frames[-1]["reason"] == reloaded.win_reason


def test_rebuild_reproduces_a_prefix_of_the_game():
    """Undo relies on this: replaying a prefix must give that exact position."""
    game = Game(seed=21)
    rng = random.Random(21)
    for _ in range(60):
        if game.decision is None:
            break
        game.step(rng.choice(game.decision.options))

    prefix = game.history[:40]
    rebuilt = play.rebuild(21, prefix)
    reference = Game(21)
    for key in prefix:
        reference.step(key)

    assert rebuilt.state.influence == reference.state.influence
    assert rebuilt.state.vp == reference.state.vp
    assert rebuilt.history == reference.history

"""pytest hooks for tests/cards; the position-building helpers live in _table.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import twilight  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "unverified_rule: expectation rests on a ruling the sources could not confirm "
        "(e.g. 'as if they played an N Ops card', Deluxe rule 7.4.3); written to the "
        "coordinator's chosen reading so it can be found and flipped in one place",
    )


def pytest_report_header(config):
    return f"twilight engine under test: {twilight.__file__}"

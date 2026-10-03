"""The reply parser of the example language-model harness."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from llm_agent import extract_key, extract_reason  # noqa: E402

LEGAL = ("card:Nasser", "card:Duck and Cover", "country:West Germany", "pass")


def test_exact_key_and_action_line():
    assert extract_key("card:Nasser", LEGAL) == "card:Nasser"
    assert extract_key("REASON: ok\nACTION: card:Nasser", LEGAL) == "card:Nasser"
    assert extract_key("ACTION: `country:West Germany`.", LEGAL) == "country:West Germany"


def test_reason_line_cannot_hijack_the_action():
    reply = "REASON: card:Duck and Cover is tempting but too risky\nACTION: card:Nasser"
    assert extract_key(reply, LEGAL) == "card:Nasser"
    assert extract_reason(reply).startswith("card:Duck and Cover")


def test_menu_number_and_fallbacks():
    assert extract_key("ACTION: [1]", LEGAL) == "card:Duck and Cover"
    assert extract_key("ACTION: 3", LEGAL) == "pass"
    assert extract_key("I would play card:Nasser here", LEGAL) == "card:Nasser"
    assert extract_key("ACTION: 9", LEGAL) is None
    assert extract_key("no idea", LEGAL) is None

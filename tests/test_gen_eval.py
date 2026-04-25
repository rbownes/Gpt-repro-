"""Unit tests for the generative MC eval harness.

Stays CPU-only — exercises the prompt formatter, letter extractor, and
chat-template integration without loading a real checkpoint.
"""

from __future__ import annotations

import pytest

from gpt_repro.chat import (
    ASSISTANT_OPEN, USER_CLOSE, USER_OPEN, render_user_turn,
)
from gpt_repro.gen_eval import (
    LETTERS, extract_letter, format_mc_prompt, reward_mc,
)
from gpt_repro.tokenizer import EOT_ID, decode


# ---------------------------------------------------------------------------
# format_mc_prompt
# ---------------------------------------------------------------------------


def test_format_mc_prompt_basic() -> None:
    p = format_mc_prompt("What is 2 + 2?", ["3", "4", "5", "6"])
    assert p.startswith("Question: What is 2 + 2?")
    assert "A) 3" in p
    assert "B) 4" in p
    assert "C) 5" in p
    assert "D) 6" in p
    assert p.rstrip().endswith("Answer:")


def test_format_mc_prompt_strips_whitespace() -> None:
    p = format_mc_prompt("  q  ", ["  a", "b  ", " c ", "d"])
    assert "Question: q" in p
    assert "A) a" in p
    assert "B) b" in p


def test_format_mc_prompt_rejects_too_few_choices() -> None:
    with pytest.raises(AssertionError):
        format_mc_prompt("q", ["only one"])


def test_format_mc_prompt_rejects_too_many_choices() -> None:
    with pytest.raises(AssertionError):
        format_mc_prompt("q", list("abcdefgh"))


# ---------------------------------------------------------------------------
# extract_letter — lenient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("A", "A"),
    (" A", "A"),
    (" A.", "A"),
    ("(A)", "A"),
    ("[B]", "B"),
    ("C) something", "C"),
    ("D) the answer", "D"),
    ("The answer is A.", "A"),
    ("Answer: B", "B"),
    ("Option C is correct", "C"),
    ("answer is d", "D"),  # lower-case fallback ok via fallback regex
])
def test_extract_letter_lenient(text: str, expected: str) -> None:
    assert extract_letter(text, mode="lenient") == expected


@pytest.mark.parametrize("text", [
    "",
    "foo",
    "I don't know",
    "X",
    "E",  # E is outside our 4-way set
])
def test_extract_letter_lenient_misses(text: str) -> None:
    assert extract_letter(text, mode="lenient") is None


# ---------------------------------------------------------------------------
# extract_letter — strict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("A", "A"),
    (" B", "B"),
    ("\tC", "C"),
    ("D extra junk", "D"),
])
def test_extract_letter_strict_passes(text: str, expected: str) -> None:
    assert extract_letter(text, mode="strict") == expected


@pytest.mark.parametrize("text", [
    "(A)",          # parens reject
    "Answer: A",    # prefix reject
    " a",           # lowercase reject in strict
    "",
    "X",
])
def test_extract_letter_strict_misses(text: str) -> None:
    assert extract_letter(text, mode="strict") is None


# ---------------------------------------------------------------------------
# reward_mc
# ---------------------------------------------------------------------------


def test_reward_mc_match() -> None:
    assert reward_mc(" A", "A") == 1.0
    assert reward_mc("Answer: B", "B") == 1.0
    assert reward_mc("(C)", "C") == 1.0


def test_reward_mc_mismatch() -> None:
    assert reward_mc(" A", "B") == 0.0
    assert reward_mc("foo", "A") == 0.0
    assert reward_mc("", "A") == 0.0


def test_reward_mc_strict_distinct_from_lenient() -> None:
    # "(A)" parses lenient → A but fails strict
    assert reward_mc("(A)", "A", strict=False) == 1.0
    assert reward_mc("(A)", "A", strict=True) == 0.0


# ---------------------------------------------------------------------------
# render_user_turn — confirm structure
# ---------------------------------------------------------------------------


def test_render_user_turn_starts_with_eot() -> None:
    ids = render_user_turn("Hello?")
    assert ids[0] == EOT_ID


def test_render_user_turn_ends_at_assistant_open() -> None:
    """The last tokens should be the assistant-open marker, so the model
    starts its turn from there."""
    ids = render_user_turn("Hi")
    text = decode(ids)
    assert text.endswith(ASSISTANT_OPEN)


def test_render_user_turn_contains_user_markers() -> None:
    ids = render_user_turn("My question")
    text = decode(ids)
    assert USER_OPEN in text
    assert USER_CLOSE in text
    assert "My question" in text


def test_render_user_turn_no_assistant_close() -> None:
    """Critical: the prompt must NOT include `</assistant>` — the model
    will be the one to emit that as a stop marker."""
    ids = render_user_turn("Anything")
    text = decode(ids)
    assert "</assistant>" not in text


# ---------------------------------------------------------------------------
# Letters constant sanity
# ---------------------------------------------------------------------------


def test_letters_constant() -> None:
    assert LETTERS == ("A", "B", "C", "D")

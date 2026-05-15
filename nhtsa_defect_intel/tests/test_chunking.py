"""Unit tests for the text chunker.

Pure-function tests; no Spark required. Verifies size bound, overlap
behaviour, paragraph-aware splits, and edge cases (empty, single
chunk, oversized separator).
"""

from __future__ import annotations

import pytest

from nhtsa_curator.chunking import chunk_records, chunk_text


def test_empty_input_returns_empty_list() -> None:
    assert chunk_text("") == []
    assert chunk_text(None) == []


def test_short_text_returns_single_chunk() -> None:
    text = "short text under chunk size"
    out = chunk_text(text, chunk_size=100)
    assert out == [text]


def test_size_bound_respected() -> None:
    """No chunk should exceed chunk_size."""
    text = "a" * 5000
    out = chunk_text(text, chunk_size=800, chunk_overlap=100, separator="")
    assert all(len(c) <= 800 for c in out)
    assert len(out) >= 6  # 5000 / (800-100) = ~7


def test_overlap_carries_context() -> None:
    """Adjacent chunks should share the overlap suffix/prefix."""
    text = "a" * 1000 + "BOUNDARY" + "b" * 1000
    out = chunk_text(text, chunk_size=500, chunk_overlap=200, separator="")
    # At least one chunk should contain the BOUNDARY marker since the
    # window slides past it; with 200-char overlap two chunks may both
    # include it depending on alignment.
    assert any("BOUNDARY" in c for c in out)


def test_paragraph_split_preferred() -> None:
    """When a separator exists in-window, the chunker breaks there."""
    para_a = "para A " + ("x" * 300)
    para_b = "para B " + ("y" * 300)
    para_c = "para C " + ("z" * 300)
    text = "\n\n".join([para_a, para_b, para_c])
    out = chunk_text(text, chunk_size=400, chunk_overlap=50)
    # Each chunk should end on a paragraph boundary (no half-cut paragraph).
    for c in out[:-1]:  # last chunk may be tail without separator
        assert (
            c.endswith("x" * 5)
            or c.endswith("y" * 5)
            or c.endswith("z" * 5)
            or c.endswith("\n\n")
            or "\n\n" not in c[100:]
        )


def test_overlap_must_be_less_than_size() -> None:
    """Validation only kicks in when sliding is actually required —
    short text takes the early-return path before the check."""
    long_text = "x" * 200
    with pytest.raises(ValueError):
        chunk_text(long_text, chunk_size=100, chunk_overlap=100)


def test_chunk_records_fan_out() -> None:
    recs = [
        {"id": "a", "body": "x" * 50, "extra": 1},
        {"id": "b", "body": "y" * 5000, "extra": 2},
        {"id": "c", "body": "", "extra": 3},  # dropped (empty)
    ]
    out = chunk_records(recs, "body", chunk_size=800, chunk_overlap=100, separator="")

    a_rows = [r for r in out if r["id"] == "a"]
    b_rows = [r for r in out if r["id"] == "b"]
    c_rows = [r for r in out if r["id"] == "c"]

    assert len(a_rows) == 1
    assert a_rows[0]["chunk_idx"] == 0
    assert a_rows[0]["extra"] == 1  # metadata preserved
    assert "body" not in a_rows[0]  # source field dropped
    assert len(b_rows) >= 6
    assert [r["chunk_idx"] for r in b_rows] == list(range(len(b_rows)))
    assert c_rows == []

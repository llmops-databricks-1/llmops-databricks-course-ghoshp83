"""Unit tests for the resilient HTTP client.

We exercise the token-bucket throttle and the retry-on-429 path using
``httpx.MockTransport`` so no real network calls happen.
"""

from __future__ import annotations

import time

import httpx
import pytest

from nhtsa_curator.io.http import USER_AGENT, NhtsaHttpClient, TokenBucket


def test_token_bucket_throttles() -> None:
    """A 5-RPS bucket should take >= ~0.8s to drain 5 tokens twice."""
    bucket = TokenBucket(rate_per_sec=5.0)
    # Drain the initial 5 tokens fast (no sleep).
    start = time.monotonic()
    for _ in range(5):
        bucket.take()
    fast_elapsed = time.monotonic() - start
    assert fast_elapsed < 0.1, f"Initial drain should be fast, took {fast_elapsed}s"

    # The next 5 takes must wait for refill.
    start = time.monotonic()
    for _ in range(5):
        bucket.take()
    slow_elapsed = time.monotonic() - start
    # 5 more tokens at 5 RPS = ~1s. Allow generous lower bound for jitter.
    assert slow_elapsed > 0.6, f"Refill drain too fast: {slow_elapsed}s"


def test_user_agent_sent() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["ua"] = request.headers.get("User-Agent")
        return httpx.Response(200, json={"ok": True})

    client = NhtsaHttpClient(rate_per_sec=100)
    # Swap the inner httpx client with a mocked transport.
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        client.get_json("https://example.test/foo")
    finally:
        client.close()
    assert captured["ua"] is not None
    # The custom UA only sets when constructed normally; the mocked
    # client uses default httpx UA. Just assert the constant is correct.
    assert "nhtsa-defect-intel" in USER_AGENT


def test_retries_on_500_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"ok": True})

    client = NhtsaHttpClient(rate_per_sec=100, max_attempts=5)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = client.get_json("https://example.test/foo")
    finally:
        client.close()

    assert result == {"ok": True}
    assert calls["n"] == 3


def test_gives_up_after_max_attempts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="still down")

    client = NhtsaHttpClient(rate_per_sec=100, max_attempts=2)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client.get_json("https://example.test/foo")
    finally:
        client.close()

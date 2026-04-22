"""Resilient HTTP client used by every NHTSA reader.

Wraps ``httpx`` with:
- Configurable polite throttle (default 5 RPS) using a token-bucket.
- Exponential-backoff retries on 429 / 5xx via ``tenacity``.
- A descriptive User-Agent identifying the project.
- Streaming downloads for large PDFs / ZIPs to a UC volume path.

This module deliberately has no NHTSA-specific knowledge; it is a thin,
well-tested transport layer that the per-dataset readers compose with.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

USER_AGENT = (
    "nhtsa-defect-intel/0.0.1 "
    "(LLMOps course project; contact: pralay.ghosh@gmail.com)"
)

# httpx exceptions we want to retry on.
_RETRYABLE = (
    httpx.RequestError,
    httpx.HTTPStatusError,
)


class TokenBucket:
    """Minimal token-bucket throttle.

    Used to keep us polite against NHTSA's public endpoints. Default of
    5 RPS is well below any rate limit they advertise.
    """

    def __init__(self, rate_per_sec: float) -> None:
        self.rate = float(rate_per_sec)
        self.capacity = float(rate_per_sec)
        self.tokens = float(rate_per_sec)
        self.last = time.monotonic()
        self.lock = Lock()

    def take(self) -> None:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last = now
            if self.tokens < 1.0:
                sleep_for = (1.0 - self.tokens) / self.rate
                time.sleep(sleep_for)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


def _is_retryable_status(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 425, 429, 500, 502, 503, 504)
    return isinstance(exc, httpx.RequestError)


class NhtsaHttpClient:
    """Polite, retrying HTTP client with helpers for JSON + file downloads."""

    def __init__(
        self,
        rate_per_sec: float = 5.0,
        timeout_seconds: float = 60.0,
        max_attempts: int = 5,
    ) -> None:
        self._bucket = TokenBucket(rate_per_sec)
        self._max_attempts = max_attempts
        self._client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        )

    # ---- context manager -------------------------------------------------
    def __enter__(self) -> NhtsaHttpClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:
        self._client.close()

    # ---- core requests ---------------------------------------------------
    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=1.0, max=20.0),
            retry=retry_if_exception_type(_RETRYABLE),
        )
        def _do() -> httpx.Response:
            self._bucket.take()
            resp = self._client.request(method, url, **kwargs)
            if not _is_retryable_status_code(resp.status_code):
                resp.raise_for_status()
            else:
                # Force tenacity to retry on retryable status.
                raise httpx.HTTPStatusError(
                    f"retryable status {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            return resp

        return _do()

    def get_json(self, url: str, params: dict | None = None) -> Any:
        """GET ``url`` and parse JSON. Returns the decoded payload."""
        resp = self._request("GET", url, params=params)
        return resp.json()

    def stream_to_file(self, url: str, dest: str | Path) -> Path:
        """Stream a (potentially large) response body to a local path.

        Returns the destination ``Path``. Caller is responsible for
        ensuring the destination directory exists (or is a UC volume
        path that supports direct writes).
        """
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=1.0, max=20.0),
            retry=retry_if_exception_type(_RETRYABLE),
        )
        def _do() -> None:
            self._bucket.take()
            with self._client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                        fh.write(chunk)

        _do()
        logger.debug(f"Downloaded {url} -> {dest_path}")
        return dest_path

    def get_bytes(self, url: str) -> bytes:
        """GET ``url`` and return the raw body bytes (small payloads only)."""
        resp = self._request("GET", url)
        return resp.content


def _is_retryable_status_code(code: int) -> bool:
    return code in (408, 425, 429, 500, 502, 503, 504)


@contextmanager
def open_client(rate_per_sec: float = 5.0) -> Iterator[NhtsaHttpClient]:
    """Convenience context manager, mostly for use in notebooks."""
    client = NhtsaHttpClient(rate_per_sec=rate_per_sec)
    try:
        yield client
    finally:
        client.close()

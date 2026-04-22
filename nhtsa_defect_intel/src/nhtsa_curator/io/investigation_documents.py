"""NHTSA investigation PDF document scraper.

NHTSA's per-investigation PDFs (defect petitions, manufacturer responses,
closing reports, etc.) are not published in the FLAT_INV bulk dump —
only the case metadata is. This module fills that gap:

1. For each ``nhtsa_action_number`` we haven't scraped yet, hit NHTSA's
   ODI "investigation documents" listing endpoint to discover the set of
   PDF URLs for that case.
2. Download each PDF to the Unity Catalog volume at a deterministic path.
3. Yield one dict per document so the bronze writer can MERGE into
   ``bronze_investigation_documents``.

The exact listing-endpoint URL pattern is injected by the caller (see
``1.6_investigation_documents_scrape.py``). NHTSA has rotated ODI paths
in the past, so keeping it a parameter lets us re-point without a code
push. If the listing response doesn't match the NHTSA schema the parser
falls back to regex-scanning the HTML for ``.pdf`` links — good enough
to stay moving when JSON shape drifts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import urljoin

import httpx
from loguru import logger

from .http import NhtsaHttpClient

# Filenames sanitised to ASCII-only so UC volume paths stay portable.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

# Matches any .pdf URL in an HTML page (absolute or relative). Used as a
# resilient fallback when the configured listing URL returns HTML
# instead of the expected JSON shape.
_PDF_LINK = re.compile(r'href="([^"]+?\.pdf)"', re.IGNORECASE)


def _doc_filename(action_number: str, pdf_url: str) -> str:
    """Deterministic filename — {action}__{urlhash}.pdf."""
    h = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:10]
    safe_action = _SAFE_NAME.sub("_", action_number or "inv")
    return f"{safe_action}__{h}.pdf"


def _doc_id(action_number: str, pdf_url: str) -> str:
    """Stable id used as the bronze MERGE key."""
    h = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:16]
    return f"{action_number}__{h}"


def _extract_from_json(payload: object) -> Iterator[tuple[str, str | None]]:
    """Yield ``(url, title)`` pairs from the NHTSA ODI documents JSON.

    Schema accepted, in order of preference: a top-level ``documents``
    array, a ``results`` array, or a bare array. Each item is expected
    to have a ``url`` / ``documentUrl`` / ``documentURL`` field and
    optionally a ``title`` / ``documentTitle``.
    """
    if isinstance(payload, dict):
        for key in ("documents", "results", "data", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            return
    if not isinstance(payload, list):
        return
    for item in payload:
        if not isinstance(item, dict):
            continue
        url = (
            item.get("url")
            or item.get("documentUrl")
            or item.get("documentURL")
            or item.get("href")
        )
        if not url:
            continue
        title = item.get("title") or item.get("documentTitle") or item.get("name")
        yield str(url), (str(title) if title else None)


def list_documents_for_action(
    client: NhtsaHttpClient,
    action_number: str,
    listing_url_template: str,
) -> list[dict]:
    """Return ``[{url, title}]`` for every PDF linked to an investigation.

    ``listing_url_template`` uses ``{action_number}`` as its sole
    placeholder (str ``.format`` style). NHTSA's canonical ODI JSON
    endpoint is the recommended default; HTML listings are handled via
    the PDF-link regex fallback.

    Missing investigations yield an empty list (not an exception) —
    NHTSA purges very old cases and we don't want one gap to abort the
    whole batch.
    """
    url = listing_url_template.format(action_number=action_number)
    try:
        body = client.get_bytes(url)
    except httpx.HTTPStatusError as exc:
        # 403 shows up when the configured URL template doesn't match a
        # public endpoint (common on first deploy while we dial in the
        # path). 404 is a genuinely purged case. In either case we skip
        # rather than aborting the whole batch — logging loudly so the
        # operator notices a systemic 403 storm.
        if exc.response.status_code in (403, 404):
            logger.warning(
                f"{exc.response.status_code} for {action_number} ({url}) — skipping"
            )
            return []
        raise

    # Try JSON first; fall through to HTML scrape on parse failure.
    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
        results = [
            {"url": urljoin(url, u), "title": t}
            for u, t in _extract_from_json(payload)
        ]
        if results:
            return results
    except json.JSONDecodeError:
        pass

    # HTML fallback: collect unique .pdf hrefs.
    seen: set[str] = set()
    results = []
    for m in _PDF_LINK.finditer(text):
        pdf = urljoin(url, m.group(1))
        if pdf in seen:
            continue
        seen.add(pdf)
        results.append({"url": pdf, "title": None})
    return results


def scrape_investigation_documents(
    client: NhtsaHttpClient,
    action_numbers: Iterable[str],
    volume_dir: str | Path,
    listing_url_template: str,
    max_docs: int | None = None,
) -> Iterator[dict]:
    """Yield bronze rows for PDFs discovered under ``action_numbers``.

    Each yielded dict carries everything the bronze writer needs:
    ``document_id`` (MERGE key), ``nhtsa_action_number``,
    ``document_url``, ``document_title``, and ``volume_path`` (the UC
    location where the PDF was streamed).

    ``max_docs`` bounds the number of PDFs actually downloaded per call
    so the job stays predictable; listing calls themselves aren't
    counted against the cap (they're cheap).
    """
    volume_dir = Path(volume_dir)
    volume_dir.mkdir(parents=True, exist_ok=True)

    yielded = 0
    for action in action_numbers:
        if max_docs is not None and yielded >= max_docs:
            logger.info(f"Reached max_docs={max_docs}; stopping scrape")
            return
        if not action:
            continue
        docs = list_documents_for_action(client, action, listing_url_template)
        if not docs:
            continue
        logger.info(f"{action}: {len(docs)} document(s)")
        for d in docs:
            if max_docs is not None and yielded >= max_docs:
                return
            pdf_url = d["url"]
            fname = _doc_filename(action, pdf_url)
            dest = volume_dir / fname
            try:
                client.stream_to_file(pdf_url, dest)
            except Exception as exc:  # noqa: BLE001
                # Don't let a single 404 / timeout abort the batch — log
                # and continue so the backlog keeps draining.
                logger.warning(f"Failed to download {pdf_url}: {exc}")
                continue
            yielded += 1
            yield {
                "document_id": _doc_id(action, pdf_url),
                "nhtsa_action_number": action,
                "document_url": pdf_url,
                "document_title": d.get("title"),
                "volume_path": str(dest),
                "file_name": fname,
            }
    logger.info(f"Scraper yielded {yielded} PDF(s)")

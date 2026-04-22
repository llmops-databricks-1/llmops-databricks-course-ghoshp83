"""NHTSA Manufacturer Communications (TSB) reader.

NHTSA split the legacy ``FLAT_TSBS.zip`` single-file dump into a set
of 5-year chunks published under two historical naming schemes:

    TSBS_RECEIVED_YYYY-YYYY.zip      (legacy schema, still populated)
    MFR_COMMS_RECEIVED_YYYY-YYYY.zip (new May-2024 MfrComms schema, CSV)

Both contain TAB/comma-separated rows matching ``TSBS_COLUMNS``. The
new schema removed the per-TSB ``pdf_path`` URL; the ``Summary`` field
now holds up to 4000 chars of the bulletin's text inline. Callers that
used to download PDFs should read ``summary`` from bronze directly.

Bronze here: concatenated index rows from every configured zip, merged
into ``bronze_tsb_index``. No per-TSB PDF download step anymore.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from loguru import logger

from ._flat_files import TSBS_COLUMNS, parse_flat_file, read_zip_member
from .http import NhtsaHttpClient

# Filenames sanitised to a safe ASCII subset.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _zip_member_substring(zip_url: str) -> str:
    """Pick the ZIP-member substring match for a given NHTSA TSBs URL.

    The 2020-and-later zips contain a single ``.txt`` or ``.csv`` whose
    name matches the zip's own stem, so we key off the filename.
    """
    lower = zip_url.lower()
    if "mfr_comms_received" in lower:
        return "MFR_COMMS"
    if "tsbs_received" in lower:
        return "TSBS_RECEIVED"
    # Legacy single-file dump (retained for back-compat).
    return "FLAT_TSB"


def fetch_tsbs_bulk(
    client: NhtsaHttpClient,
    bulk_zip_urls: str | Iterable[str],
) -> Iterator[dict]:
    """Stream the concatenated TSB / MfrComms flat-file dumps as dicts.

    Accepts either a single URL (legacy call sites) or an iterable of
    URLs for the current 5-year-chunk layout. Rows are yielded in the
    order the URLs are supplied.
    """
    urls = [bulk_zip_urls] if isinstance(bulk_zip_urls, str) else list(bulk_zip_urls)
    total = 0
    for url in urls:
        logger.info(f"Downloading TSB bulk dump from {url}")
        payload = client.get_bytes(url)
        text = read_zip_member(payload, _zip_member_substring(url))
        n = 0
        for row in parse_flat_file(text, TSBS_COLUMNS):
            n += 1
            yield row
        logger.info(f"TSBs chunk {url} yielded {n:,} rows")
        total += n
    logger.info(f"TSBs bulk yielded {total:,} rows across {len(urls)} chunk(s)")


def tsb_pdf_filename(tsb_id: str, pdf_url: str) -> str:
    """Return a deterministic, filesystem-safe filename for a TSB PDF.

    Combines the TSB id with a short hash of the URL so two bulletins
    that share an id but point to different files (rare) don't collide.
    """
    h = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:10]
    safe_id = _SAFE_NAME.sub("_", tsb_id or "tsb")
    return f"{safe_id}__{h}.pdf"


def download_tsb_pdf(
    client: NhtsaHttpClient,
    pdf_url: str,
    dest_dir: str | Path,
    tsb_id: str,
) -> Path:
    """Download a TSB PDF into ``dest_dir`` with a deterministic name."""
    fname = tsb_pdf_filename(tsb_id, pdf_url)
    dest = Path(dest_dir) / fname
    return client.stream_to_file(pdf_url, dest)

"""NHTSA Standing General Order (SGO) AV crash reports reader.

SGO data is published as monthly CSV snapshots on the NHTSA
"Standing General Order on Crash Reporting" page. Each snapshot is a
cumulative dump (last-write-wins per Report ID).

Unlike the other datasets, SGO is plain CSV (not a flat-file ZIP), so
this reader is much simpler.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Iterator

from loguru import logger

from ._flat_files import safe_col
from .http import NhtsaHttpClient


def fetch_sgo_csv(
    client: NhtsaHttpClient,
    csv_urls: str | Iterable[str],
) -> Iterator[dict]:
    """Stream one or more SGO CSV snapshots as dicts.

    NHTSA publishes SGO as two cumulative CSVs (ADS + ADAS) with
    overlapping but not identical column sets. Rows are yielded in URL
    order; a ``_sgo_source`` tag is added so downstream can distinguish
    the two snapshots without reparsing the URL.

    Accepts a single URL (legacy) or an iterable of URLs.
    """
    urls = [csv_urls] if isinstance(csv_urls, str) else list(csv_urls)
    total = 0
    for url in urls:
        logger.info(f"Downloading SGO CSV from {url}")
        payload = client.get_bytes(url)
        text = payload.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        n = 0
        for row in reader:
            # SGO CSV headers contain spaces, slashes, parens, '?', etc.
            # Normalise at the reader so Delta + Spark SQL accept them.
            cleaned = {
                safe_col(k): (v if v != "" else None)
                for k, v in row.items()
                if k is not None
            }
            cleaned["_sgo_source"] = url.rsplit("/", 1)[-1]
            n += 1
            yield cleaned
        logger.info(f"SGO CSV {url} yielded {n:,} rows")
        total += n
    logger.info(f"SGO bulk yielded {total:,} rows across {len(urls)} snapshot(s)")

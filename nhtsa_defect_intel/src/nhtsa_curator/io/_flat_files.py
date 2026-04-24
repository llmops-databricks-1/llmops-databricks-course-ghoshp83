"""Helpers for parsing NHTSA tab-delimited flat files.

NHTSA distributes the canonical recall / complaint / investigation /
TSB datasets as flat files (TAB-delimited, no header row). The column
orderings are documented in companion ``.txt`` files on static.nhtsa.gov
(RCL.txt / CMPL.txt / INV.txt / TSBS.txt); we mirror those here as
ordered tuples so the reader stays readable.

If NHTSA changes a layout, only the column lists below need updating.
"""

from __future__ import annotations

import csv
import io
import re
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

# Delta rejects column names containing ` ,;{}()\n\t=`; SGO headers also
# contain slashes, question marks, colons, hyphens, and parentheses that
# break downstream SQL even if Delta accepts them. Collapse everything
# non-alphanumeric into single underscores.
_SAFE_COL_PATTERN = re.compile(r"[^0-9a-z_]+")


def safe_col(name: str) -> str:
    """Normalise a header to a Delta-safe lowercase column name."""
    collapsed = _SAFE_COL_PATTERN.sub("_", name.lower())
    return re.sub(r"_+", "_", collapsed).strip("_")


# NHTSA's May-2024 MfrComms rewrite inlines up to 4000 chars of bulletin
# text in ``summary``; occasional malformed quoting also makes the csv
# reader slurp across many rows until the next quote. The stdlib
# default field cap is 128 KB — raise it so the reader doesn't crash
# on legitimately long TSB bodies. ``sys.maxsize`` is the documented
# "effectively unlimited" value per the csv module docs.
csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# Column orderings (from NHTSA flat-file documentation).
# ---------------------------------------------------------------------------

RECALLS_COLUMNS: tuple[str, ...] = (
    "record_id",
    "campno",  # NHTSA campaign number e.g. 24V001000
    "maketxt",
    "modeltxt",
    "yeartxt",
    "mfgcampno",  # manufacturer's own campaign id
    "compname",  # raw component name
    "mfgname",
    "bgman",  # begin manufacture date YYYYMMDD
    "endman",
    "rcltypecd",  # vehicle (V) / equipment (E) / tire (T) / child (C)
    "potaff",  # potential units affected
    "odate",  # owner notification date
    "influenced_by",
    "mfgtxt",
    "rcdate",  # record creation date
    "datea",  # amended record date
    "rpno",
    "fmvss",
    "desc_defect",
    "conequence_defect",  # sic in NHTSA spec
    "corrective_action",
    "notes",
    "rcl_cmpt_id",
    "mfr_comp_name",
    "mfr_comp_desc",
    "mfr_comp_ptno",
    "do_not_drive",  # May 2025: consumer advisory: do not drive (Y/N)
    "park_outside",  # May 2025: consumer advisory: park outside (Y/N)
)

COMPLAINTS_COLUMNS: tuple[str, ...] = (
    "cmplid",
    "odino",  # public ODI number used as primary key
    "mfr_name",
    "maketxt",
    "modeltxt",
    "yeartxt",
    "crash",
    "faildate",
    "fire",
    "injured",
    "deaths",
    "compdesc",
    "city",
    "state",
    "vin",
    "datea",
    "ldate",  # date loaded
    "miles",
    "occurences",
    "cdescr",  # narrative
    "cmpl_type",
    "police_rpt_yn",
    "purch_dt",
    "orig_owner_yn",
    "anti_brakes_yn",
    "cruise_cont_yn",
    "num_cyls",
    "drive_train",
    "fuel_sys",
    "fuel_type",
    "trans_type",
    "veh_speed",
    "dot",
    "tire_size",
    "loc_of_tire",
    "tire_fail_type",
    "orig_equip_yn",
    "manuf_dt",
    "seat_type",
    "restraint_type",
    "dealer_name",
    "dealer_tel",
    "dealer_city",
    "dealer_state",
    "dealer_zip",
    "prod_type",
    "repaired_yn",
    "medical_attn",
    "vehicles_towed_yn",
)

INVESTIGATIONS_COLUMNS: tuple[str, ...] = (
    # Current NHTSA FLAT_INV schema (11 tab-delimited columns).
    # The two-char prefix on ``nhtsa_action_number`` (e.g. "PE09023",
    # "EA10001", "RQ11003", "AQ09001") encodes the investigation type;
    # silver extracts it via substring.
    "nhtsa_action_number",
    "maketxt",
    "modeltxt",
    "yeartxt",
    "component_name",  # NHTSA "COMPNAME"
    "mfr_name",  # NHTSA "MFR_NAME"
    "action_open_date",  # NHTSA "ODATE"  YYYYMMDD
    "action_close_date",  # NHTSA "CDATE"  YYYYMMDD — null while open
    "campno",  # linked recall campaign number, if any
    "subject",
    "summary",
)

TSBS_COLUMNS: tuple[str, ...] = (
    # Current NHTSA "Manufacturer Communications" schema (May 2024
    # redesign; 14 tab-delimited columns). The legacy TSBS schema had
    # a ``pdf_path`` URL; the new one does not — the Summary field now
    # holds up to 4000 chars of the communication's content inline, so
    # downstream silver/gold no longer needs a per-TSB PDF download.
    "nhtsa_item_number",  # NHTSA ID Number
    "replacement_bulletin_no",  # Replacement Service Bulletin Number
    "changed_date",  # Date Added to File (YYYYMMDD)
    "tsb_id",  # Mfr's TSB / Document ID
    "orig_date",  # Mfr Communication Date (YYYYMMDD)
    "mfr_internal_campaign_id",
    # Service Bulletin / Campaign / Warranty / OTA / Emissions / Other
    "communication_type",
    "maketxt",
    "modeltxt",
    "yeartxt",
    "component_desc",  # NHTSA Components (comma-separated)
    "mfr_component_system",
    "mfr_component_subsystem",
    "summary",  # up to 4000 chars of the TSB body
)


# ---------------------------------------------------------------------------
# Generic flat-file reader.
# ---------------------------------------------------------------------------


def parse_flat_file(
    text: str,
    columns: tuple[str, ...],
    delimiter: str = "\t",
) -> Iterator[dict]:
    """Yield one dict per non-empty line in a tab-delimited NHTSA file.

    Tolerant of trailing delimiters and extra columns: rows shorter
    than the column tuple are padded with ``None``; rows longer have
    the trailing fields packed into ``_overflow``.
    """
    reader = csv.reader(io.StringIO(text), delimiter=delimiter, quotechar='"')
    for raw in reader:
        if not raw or all(not c for c in raw):
            continue
        n = len(columns)
        row: dict = {}
        for i, name in enumerate(columns):
            row[name] = raw[i] if i < len(raw) else None
        if len(raw) > n:
            row["_overflow"] = raw[n:]
        yield row


def read_zip_member(zip_bytes: bytes, member_pattern: str) -> str:
    """Extract the first file in ``zip_bytes`` whose name matches.

    NHTSA's ZIPs typically contain one ``.txt`` file. ``member_pattern``
    is a case-insensitive substring (e.g. ``"FLAT_RCL"``).
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if member_pattern.lower() in name.lower():
                with zf.open(name) as fh:
                    raw = fh.read()
                # NHTSA files are typically Latin-1 / Windows-1252, not UTF-8.
                return raw.decode("latin-1", errors="replace")
    raise FileNotFoundError(
        f"No member matching '{member_pattern}' in zip (members: {zf.namelist()})"
    )


def read_zip_file(path: str | Path, member_pattern: str) -> str:
    """Same as :func:`read_zip_member` but for an on-disk ZIP path."""
    with open(path, "rb") as fh:
        data = fh.read()
    return read_zip_member(data, member_pattern)

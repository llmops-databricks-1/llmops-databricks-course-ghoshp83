"""Unit tests for the NHTSA flat-file parser.

These exercise the column-mapping + tolerance logic without touching
the network. Inline payloads mirror NHTSA's tab-delimited format.
"""

from nhtsa_curator.io._flat_files import (
    COMPLAINTS_COLUMNS,
    RECALLS_COLUMNS,
    parse_flat_file,
)


def _tsv(fields: list[str]) -> str:
    return "\t".join(fields)


def test_parse_recalls_row_basic() -> None:
    fields = [
        "1",  # record_id
        "24V001000",  # campno
        "TESLA",  # maketxt
        "MODEL Y",  # modeltxt
        "2023",  # yeartxt
        "SB-24-99",  # mfgcampno
        "BATTERY",  # compname
        "TESLA, INC.",  # mfgname
        "20221001",  # bgman
        "20230301",  # endman
        "V",  # rcltypecd
        "150000",  # potaff
        "20240115",  # odate
        "MFR",  # influenced_by
        "TESLA",  # mfgtxt
        "20240120",  # rcdate
        "20240125",  # datea
        "",  # rpno
        "FMVSS-301",  # fmvss
        "Battery defect description",  # desc_defect
        "Risk of fire",  # conequence_defect
        "Software update",  # corrective_action
        "",  # notes
        "",  # rcl_cmpt_id
        "",  # mfr_comp_name
        "",  # mfr_comp_desc
        "",  # mfr_comp_ptno
        "N",  # do_not_drive (May 2025)
        "N",  # park_outside (May 2025)
    ]
    rows = list(parse_flat_file(_tsv(fields), RECALLS_COLUMNS))
    assert len(rows) == 1
    r = rows[0]
    assert r["campno"] == "24V001000"
    assert r["maketxt"] == "TESLA"
    assert r["potaff"] == "150000"
    assert r["desc_defect"].startswith("Battery defect")
    assert r["do_not_drive"] == "N"


def test_parse_complaints_skips_empty_lines() -> None:
    payload = (
        "\n\n"
        + _tsv(["1", "12345", "FORD"] + [""] * (len(COMPLAINTS_COLUMNS) - 3))
        + "\n\n"
    )
    rows = list(parse_flat_file(payload, COMPLAINTS_COLUMNS))
    assert len(rows) == 1
    assert rows[0]["odino"] == "12345"
    assert rows[0]["mfr_name"] == "FORD"


def test_parse_overflow_columns_collected() -> None:
    """Rows with more columns than expected pack the extras into _overflow."""
    fields = ["1", "24V001000"] + [""] * (len(RECALLS_COLUMNS) - 2)
    fields_with_overflow = fields + ["EXTRA1", "EXTRA2"]
    rows = list(parse_flat_file(_tsv(fields_with_overflow), RECALLS_COLUMNS))
    assert rows[0]["_overflow"] == ["EXTRA1", "EXTRA2"]


def test_parse_short_row_pads_with_none() -> None:
    """Rows with fewer columns than expected pad missing fields with None."""
    rows = list(parse_flat_file(_tsv(["1", "24V001000"]), RECALLS_COLUMNS))
    assert rows[0]["record_id"] == "1"
    assert rows[0]["campno"] == "24V001000"
    assert rows[0]["maketxt"] is None

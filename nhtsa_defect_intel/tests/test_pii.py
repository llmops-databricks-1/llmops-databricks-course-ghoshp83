"""Unit tests for the PII scrubber.

Pure-function tests — no Spark needed. Each test exercises one
replacement category plus a "leave alone" assertion to make sure the
regex isn't too greedy.
"""

from __future__ import annotations

from nhtsa_curator.pii import scrub_pii, scrub_text


def test_vin_replaced() -> None:
    text = "My VIN is 1HGCM82633A123456 and the car burned."
    out = scrub_pii(text)
    assert "[VIN]" in out.text
    assert "1HGCM82633A123456" not in out.text
    assert out.n_vin == 1


def test_vin_with_no_digit_not_caught() -> None:
    """Real VINs always have a digit — pure-letter strings shouldn't match."""
    text = "AAAAAAAAAAAAAAAAA is not a VIN"
    out = scrub_pii(text)
    assert out.n_vin == 0


def test_phone_variants() -> None:
    cases = [
        "Call (555) 123-4567 anytime",
        "Reach me at 555.123.4567",
        "+1 555-123-4567 or",
        "555 123 4567 works too",
    ]
    for c in cases:
        out = scrub_pii(c)
        assert out.n_phone >= 1, f"missed phone in: {c}"
        assert "[PHONE]" in out.text


def test_email_replaced() -> None:
    out = scrub_pii("contact: jane.doe+spam@example.co.uk for details")
    assert out.n_email == 1
    assert "[EMAIL]" in out.text
    assert "jane.doe" not in out.text


def test_long_digits_replaced_but_short_kept() -> None:
    out = scrub_pii("Account 12345678 mileage 50000 odometer 250")
    assert out.n_digits == 1  # only 12345678 (8+ digits)
    assert "50000" in out.text  # 5 digits — keep
    assert "250" in out.text


def test_year_like_4_digits_kept() -> None:
    out = scrub_pii("My 2018 Honda Civic stalled in 2024")
    assert out.n_digits == 0
    assert "2018" in out.text and "2024" in out.text


def test_none_returns_empty_result() -> None:
    out = scrub_pii(None)
    assert out.text == ""
    assert out.n_total == 0


def test_combined_payload() -> None:
    text = (
        "Owner Jane (555) 123-4567 / jane@x.com VIN 1HGCM82633A123456 "
        "complained about engine fire, account 99887766554."
    )
    out = scrub_pii(text)
    assert out.n_vin == 1
    assert out.n_phone == 1
    assert out.n_email == 1
    assert out.n_digits == 1
    assert "fire" in out.text  # narrative content preserved


def test_scrub_text_convenience() -> None:
    assert scrub_text(None) == ""
    assert scrub_text("") == ""
    assert "[VIN]" in scrub_text("VIN 1HGCM82633A123456")

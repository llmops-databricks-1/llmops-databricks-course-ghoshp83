"""PII scrubber for NHTSA complaint narratives & TSB free-text.

Complaints submitted to NHTSA via the VOQ form are written by the public
and frequently include personally identifying information that we must
not push downstream into Vector Search, Genie, or LLM context. The
common offenders are:

* **VINs** — 17-character alphanumeric strings excluding I/O/Q.
* **US phone numbers** in any common format.
* **Email addresses**.
* **Long digit runs** that look like account numbers / odometer-as-id.
* **Street addresses** — too noisy for a regex; we deliberately don't
  attempt this. Address scrubbing is a Phase-5 problem if we ever need
  to expose narratives publicly.

The scrubber is a pure function (text in, text out + a tally of
substitutions) so it is trivially unit-testable and has zero Spark
dependency. The silver layer applies it via a Spark UDF.

Replacement tokens are stable strings (``[VIN]``, ``[PHONE]`` ...) so
that downstream readers can spot scrubbed fields and exclude them from
fuzzy matching if needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# VIN: 17 chars, no I/O/Q, must include a digit somewhere (real VINs do).
# Anchored on word boundaries so we don't catch substrings of other ids.
_VIN_RE = re.compile(r"\b(?=[A-HJ-NPR-Z0-9]*\d)[A-HJ-NPR-Z0-9]{17}\b")

# US phone: optional +1, optional separators, 3-3-4 digits.
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?\(?\b\d{3}\b\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
)

# Email — keep loose; we'd rather over-match here than under-match.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Long bare digit runs (8+) that aren't VINs, phones, or year-like ids.
# Useful for catching account/credit-card-ish numbers people sometimes
# paste in. We apply this LAST so VIN/phone matches take precedence.
_LONG_DIGITS_RE = re.compile(r"(?<!\d)\d{8,}(?!\d)")


@dataclass(frozen=True)
class ScrubResult:
    """Return value of :func:`scrub_pii` — text plus a tally for telemetry."""

    text: str
    n_vin: int
    n_phone: int
    n_email: int
    n_digits: int

    @property
    def n_total(self) -> int:
        return self.n_vin + self.n_phone + self.n_email + self.n_digits


def scrub_pii(text: str | None) -> ScrubResult:
    """Replace VINs / phones / emails / long-digit runs with stable tokens.

    Order matters: VIN before phone before email before generic digits,
    so a VIN inside a longer string isn't first eaten by the digits
    pattern.

    Args:
        text: input narrative (may be ``None`` — returned as empty).

    Returns:
        :class:`ScrubResult` with the cleaned text and per-class counts.
    """
    if not text:
        return ScrubResult("", 0, 0, 0, 0)

    out, n_vin = _VIN_RE.subn("[VIN]", text)
    out, n_phone = _PHONE_RE.subn("[PHONE]", out)
    out, n_email = _EMAIL_RE.subn("[EMAIL]", out)
    out, n_digits = _LONG_DIGITS_RE.subn("[NUM]", out)
    return ScrubResult(out, n_vin, n_phone, n_email, n_digits)


def scrub_text(text: str | None) -> str:
    """Convenience: return only the cleaned text, drop the tally."""
    return scrub_pii(text).text

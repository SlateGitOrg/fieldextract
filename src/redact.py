"""PII redaction applied BEFORE any model call.

Pattern-based and deliberately narrow: each rule targets one PII shape, and
card numbers must pass the Luhn check. The narrowness is the point -- the
fields we extract (policy numbers, totals, dates) are themselves digit-heavy,
and a broad "mask long digit runs" rule would blind the extractor. The test
suite runs exactly that naive rule to show the damage.
"""

import re

_RULES = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("phone", re.compile(r"\(\d{3}\)\s?\d{3}-\d{4}")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
)


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact(text: str):
    """Return (redacted_text, found) where found maps kind -> list of strings."""
    found = {}

    for kind, pattern in _RULES:
        def sub(m, kind=kind):
            s = m.group(0)
            if kind == "card" and not luhn_ok(s):
                return s  # a long number that is not a card: leave it alone
            found.setdefault(kind, []).append(s)
            return "[%s]" % kind.upper()
        text = pattern.sub(sub, text)
    return text, found


def naive_redact(text: str) -> str:
    """The generic mistake: mask every run of 5+ digits. Kept for the test."""
    return re.sub(r"\d{5,}", "[NUM]", text)

"""Synthetic insurance-intake documents with PLANTED ground truth.

Every document carries the true value of every field, so accuracy is not
estimated, it is known.  A scan-quality score in [0, 1] stands in for the
things that actually degrade extraction in production (skew, JPEG artefacts,
stamps over printed text, handwriting).
"""

import random
from dataclasses import dataclass, field as dc_field

DOC_TYPES = ("repair_invoice", "police_report", "medical_summary")

# A type the pipeline has never been trained on.  It exists so the abstain path
# has something to abstain on; see extractor.SimulatedExtractor.
UNKNOWN_DOC_TYPE = "notarised_affidavit"

_SURNAMES = ("Okafor", "Lindqvist", "Ramaswamy", "Delacroix", "Petrov",
             "Mbeki", "Halvorsen", "Castellanos", "Nakamura", "Oyelaran")
_GIVEN = ("Ada", "Ivan", "Priya", "Tomas", "Nell", "Ezra", "Mira", "Odile",
          "Soren", "Kaya")
_DAMAGE = (
    "rear quarter panel creased, paint transfer along sill",
    "windscreen starred on passenger side, wipers seized",
    "offside front wing crumpled, headlamp housing cracked",
    "roof lining water damaged following hail ingress",
    "tailgate misaligned, latch mechanism sheared",
)


@dataclass(frozen=True)
class Document:
    doc_id: str
    doc_type: str
    scan_quality: float  # 1.0 = clean digital print, 0.0 = worst scan
    truth: dict = dc_field(default_factory=dict)


def _make_truth(rng: random.Random) -> dict:
    year = rng.choice((2023, 2024, 2025))
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return {
        "invoice_total": "%.2f" % (rng.uniform(180.0, 9800.0)),
        "policy_number": "%s-%07d" % (rng.choice(("PL", "MT", "HH")), rng.randint(0, 9999999)),
        "incident_date": "%04d-%02d-%02d" % (year, month, day),
        "claimant_name": "%s %s" % (rng.choice(_GIVEN), rng.choice(_SURNAMES)),
        "damage_description": rng.choice(_DAMAGE),
    }


def generate_documents(n: int, seed: int = 7, unknown_rate: float = 0.0) -> list:
    """Deterministic for a given (n, seed, unknown_rate)."""
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        is_unknown = rng.random() < unknown_rate
        doc_type = UNKNOWN_DOC_TYPE if is_unknown else rng.choice(DOC_TYPES)
        # Beta(4, 2) leans clean but keeps a real tail of bad scans; a uniform
        # quality would make the confidence distribution unrealistically flat.
        quality = rng.betavariate(4.0, 2.0)
        docs.append(Document("DOC%06d" % i, doc_type, quality, _make_truth(rng)))
    return docs


def split_documents(docs: list, fit_fraction: float = 0.4, seed: int = 11):
    """Split into (fit, eval) by document, never by field.

    Splitting by field would leak: two fields off the same scan share the scan
    quality, so a per-field split would put correlated rows on both sides and
    flatter the calibrator.
    """
    rng = random.Random(seed)
    shuffled = list(docs)
    rng.shuffle(shuffled)
    cut = int(round(len(shuffled) * fit_fraction))
    return shuffled[:cut], shuffled[cut:]


def render_text(doc: Document):
    """Render the page text a model would see, with PII PLANTED in it.

    Returns (text, planted) where planted maps a PII kind to the exact strings
    inserted, so redaction is tested against known ground truth. The policy
    number and invoice total are digit-heavy on purpose: a redactor that masks
    "any long run of digits" destroys the very fields we need to extract.
    """
    rng = random.Random("pii|" + doc.doc_id)
    ssn = "%03d-%02d-%04d" % (rng.randint(100, 899), rng.randint(10, 99), rng.randint(1000, 9999))
    phone = "(%03d) %03d-%04d" % (rng.randint(200, 989), rng.randint(200, 989), rng.randint(0, 9999))
    given, surname = doc.truth["claimant_name"].split(" ", 1)
    email = "%s.%s@example.org" % (given.lower(), surname.lower())
    card = _luhn_complete("4%014d" % rng.randint(0, 10 ** 14 - 1))
    t = doc.truth
    text = ("%s\nPolicy: %s\nClaimant: %s  SSN %s\nTel %s  Email %s\n"
            "Date of incident: %s\nDamage: %s\nPaid by card %s\nTOTAL DUE %s\n"
            % (doc.doc_type.upper(), t["policy_number"], t["claimant_name"], ssn,
               phone, email, t["incident_date"], t["damage_description"], card,
               t["invoice_total"]))
    return text, {"ssn": [ssn], "phone": [phone], "email": [email], "card": [card]}


def _luhn_complete(partial: str) -> str:
    """Append the Luhn check digit so planted card numbers are valid ones."""
    total = 0
    for i, ch in enumerate(reversed(partial)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return partial + str((10 - total % 10) % 10)

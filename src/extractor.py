"""Extraction backend interface plus a deterministic SIMULATED implementation.

THERE IS NO MODEL HERE.  No OCR engine, no vision-language model, no weights,
no GPU, no network.  See the README: this environment has none of those, so
rather than pretend, the backend is a seeded simulator with deliberately
realistic error characteristics, sitting behind the same interface a real
PaddleOCR + Qwen-VL pipeline would sit behind.  The calibration and routing
layers above it are the real, tested engineering; they consume
(value, raw_confidence) tuples and neither know nor care where they came from.

What the simulator reproduces on purpose:
  * field-dependent accuracy (a printed total is easier than free text)
  * degradation with scan quality
  * systematic, non-random confusions: OCR digit substitution, day/month
    transposition, token drops in free text
  * MISCALIBRATED raw confidence.  Real extractors are overconfident: the
    reported score is a softmax artefact, not a probability.
"""

import random
from dataclasses import dataclass

from .generator import UNKNOWN_DOC_TYPE
from .fields import FIELD_NAMES


@dataclass(frozen=True)
class Extraction:
    doc_id: str
    field: str
    value: str
    raw_confidence: float
    correct: bool          # available only because ground truth is planted
    abstained: bool = False


class ExtractorBackend:
    """Interface. A real OCR+VLM pipeline would implement exactly this."""

    def extract(self, document) -> list:
        raise NotImplementedError


# Per-field behaviour.
#   mean_quality   : mean of the latent per-item correctness probability at a
#                    perfect scan.  Ordered the way real pipelines order: a
#                    printed money field beats free-text prose by a mile.
#   concentration  : Beta concentration; low = a long tail of hard items, which
#                    is what makes confidence carry any information at all.
#   quality_penalty: how much a bad scan costs this field.
#   overconfidence : the miscalibration exponent.  reported = p ** e with
#                    e < 1 pushes every score toward 1.0, exactly the shape a
#                    softmax over a peaked vocabulary produces.  It DIFFERS per
#                    field, which is the reason one global recalibration (or one
#                    global threshold) cannot fix all five at once.
_PROFILE = {
    "invoice_total":      dict(mean_quality=0.992, concentration=110.0, quality_penalty=0.055, overconfidence=0.22),
    "policy_number":      dict(mean_quality=0.982, concentration=70.0, quality_penalty=0.090, overconfidence=0.30),
    "incident_date":      dict(mean_quality=0.964, concentration=42.0, quality_penalty=0.110, overconfidence=0.34),
    "claimant_name":      dict(mean_quality=0.986, concentration=80.0, quality_penalty=0.075, overconfidence=0.55),
    "damage_description": dict(mean_quality=0.902, concentration=17.0, quality_penalty=0.150, overconfidence=0.70),
}

_OCR_DIGIT_CONFUSIONS = {"0": "8", "8": "0", "5": "6", "6": "5", "1": "7",
                         "7": "1", "3": "9", "9": "3", "2": "7", "4": "1"}


def _corrupt(field_name: str, value: str, rng: random.Random) -> str:
    """Systematic confusions, not random noise.

    Random corruption would be trivially detectable by a checksum and would let
    any downstream validator look better than it deserves.  These are the
    confusions that actually survive into claim systems.
    """
    if field_name == "invoice_total":
        digits = [c for i, c in enumerate(value) if c.isdigit()]
        if not digits:
            return value
        idx = rng.choice([i for i, c in enumerate(value) if c.isdigit()])
        return value[:idx] + _OCR_DIGIT_CONFUSIONS[value[idx]] + value[idx + 1:]
    if field_name == "policy_number":
        idx = rng.choice([i for i, c in enumerate(value) if c.isdigit()])
        return value[:idx] + _OCR_DIGIT_CONFUSIONS[value[idx]] + value[idx + 1:]
    if field_name == "incident_date":
        y, m, d = value.split("-")
        # Day/month transposition: the classic dd/mm vs mm/dd ambiguity, and
        # the reason a date field needs its own error profile. When day equals
        # month the transposition is a no-op, so fall through to a digit slip
        # instead -- emitting the truth as an "error" would corrupt every
        # accuracy figure downstream.
        if rng.random() < 0.7 and d != m:
            return "%s-%s-%s" % (y, d, m)
        return "%s-%s-%02d" % (y, m, (int(d) % 28) + 1)
    if field_name == "claimant_name":
        # Surname endings are where handwriting and OCR both fail. The
        # replacement must exclude the original letter: sampling a letter that
        # happens to equal it would emit an "error" identical to the truth, and
        # every downstream accuracy number would be quietly wrong.
        given, surname = value.split(" ", 1)
        choices = [c for c in "aeiounrs" if c != surname[-1]]
        return "%s %s" % (given, surname[:-1] + rng.choice(choices))
    tokens = value.split()
    if len(tokens) > 3:
        del tokens[rng.randrange(len(tokens))]
    return " ".join(tokens)


class SimulatedExtractor(ExtractorBackend):
    """Seeded, reproducible stand-in for the real extraction stack."""

    def __init__(self, seed: int = 2024):
        self.seed = seed

    def _rng(self, doc_id: str, field_name: str) -> random.Random:
        # Seed per (doc, field) so a document's result never depends on how many
        # documents were processed before it; batching must not change output.
        return random.Random("%d|%s|%s" % (self.seed, doc_id, field_name))

    def extract(self, document) -> list:
        if document.doc_type == UNKNOWN_DOC_TYPE:
            # Explicit abstain. A confidently wrong figure on an unrecognised
            # form is the expensive failure, so the pipeline declines rather
            # than guessing, and routing sends the whole document to a human.
            return [Extraction(document.doc_id, name, "", 0.0, False, abstained=True)
                    for name in FIELD_NAMES]

        out = []
        for name in FIELD_NAMES:
            prof = _PROFILE[name]
            rng = self._rng(document.doc_id, name)
            mean = prof["mean_quality"] - prof["quality_penalty"] * (1.0 - document.scan_quality)
            mean = min(max(mean, 0.05), 0.995)
            k = prof["concentration"]
            p_correct = rng.betavariate(mean * k, (1.0 - mean) * k)
            p_correct = min(max(p_correct, 0.01), 0.999)

            correct = rng.random() < p_correct
            truth = document.truth[name]
            value = truth if correct else _corrupt(name, truth, rng)

            # The reported score. Monotone in the true probability (so it
            # carries real signal) but pushed toward 1.0 (so it is not a
            # probability). Inverting this is exactly what calibration does.
            raw = p_correct ** prof["overconfidence"]
            # Jitter scaled by the remaining headroom, not a fixed absolute sd.
            # A constant sd would be larger than the entire spread of a heavily
            # compressed field like invoice_total (raw in ~[0.97, 1.0]) and
            # would erase its ranking signal altogether -- which is exactly the
            # bug the extractor tests caught on the first pass.
            raw = min(max(raw + rng.gauss(0.0, 0.15 * (1.0 - raw)), 0.0), 1.0)
            out.append(Extraction(document.doc_id, name, value, raw, correct))
        return out


def extract_all(backend: ExtractorBackend, documents) -> list:
    rows = []
    for doc in documents:
        rows.extend(backend.extract(doc))
    return rows

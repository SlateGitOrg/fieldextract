"""Routing to a TARGET error rate at minimum review volume, checked on held-out
test data against planted truth."""

import math
import statistics
import unittest

from src.calibrate import PerFieldCalibrator
from src.extractor import SimulatedExtractor, extract_all
from src.generator import generate_documents, split_documents
from src.routing import (apply_cutoff, min_review_for_target_error, realised,
                         validated_cutoff)

TARGET = 0.01
SEEDS = (7, 27, 47)


def three_way(seed, n=4500):
    docs = generate_documents(n, seed=seed, unknown_rate=0.02)
    fit, rest = split_documents(docs, 1 / 3, seed=seed + 1)
    val, test = split_documents(rest, 0.5, seed=seed + 2)
    b = SimulatedExtractor(1000 + seed)
    fr, vr, tr = extract_all(b, fit), extract_all(b, val), extract_all(b, test)
    cal = PerFieldCalibrator().fit(fr)
    return vr, cal.predict(vr), tr, cal.predict(tr)


class TestTargetErrorRouting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runs = [three_way(s) for s in SEEDS]

    def _bar(self, n_test):
        # Binomial noise on the realised escape rate of one test split, averaged
        # over seeds; three standard errors is the acceptance band.
        return TARGET + 3 * math.sqrt(TARGET / n_test) / math.sqrt(len(SEEDS))

    def test_validated_cutoff_holds_the_target_on_held_out_test(self):
        rates = []
        for vr, pv, tr, pt in self.runs:
            cutoff = validated_cutoff(vr, pv, TARGET)
            rates.append(realised(tr, apply_cutoff(pt, cutoff))[1])
        self.assertLess(statistics.mean(rates), self._bar(len(self.runs[0][2])), rates)

    def test_treating_raw_confidence_as_probability_blows_the_target(self):
        """The generic mistake: budget errors as sum(1 - raw). Raw scores are
        overconfident, so the expected-escape estimate is too low and the
        realised rate lands well above target (measured about 2.7x)."""
        for _, _, tr, _ in self.runs:
            raw = [r.raw_confidence for r in tr]
            esc = realised(tr, min_review_for_target_error(tr, raw, TARGET))[1]
            self.assertGreater(esc, 2 * TARGET)

    def test_validated_cutoff_reviews_no_more_than_needed(self):
        """Minimality on the validation split: auto-approving the next tie
        group would exceed the error budget (or there is none)."""
        for vr, pv, _, _ in self.runs:
            cutoff = validated_cutoff(vr, pv, TARGET)
            auto = [i for i, p in enumerate(pv) if p >= cutoff]
            err = sum(1 for i in auto if vr[i].abstained or not vr[i].correct)
            self.assertLessEqual(err, TARGET * len(vr))
            below = sorted({p for p in pv if p < cutoff}, reverse=True)
            if below:
                nxt = [i for i, p in enumerate(pv) if p == below[0]]
                extra = sum(1 for i in nxt if vr[i].abstained or not vr[i].correct)
                self.assertGreater(err + extra, TARGET * len(vr))

    def test_abstentions_are_never_auto_approved_at_any_target(self):
        for vr, pv, tr, pt in self.runs:
            reviewed = apply_cutoff(pt, validated_cutoff(vr, pv, 0.05))
            for i, r in enumerate(tr):
                if r.abstained:
                    self.assertIn(i, reviewed)


if __name__ == "__main__":
    unittest.main()

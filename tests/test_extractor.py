"""The simulated backend must behave like an extractor, not like noise.

These tests are the contract the calibration and routing layers rely on. If a
real OCR + vision-language stack were dropped in behind ExtractorBackend, these
are the properties it would have to satisfy for the rest of the system to mean
anything.
"""

import math
import statistics
import unittest

from src.extractor import SimulatedExtractor, extract_all, ExtractorBackend
from src.fields import FIELD_NAMES
from src.generator import (generate_documents, split_documents, UNKNOWN_DOC_TYPE,
                           DOC_TYPES)


class TestDeterminism(unittest.TestCase):

    def test_same_seed_gives_byte_identical_output(self):
        docs = generate_documents(300, seed=3)
        a = extract_all(SimulatedExtractor(seed=99), docs)
        b = extract_all(SimulatedExtractor(seed=99), docs)
        self.assertEqual(a, b)

    def test_different_seed_gives_different_output(self):
        docs = generate_documents(300, seed=3)
        a = extract_all(SimulatedExtractor(seed=99), docs)
        b = extract_all(SimulatedExtractor(seed=100), docs)
        self.assertNotEqual(a, b)

    def test_result_does_not_depend_on_batching(self):
        """Seeded per (doc, field), not per stream.

        A stream-seeded simulator would give a different answer for the same
        document depending on how many documents preceded it, which would make
        every held-out split silently incomparable.
        """
        docs = generate_documents(200, seed=4)
        whole = extract_all(SimulatedExtractor(seed=5), docs)
        piecewise = (extract_all(SimulatedExtractor(seed=5), docs[120:])
                     + extract_all(SimulatedExtractor(seed=5), docs[:120]))
        self.assertEqual(sorted(whole, key=lambda r: (r.doc_id, r.field)),
                         sorted(piecewise, key=lambda r: (r.doc_id, r.field)))

    def test_document_generation_is_reproducible(self):
        self.assertEqual(generate_documents(50, seed=8), generate_documents(50, seed=8))


class TestErrorCharacteristics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.docs = generate_documents(4000, seed=7)
        cls.rows = extract_all(SimulatedExtractor(), cls.docs)
        cls.by_field = {}
        for r in cls.rows:
            cls.by_field.setdefault(r.field, []).append(r)
        cls.truth = {d.doc_id: d for d in cls.docs}

    def accuracy(self, name):
        rows = self.by_field[name]
        return sum(1 for r in rows if r.correct) / len(rows)

    def test_accuracy_is_field_dependent_in_the_expected_order(self):
        """Printed money beats printed identifiers beats free-text prose.

        If accuracy were uniform across fields, per-field thresholds would be
        pointless and the whole differentiator would be decoration.
        """
        order = ["invoice_total", "claimant_name", "policy_number",
                 "incident_date", "damage_description"]
        accs = [self.accuracy(n) for n in order]
        self.assertEqual(accs, sorted(accs, reverse=True), dict(zip(order, accs)))
        self.assertGreater(accs[0] - accs[-1], 0.05,
                           "the spread must be large enough to matter operationally")

    def test_wrong_values_are_never_equal_to_the_truth(self):
        for r in self.rows:
            doc = self.truth[r.doc_id]
            if r.correct:
                self.assertEqual(r.value, doc.truth[r.field])
            else:
                self.assertNotEqual(r.value, doc.truth[r.field])

    def test_invoice_total_errors_are_single_digit_substitutions(self):
        """OCR does not hallucinate a new number; it misreads one glyph.

        Asserted because it is what makes the errors expensive AND plausible:
        3948.00 read as 3948.08 passes every eyeball check a busy operator
        gives it.
        """
        errs = [r for r in self.by_field["invoice_total"] if not r.correct]
        self.assertGreater(len(errs), 50)
        for r in errs:
            truth = self.truth[r.doc_id].truth["invoice_total"]
            self.assertEqual(len(r.value), len(truth))
            diffs = [i for i, (a, b) in enumerate(zip(r.value, truth)) if a != b]
            self.assertEqual(len(diffs), 1)
            self.assertTrue(truth[diffs[0]].isdigit())

    def test_date_errors_are_dominated_by_day_month_transposition(self):
        errs = [r for r in self.by_field["incident_date"] if not r.correct]
        self.assertGreater(len(errs), 50)
        swapped = 0
        for r in errs:
            y, m, d = self.truth[r.doc_id].truth["incident_date"].split("-")
            if r.value == "%s-%s-%s" % (y, d, m):
                swapped += 1
        share = swapped / len(errs)
        # The generator transposes on 70% of date errors; three binomial
        # standard errors either side is the acceptance band.
        se = math.sqrt(0.7 * 0.3 / len(errs))
        self.assertGreater(share, 0.7 - 3 * se)
        self.assertLess(share, 0.7 + 3 * se)

    def test_accuracy_degrades_with_scan_quality(self):
        quality = sorted(d.scan_quality for d in self.docs)
        lo_cut, hi_cut = quality[len(quality) // 4], quality[3 * len(quality) // 4]
        lo_docs = {d.doc_id for d in self.docs if d.scan_quality <= lo_cut}
        hi_docs = {d.doc_id for d in self.docs if d.scan_quality >= hi_cut}
        lo = [r for r in self.rows if r.doc_id in lo_docs]
        hi = [r for r in self.rows if r.doc_id in hi_docs]
        acc_lo = sum(1 for r in lo if r.correct) / len(lo)
        acc_hi = sum(1 for r in hi if r.correct) / len(hi)
        se = math.sqrt(0.25 / len(lo) + 0.25 / len(hi))
        self.assertGreater(acc_hi - acc_lo, 3 * se,
                           "bad scans must extract worse; acc_hi=%.4f acc_lo=%.4f" % (acc_hi, acc_lo))

    def test_confidence_ranks_correctness(self):
        """The score must carry signal, or calibration has nothing to work with
        and routing degenerates to random sampling."""
        for name in FIELD_NAMES:
            rows = self.by_field[name]
            ok = [r.raw_confidence for r in rows if r.correct]
            bad = [r.raw_confidence for r in rows if not r.correct]
            self.assertGreater(len(bad), 20, name)
            mean_ok = statistics.fmean(ok)
            mean_bad = statistics.fmean(bad)
            # Welch standard error on the OBSERVED confidence spread. A
            # Bernoulli 0.25/n bar would be wrong here: confidences are not
            # coin flips, and on invoice_total they are packed into a band a
            # couple of percent wide, where a real gap is still significant.
            se = math.sqrt(statistics.variance(ok) / len(ok)
                           + statistics.variance(bad) / len(bad))
            self.assertGreater(mean_ok - mean_bad, 3 * se,
                               "%s: gap=%.5f se=%.5f" % (name, mean_ok - mean_bad, se))


class TestAbstainPath(unittest.TestCase):

    def test_unknown_document_type_abstains_on_every_field(self):
        docs = generate_documents(400, seed=6, unknown_rate=1.0)
        self.assertTrue(all(d.doc_type == UNKNOWN_DOC_TYPE for d in docs))
        rows = extract_all(SimulatedExtractor(), docs)
        self.assertEqual(len(rows), len(docs) * len(FIELD_NAMES))
        for r in rows:
            self.assertTrue(r.abstained)
            self.assertEqual(r.value, "")
            self.assertEqual(r.raw_confidence, 0.0)

    def test_known_types_never_abstain(self):
        docs = generate_documents(400, seed=6, unknown_rate=0.0)
        self.assertTrue(all(d.doc_type in DOC_TYPES for d in docs))
        self.assertFalse(any(r.abstained for r in extract_all(SimulatedExtractor(), docs)))


class TestInterface(unittest.TestCase):

    def test_backend_is_pluggable(self):
        """Everything above the backend talks to ExtractorBackend, so a real
        OCR stack would be a drop-in. Nothing in calibration or routing reaches
        into SimulatedExtractor."""
        self.assertTrue(issubclass(SimulatedExtractor, ExtractorBackend))
        with self.assertRaises(NotImplementedError):
            ExtractorBackend().extract(generate_documents(1)[0])


class TestSplitting(unittest.TestCase):

    def test_split_is_by_document_and_disjoint(self):
        docs = generate_documents(1000, seed=2)
        fit, ev = split_documents(docs, fit_fraction=0.4)
        self.assertEqual(len(fit) + len(ev), 1000)
        self.assertAlmostEqual(len(fit) / 1000, 0.4, places=6)
        self.assertEqual(set(d.doc_id for d in fit) & set(d.doc_id for d in ev), set())


if __name__ == "__main__":
    unittest.main()

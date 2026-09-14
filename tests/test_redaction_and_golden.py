"""PII redaction against planted PII, and golden-file regression per doc type."""

import hashlib
import unittest

from src.extractor import SimulatedExtractor, extract_all
from src.generator import DOC_TYPES, generate_documents, render_text
from src.redact import luhn_ok, naive_redact, redact


class TestRedaction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.docs = generate_documents(500, seed=31)

    def test_every_planted_pii_string_is_removed(self):
        for doc in self.docs:
            text, planted = render_text(doc)
            clean, found = redact(text)
            for kind, values in planted.items():
                for v in values:
                    self.assertNotIn(v, clean, "%s leaked on %s" % (kind, doc.doc_id))
                    self.assertIn(v, found.get(kind, []), "%s mislabelled" % kind)

    def test_fields_to_extract_survive_redaction(self):
        """Redacting before the model call must not blind the model."""
        for doc in self.docs:
            clean, _ = redact(render_text(doc)[0])
            for name in ("policy_number", "invoice_total", "incident_date", "claimant_name"):
                self.assertIn(doc.truth[name], clean, "%s destroyed on %s" % (name, doc.doc_id))

    def test_naive_digit_run_redaction_gets_it_wrong_both_ways(self):
        """The generic approach: mask long digit runs. It destroys policy
        numbers AND leaks SSNs, phones and emails. Pinned so the precise rules
        cannot be quietly replaced by it."""
        destroyed = leaked = 0
        for doc in self.docs:
            text, planted = render_text(doc)
            naive = naive_redact(text)
            destroyed += doc.truth["policy_number"] not in naive
            leaked += any(v in naive for k in ("ssn", "phone", "email") for v in planted[k])
        self.assertEqual(destroyed, len(self.docs))
        self.assertEqual(leaked, len(self.docs))

    def test_luhn_gate_leaves_non_card_numbers_alone(self):
        self.assertTrue(luhn_ok("4111111111111111"))
        self.assertFalse(luhn_ok("4111111111111112"))
        clean, found = redact("reference 4111111111111112 card 4111111111111111")
        self.assertIn("4111111111111112", clean)
        self.assertEqual(found["card"], ["4111111111111111"])


# Digests of extraction output for a fixed 300-document corpus, per document
# type. Recorded from the implementation once; any change to the generator,
# the corruption model or the confidence model changes them. Update them only
# deliberately, together with the README's measured numbers.
GOLDEN = {
    "repair_invoice": (510, "fa86efaf3c0e7717b9eb6878392f1d7371530d8a961be2ccb3bfa4da2bf6a0ed"),
    "police_report": (535, "ae0ed343f2954d34281d0f17e1e25edc5f59645f61b3ba6b85f10ac2d6318f3d"),
    "medical_summary": (455, "f90ba13ea2a7792c2e1100c5d01784c68da0fc2bcdadddb284dfae5416130e3f"),
}


class TestGoldenExtraction(unittest.TestCase):

    def test_extraction_matches_golden_digest_per_document_type(self):
        docs = generate_documents(300, seed=99)
        for doc_type in DOC_TYPES:
            rows = extract_all(SimulatedExtractor(2024), [d for d in docs if d.doc_type == doc_type])
            blob = repr([(r.doc_id, r.field, r.value, round(r.raw_confidence, 9), r.correct)
                         for r in rows]).encode()
            n, digest = GOLDEN[doc_type]
            self.assertEqual(len(rows), n, doc_type)
            self.assertEqual(hashlib.sha256(blob).hexdigest(), digest, doc_type)


if __name__ == "__main__":
    unittest.main()

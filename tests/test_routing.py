"""Routing correctness: the threshold is derived, and the derivation pays."""

import random
import statistics
import unittest

from src.calibrate import PerFieldCalibrator
from src.extractor import SimulatedExtractor, extract_all, Extraction
from src.fields import (FIELDS, FIELD_BY_NAME, FIELD_NAMES, REVIEW_COST_USD,
                        FieldSpec, cost_optimal_threshold)
from src.generator import generate_documents, split_documents
from src.routing import (evaluate, per_field_thresholds, global_threshold_for_budget,
                         sweep_global_threshold)


def pipeline(n=4000, doc_seed=7, backend_seed=2024, unknown_rate=0.0):
    """Fit calibration on one split, evaluate on the other. Returns
    (eval_rows, calibrated_probs, raw_scores)."""
    docs = generate_documents(n, seed=doc_seed, unknown_rate=unknown_rate)
    fit_docs, eval_docs = split_documents(docs, seed=doc_seed + 1)
    backend = SimulatedExtractor(seed=backend_seed)
    fit_rows = extract_all(backend, fit_docs)
    eval_rows = extract_all(backend, eval_docs)
    cal = PerFieldCalibrator("isotonic").fit(fit_rows)
    return eval_rows, cal.predict(eval_rows), [r.raw_confidence for r in eval_rows]


class TestThresholdDerivation(unittest.TestCase):

    def test_threshold_is_the_cost_ratio_and_nothing_else(self):
        for spec in FIELDS:
            self.assertAlmostEqual(cost_optimal_threshold(spec),
                                   1.0 - REVIEW_COST_USD / spec.escape_cost_usd, places=12)

    def test_thresholds_differ_across_fields_and_order_by_error_cost(self):
        thresholds = per_field_thresholds()
        by_cost = sorted(FIELDS, key=lambda s: s.escape_cost_usd)
        ordered = [thresholds[s.name] for s in by_cost]
        self.assertEqual(ordered, sorted(ordered),
                         "a costlier error must earn a stricter threshold")
        self.assertGreater(max(ordered) - min(ordered), 0.35,
                           "a single global threshold cannot stand in for this spread")
        for t in ordered:
            # Nobody picked these. If one were a round number it would be a
            # coincidence of the cost figures, not a design choice.
            self.assertNotAlmostEqual(t, round(t, 2), places=6)

    def test_review_that_costs_more_than_the_error_is_never_worth_it(self):
        cheap = FieldSpec("trivial", "text", 1.00)
        self.assertEqual(cost_optimal_threshold(cheap, review_cost=2.50), 0.0)

    def test_derived_threshold_minimises_expected_cost(self):
        """The proof that the formula is right, not merely plausible.

        Uses EXPECTED cost against known true probabilities, so the comparison
        is exact and free of sampling noise: any threshold other than
        1 - R/E must cost at least as much, and strictly more once it moves off
        by a grid step.
        """
        rng = random.Random(13)
        spec = FIELD_BY_NAME["policy_number"]
        true_p = [0.5 + 0.499 * rng.random() for _ in range(40000)]
        t_star = cost_optimal_threshold(spec)

        def expected_cost(t):
            total = 0.0
            for p in true_p:
                total += REVIEW_COST_USD if p < t else (1.0 - p) * spec.escape_cost_usd
            return total

        best = expected_cost(t_star)
        grid = [i / 200.0 for i in range(201)]
        for t in grid:
            self.assertLessEqual(best, expected_cost(t) + 1e-9,
                                 "threshold %.3f beat the derived optimum" % t)
        # And the optimum is a genuine minimum, not a flat region.
        self.assertGreater(expected_cost(t_star - 0.05), best * 1.001)
        self.assertGreater(expected_cost(min(t_star + 0.05, 1.0)), best * 1.001)


class TestOperatingPoint(unittest.TestCase):

    def setUp(self):
        self.rows, self.cal, self.raw = pipeline()
        self.result = evaluate("per-field calibrated", self.rows, self.cal,
                               lambda f: per_field_thresholds()[f])

    def test_the_three_numbers_account_for_every_field(self):
        res = self.result
        self.assertEqual(res.n, len(self.rows))
        self.assertAlmostEqual(res.auto_rate + res.review_rate, 1.0, places=12)
        self.assertEqual(res.n_errors_caught + res.n_escaped, res.n_errors)

    def test_review_volume_differs_per_field_as_the_cost_model_demands(self):
        rates = {name: d["reviewed"] / d["n"] for name, d in self.result.per_field.items()}
        self.assertGreater(rates["invoice_total"], rates["damage_description"] * 5,
                           "the expensive field must absorb far more review")

    def test_abstentions_always_route_to_a_human(self):
        rows, cal, _ = pipeline(n=1500, unknown_rate=0.10)
        abstained = [i for i, r in enumerate(rows) if r.abstained]
        self.assertGreater(len(abstained), 0)
        thresholds = per_field_thresholds()
        for i in abstained:
            self.assertEqual(cal[i], 0.0)
            self.assertLess(cal[i], thresholds[rows[i].field],
                            "an unrecognised document type must never auto-approve")


class TestHeadlineGlobalVersusPerField(unittest.TestCase):
    """THE headline comparison, at equal review budget.

    Baseline A: one global threshold applied to RAW confidence -- both generic
    mistakes at once, and it is the common shipped design.
    Baseline B: one global threshold applied to CALIBRATED probability --
    isolates the per-field half of the differentiator.
    Policy C : per-field cost-optimal thresholds on calibrated probability.

    Both baselines get their threshold tuned on the evaluation set itself to
    match C's review volume exactly. That is a hindsight advantage no
    production baseline would have; C wins anyway.
    """

    SEEDS = (7, 17, 27, 37, 47)

    @classmethod
    def setUpClass(cls):
        cls.runs = []
        thresholds = per_field_thresholds()
        for seed in cls.SEEDS:
            rows, cal, raw = pipeline(n=3000, doc_seed=seed, backend_seed=1000 + seed)
            c = evaluate("per-field", rows, cal, lambda f: thresholds[f])
            ta = global_threshold_for_budget(raw, c.n_reviewed)
            a = evaluate("global-raw", rows, raw, lambda f, t=ta: t)
            tb = global_threshold_for_budget(cal, c.n_reviewed)
            b = evaluate("global-cal", rows, cal, lambda f, t=tb: t)
            cls.runs.append((a, b, c))

    def test_per_field_policy_costs_materially_less_at_equal_budget(self):
        ratios = [a.total_cost / c.total_cost for a, _, c in self.runs]
        # Bar from the measured distribution over seeds, not a round number:
        # mean 1.55, sd 0.13, so mean - 3sd is about 1.15. A per-field policy
        # that failed to clear that would not be worth the extra machinery.
        self.assertGreater(min(ratios), 1.15,
                           "measured cost ratios: %s" % [round(r, 3) for r in ratios])
        self.assertGreater(statistics.mean(ratios), 1.4)

    def test_per_field_beats_a_globally_thresholded_calibrated_score_too(self):
        """Calibration alone is not enough. If someone calibrates and then
        still uses one threshold, they leave most of the money on the table."""
        for _, b, c in self.runs:
            self.assertGreater(b.total_cost, c.total_cost)
            self.assertGreaterEqual(b.n_reviewed, c.n_reviewed)

    def test_review_budgets_really_were_held_equal(self):
        for a, _, c in self.runs:
            self.assertLessEqual(abs(a.n_reviewed - c.n_reviewed), 1)

    def test_the_win_is_in_escape_COST_not_escape_COUNT(self):
        """The honest shape of the result, pinned so nobody overclaims.

        Per-field routing lets MORE errors through by count -- deliberately:
        a dropped word in a damage description costs 6 USD and is not worth
        2.50 USD of an operator's attention. It lets far less COST through.
        Any project reporting "errors caught" as the headline is measuring the
        wrong thing, and this test asserts both directions so the trade is
        visible rather than buried.
        """
        for a, _, c in self.runs:
            self.assertLess(c.escape_cost, a.escape_cost / 1.5,
                            "per-field must cut escaped COST sharply")
            self.assertGreater(c.n_escaped, a.n_escaped,
                               "and it does so while letting more cheap errors through")

    def test_per_field_catches_more_errors_where_errors_are_expensive(self):
        expensive = ("invoice_total", "policy_number")
        for a, _, c in self.runs:
            caught_c = sum(c.per_field[f]["errors"] - c.per_field[f]["escaped"] for f in expensive)
            caught_a = sum(a.per_field[f]["errors"] - a.per_field[f]["escaped"] for f in expensive)
            self.assertGreater(caught_c, caught_a)

    def test_even_the_hindsight_optimal_global_threshold_loses(self):
        """Strongest possible baseline: the single threshold chosen by
        exhaustive search to minimise cost on the evaluation set itself."""
        rows, cal, raw = pipeline(n=1500)
        thresholds = per_field_thresholds()
        c = evaluate("per-field", rows, cal, lambda f: thresholds[f])
        grid = [i / 200.0 for i in range(201)]
        _, best_cal = sweep_global_threshold(rows, cal, grid)
        _, best_raw = sweep_global_threshold(rows, raw, grid)
        self.assertLess(c.total_cost, best_cal.total_cost)
        self.assertLess(c.total_cost, best_raw.total_cost)


class TestRoutingArithmetic(unittest.TestCase):

    def test_costs_are_tallied_correctly_on_a_hand_built_case(self):
        spec = FIELD_BY_NAME["invoice_total"]
        rows = [Extraction("D1", "invoice_total", "1", 0.5, True),
                Extraction("D2", "invoice_total", "2", 0.5, False),
                Extraction("D3", "invoice_total", "3", 0.5, False)]
        # Probabilities chosen so D1 auto-approves, D2 auto-approves (escapes),
        # D3 is reviewed (caught).
        res = evaluate("hand", rows, [0.999, 0.999, 0.10], lambda f: 0.5)
        self.assertEqual(res.n_reviewed, 1)
        self.assertEqual(res.n_escaped, 1)
        self.assertEqual(res.n_errors_caught, 1)
        self.assertAlmostEqual(res.review_cost, REVIEW_COST_USD)
        self.assertAlmostEqual(res.escape_cost, spec.escape_cost_usd)
        self.assertAlmostEqual(res.total_cost, REVIEW_COST_USD + spec.escape_cost_usd)
        self.assertAlmostEqual(res.escape_rate, 1.0 / 3.0)

    def test_budget_matching_hits_the_requested_volume(self):
        rng = random.Random(2)
        scores = [rng.random() for _ in range(5000)]
        for target in (0, 1, 137, 2500, 4999, 5000):
            t = global_threshold_for_budget(scores, target)
            self.assertEqual(sum(1 for s in scores if s < t), target)


if __name__ == "__main__":
    unittest.main()

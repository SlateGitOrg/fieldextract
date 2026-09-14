"""Calibration correctness: the maths, the metrics, and the leakage guard."""

import math
import random
import unittest

from src.calibrate import (IsotonicCalibrator, PlattCalibrator, PerFieldCalibrator,
                           GlobalCalibrator, LeakageError, brier_score,
                           expected_calibration_error)
from src.extractor import SimulatedExtractor, extract_all, Extraction
from src.fields import FIELD_NAMES
from src.generator import generate_documents, split_documents


def ece_noise_floor(probs, outcomes, n_bins=10):
    """Expected ECE if the probabilities were PERFECTLY calibrated.

    Needed because a flat "3 standard errors of n/10" bar is wrong here: the
    confidence distribution is heavily concentrated in the top bins, so bin
    occupancies are wildly uneven. With n_b items in a bin at probability p,
    the accuracy estimate has sd sqrt(p(1-p)/n_b) and E|gap| = sqrt(2/pi)*sd
    for a normal. Summing that weighted by occupancy gives the irreducible
    floor any honest calibrator must sit near, and nothing can beat.
    """
    if not probs:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for prob, y in zip(probs, outcomes):
        bins[min(int(prob * n_bins), n_bins - 1)].append(prob)
    total = len(probs)
    floor = 0.0
    for rows in bins:
        if not rows:
            continue
        p_bar = sum(rows) / len(rows)
        sd = math.sqrt(max(p_bar * (1.0 - p_bar), 1e-12) / len(rows))
        floor += (len(rows) / total) * math.sqrt(2.0 / math.pi) * sd
    return floor


def build(n=4000, seed=7, backend_seed=2024):
    docs = generate_documents(n, seed=seed)
    fit_docs, eval_docs = split_documents(docs, seed=seed + 1)
    backend = SimulatedExtractor(seed=backend_seed)
    return extract_all(backend, fit_docs), extract_all(backend, eval_docs)


class TestIsotonicMechanics(unittest.TestCase):

    def test_pav_output_is_monotone(self):
        rng = random.Random(3)
        scores = [rng.random() for _ in range(2000)]
        outcomes = [rng.random() < s for s in scores]
        iso = IsotonicCalibrator().fit(scores, outcomes)
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(iso.y, iso.y[1:])),
                        "pool-adjacent-violators must emit a non-decreasing step function")

    def test_recovers_a_planted_miscalibration_map(self):
        """Plant reported = p ** 0.3 and check isotonic inverts it.

        Tolerance is not chosen by taste: with n items landing in a
        neighbourhood, the empirical rate has standard error sqrt(p(1-p)/m).
        The check below is 4 standard errors of the smallest bucket used.
        """
        rng = random.Random(11)
        n = 40000
        true_p = [0.5 + 0.499 * rng.random() for _ in range(n)]
        reported = [p ** 0.3 for p in true_p]
        outcomes = [rng.random() < p for p in true_p]
        iso = IsotonicCalibrator().fit(reported, outcomes)

        # Compare on a grid, averaging the truth in a window around each probe
        # so we compare like with like.
        worst = 0.0
        m_min = n
        for probe in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
            window = [(r, p) for r, p in zip(reported, true_p) if abs(r - probe) < 0.01]
            if len(window) < 100:
                continue
            m_min = min(m_min, len(window))
            expected = sum(p for _, p in window) / len(window)
            worst = max(worst, abs(iso.predict_one(probe) - expected))
        bar = 4.0 * math.sqrt(0.25 / m_min)
        self.assertLess(worst, bar,
                        "isotonic should recover the planted map to within sampling noise")

    def test_ece_is_near_zero_on_perfectly_calibrated_data(self):
        rng = random.Random(5)
        n = 40000
        probs = [rng.random() for _ in range(n)]
        outcomes = [rng.random() < p for p in probs]
        ece, table = expected_calibration_error(probs, outcomes)
        self.assertLess(ece, 3.0 * ece_noise_floor(probs, outcomes))
        self.assertEqual(sum(row["n"] for row in table), n)

    def test_ece_alone_is_gameable_but_brier_is_not(self):
        """A constant base-rate predictor scores a great ECE and is useless.

        This is why the project reports both. If only ECE were asserted, a
        deliberately broken calibrator that threw away the score entirely would
        pass the suite.
        """
        rng = random.Random(9)
        n = 20000
        probs = [0.5 + 0.5 * rng.random() for _ in range(n)]
        outcomes = [rng.random() < p for p in probs]
        base = sum(1 for o in outcomes if o) / n
        constant = [base] * n
        ece_const, _ = expected_calibration_error(constant, outcomes)
        ece_real, _ = expected_calibration_error(probs, outcomes)
        self.assertLess(ece_const, 0.01, "the useless predictor does score a fine ECE")
        self.assertGreater(brier_score(constant, outcomes), brier_score(probs, outcomes),
                           "Brier must expose the loss of sharpness that ECE hides")
        self.assertLess(ece_real, 0.01)


class TestRawConfidenceIsNotAProbability(unittest.TestCase):

    def setUp(self):
        self.fit_rows, self.eval_rows = build()
        self.y = [r.correct for r in self.eval_rows]
        self.raw = [r.raw_confidence for r in self.eval_rows]

    def test_raw_confidence_is_overconfident_in_every_field(self):
        for name in FIELD_NAMES:
            rows = [r for r in self.eval_rows if r.field == name]
            acc = sum(1 for r in rows if r.correct) / len(rows)
            conf = sum(r.raw_confidence for r in rows) / len(rows)
            self.assertGreater(conf, acc,
                               "%s: mean raw confidence should exceed accuracy" % name)

    def test_isotonic_calibration_materially_reduces_ece(self):
        cal = PerFieldCalibrator("isotonic").fit(self.fit_rows)
        p = cal.predict(self.eval_rows)
        raw_ece, _ = expected_calibration_error(self.raw, self.y)
        cal_ece, _ = expected_calibration_error(p, self.y)
        # Bar from the measured noise floor of this exact binning, not a
        # round number. Raw must sit far above it; calibrated must sit near it.
        floor = ece_noise_floor(self.raw, self.y)
        self.assertGreater(raw_ece, 5 * floor, "raw scores should be visibly miscalibrated")
        self.assertLess(cal_ece, raw_ece / 3.0)
        self.assertLess(cal_ece, 3 * ece_noise_floor(p, self.y),
                        "calibrated ECE should be within sampling noise of perfect")
        self.assertLessEqual(brier_score(p, self.y), brier_score(self.raw, self.y))

    def test_isotonic_and_platt_are_indistinguishable_here(self):
        """A claim that had to be weakened, kept as the honest version.

        The first draft of this project asserted that Platt scaling FAILS on a
        power-law miscalibration while isotonic fixes it, on the reasoning that
        Platt can only apply an affine map in logit space. Measured, that is
        wrong: over the confidence range these scores actually occupy,
        logit(p ** k) is very nearly affine in logit(p), so the one-parameter
        method has plenty of freedom. Both methods cut ECE by roughly 4x and
        the gap between them sits inside the sampling noise floor.

        So the project reports both and defaults to isotonic for a reason that
        is about assumptions, not measured superiority: it does not presume a
        functional form. It does not claim a win it cannot demonstrate.
        """
        iso = PerFieldCalibrator("isotonic").fit(self.fit_rows).predict(self.eval_rows)
        platt = PerFieldCalibrator("platt").fit(self.fit_rows).predict(self.eval_rows)
        ece_iso, _ = expected_calibration_error(iso, self.y)
        ece_platt, _ = expected_calibration_error(platt, self.y)
        ece_raw, _ = expected_calibration_error(self.raw, self.y)
        self.assertLess(ece_iso, ece_raw / 3.0)
        self.assertLess(ece_platt, ece_raw / 3.0)
        floor = max(ece_noise_floor(iso, self.y), ece_noise_floor(platt, self.y))
        self.assertLess(abs(ece_iso - ece_platt), 3 * floor,
                        "iso=%.4f platt=%.4f floor=%.4f: no winner is claimable"
                        % (ece_iso, ece_platt, floor))

    def test_platt_is_still_correct_where_it_is_applicable(self):
        """Sanity check that the Platt implementation itself is not broken:
        on a genuinely logistic miscalibration it recovers the true map."""
        rng = random.Random(21)
        n = 30000
        rows, outcomes = [], []
        for _ in range(n):
            z = rng.gauss(0.0, 2.0)
            p_true = 1.0 / (1.0 + math.exp(-(1.7 * z - 0.4)))
            reported = 1.0 / (1.0 + math.exp(-z))
            rows.append(reported)
            outcomes.append(rng.random() < p_true)
        model = PlattCalibrator().fit(rows, outcomes)
        self.assertAlmostEqual(model.a, 1.7, delta=0.12)
        self.assertAlmostEqual(model.b, -0.4, delta=0.12)


class TestHeldOutEnforcement(unittest.TestCase):

    def test_scoring_the_fitting_split_raises(self):
        fit_rows, eval_rows = build(n=1200)
        cal = PerFieldCalibrator().fit(fit_rows)
        with self.assertRaises(LeakageError):
            cal.predict(fit_rows)
        cal.predict(eval_rows)  # held out: must not raise

    def test_partial_overlap_also_raises(self):
        fit_rows, eval_rows = build(n=1200)
        cal = PerFieldCalibrator().fit(fit_rows)
        contaminated = eval_rows[:500] + fit_rows[:5]
        with self.assertRaises(LeakageError):
            cal.predict(contaminated)

    def test_in_sample_ece_is_optimistic(self):
        """Why the guard exists: fitting and scoring on the same rows reports a
        calibration error that generalisation does not support."""
        fit_rows, eval_rows = build(n=4000)
        cal = PerFieldCalibrator().fit(fit_rows)
        in_sample = cal.predict(fit_rows, check_leakage=False)
        held_out = cal.predict(eval_rows)
        ece_in, _ = expected_calibration_error(in_sample, [r.correct for r in fit_rows])
        ece_out, _ = expected_calibration_error(held_out, [r.correct for r in eval_rows])
        self.assertLess(ece_in, ece_out,
                        "in-sample ECE should look better than the honest held-out number")


class TestPerFieldVersusPooledCalibration(unittest.TestCase):

    def test_pooling_breaks_when_fields_share_a_score_range(self):
        """The mechanism, isolated.

        Two fields emit an identical raw-confidence distribution but have
        different true accuracies. A single pooled monotone map must assign
        them the same probability, so it is wrong for both by construction;
        a per-field map gets both right.
        """
        rng = random.Random(4)
        rows = []
        for i in range(12000):
            conf = 0.80 + 0.19 * rng.random()
            for name, offset in (("invoice_total", 0.10), ("damage_description", -0.10)):
                p = min(max(conf + offset, 0.01), 0.99)
                rows.append(Extraction("D%05d" % i, name, "v", conf, rng.random() < p))
        half = len(rows) // 2
        fit_rows = [r for r in rows if int(r.doc_id[1:]) < 6000]
        eval_rows = [r for r in rows if int(r.doc_id[1:]) >= 6000]
        self.assertGreater(half, 0)

        per_field = PerFieldCalibrator().fit(fit_rows).predict(eval_rows)
        pooled = GlobalCalibrator().fit(fit_rows).predict(eval_rows)

        for name in ("invoice_total", "damage_description"):
            idx = [i for i, r in enumerate(eval_rows) if r.field == name]
            y = [eval_rows[i].correct for i in idx]
            ece_pf, _ = expected_calibration_error([per_field[i] for i in idx], y)
            ece_pool, _ = expected_calibration_error([pooled[i] for i in idx], y)
            # The planted gap between the two fields is 0.20 in probability, so
            # a pooled map that splits the difference must be off by ~0.10.
            self.assertGreater(ece_pool, 0.05, "%s: pooled map must be badly off" % name)
            self.assertLess(ece_pf, ece_pool / 3.0, "%s: per-field map must fix it" % name)

    def test_pooled_calibration_is_not_harmful_on_the_main_simulation(self):
        """Reported as measured, not as hoped.

        On the main simulated population the five fields occupy largely
        different raw-confidence ranges, so a single monotone pooled map is
        already close to field-specific and pooling costs almost nothing. The
        per-field WIN in this project comes from the thresholds, not the
        calibrators -- see tests/test_routing.py. Claiming otherwise would be
        unsupported by the data, so this test pins the honest version.
        """
        fit_rows, eval_rows = build()
        y = [r.correct for r in eval_rows]
        pf = PerFieldCalibrator().fit(fit_rows).predict(eval_rows)
        pooled = GlobalCalibrator().fit(fit_rows).predict(eval_rows)
        mean_pf = mean_pool = 0.0
        for name in FIELD_NAMES:
            idx = [i for i, r in enumerate(eval_rows) if r.field == name]
            yy = [y[i] for i in idx]
            mean_pf += expected_calibration_error([pf[i] for i in idx], yy)[0] / len(FIELD_NAMES)
            mean_pool += expected_calibration_error([pooled[i] for i in idx], yy)[0] / len(FIELD_NAMES)
        self.assertLess(mean_pf, mean_pool)
        self.assertGreater(mean_pf, mean_pool / 2.0,
                           "the pooled calibrator is close behind here; do not overclaim")


if __name__ == "__main__":
    unittest.main()

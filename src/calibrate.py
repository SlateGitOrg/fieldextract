"""Calibration: turning a score into a probability, and proving it worked.

Implemented from scratch in pure stdlib Python -- isotonic regression by the
pool-adjacent-violators algorithm, and Platt scaling by iteratively reweighted
least squares.  No sklearn is available (or needed: both are short).

Two measurements decide whether it worked:
  * Expected Calibration Error -- does a 0.9 score mean 90% correct?
  * Brier score -- is it also SHARP, i.e. did we keep the discrimination?
    ECE alone is gameable: predicting the base rate for every item scores a
    near-perfect ECE and is useless, and its Brier score exposes that.
"""

import math


class LeakageError(RuntimeError):
    """Raised when a calibrator is asked to score data it was fitted on."""


def _clip(p, lo=1e-6, hi=1.0 - 1e-6):
    return min(max(p, lo), hi)


def brier_score(probs, outcomes) -> float:
    n = len(probs)
    if n == 0:
        return 0.0
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(probs, outcomes)) / n


def expected_calibration_error(probs, outcomes, n_bins: int = 10):
    """Equal-width binned ECE, plus the reliability table behind it."""
    if not probs:
        return 0.0, []
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    total = len(probs)
    ece = 0.0
    table = []
    for i, rows in enumerate(bins):
        if not rows:
            continue
        mean_conf = sum(p for p, _ in rows) / len(rows)
        acc = sum(1.0 for _, y in rows if y) / len(rows)
        ece += (len(rows) / total) * abs(mean_conf - acc)
        table.append(dict(lo=i / n_bins, hi=(i + 1) / n_bins, n=len(rows),
                          mean_conf=mean_conf, accuracy=acc, gap=mean_conf - acc))
    return ece, table


def reliability_diagram(table, width: int = 26) -> list:
    """ASCII reliability diagram. Windows cp1252 console: ASCII only."""
    lines = ["  bin         n   conf    acc     gap  confidence(=) vs accuracy(#)"]
    for row in table:
        c = int(round(row["mean_conf"] * width))
        a = int(round(row["accuracy"] * width))
        bar = "".join("#" if i < a else ("=" if i < c else " ") for i in range(width))
        lines.append("  %.1f-%.1f %6d  %.3f  %.3f  %+.3f  |%s|"
                     % (row["lo"], row["hi"], row["n"], row["mean_conf"],
                        row["accuracy"], row["gap"], bar))
    return lines


class IsotonicCalibrator:
    """Monotone non-parametric map score -> probability, fitted by PAV.

    Chosen over Platt as the default because the miscalibration here is a power
    transform, not a logistic shift; Platt can only apply an affine map in
    logit space and so cannot straighten it. Both are implemented and the demo
    reports both, which is the honest way to make that claim.
    """

    def __init__(self):
        self.x = []   # right-hand edge of each pooled block
        self.y = []   # pooled probability of each block

    def fit(self, scores, outcomes):
        pairs = sorted(zip(scores, [1.0 if o else 0.0 for o in outcomes]))
        if not pairs:
            raise ValueError("cannot fit isotonic regression on zero rows")
        # Pool-adjacent-violators: walk left to right, merge any block whose
        # value dips below its left neighbour. Each block keeps (sum, weight)
        # so merging is exact rather than an average-of-averages.
        blocks = []  # [x_right, total, weight]
        for x, y in pairs:
            blocks.append([x, y, 1.0])
            while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] >= blocks[-1][1] / blocks[-1][2]:
                last = blocks.pop()
                blocks[-1][0] = last[0]
                blocks[-1][1] += last[1]
                blocks[-1][2] += last[2]
        self.x = [b[0] for b in blocks]
        self.y = [b[1] / b[2] for b in blocks]
        return self

    def predict_one(self, score: float) -> float:
        if not self.x:
            raise RuntimeError("calibrator not fitted")
        lo, hi = 0, len(self.x) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.x[mid] < score:
                lo = mid + 1
            else:
                hi = mid
        return self.y[lo]

    def predict(self, scores):
        return [self.predict_one(s) for s in scores]


class PlattCalibrator:
    """p = sigmoid(a * logit(score) + b), fitted by IRLS (Newton) on log loss.

    Fitted in LOGIT space rather than on the raw score: a raw score of 0.999 and
    one of 0.9999 are very different events, and a sigmoid linear in the raw
    score cannot separate them.
    """

    def __init__(self, max_iter: int = 60, tol: float = 1e-9):
        self.a = 1.0
        self.b = 0.0
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _logit(p):
        p = _clip(p)
        return math.log(p / (1.0 - p))

    def fit(self, scores, outcomes):
        xs = [self._logit(s) for s in scores]
        ys = [1.0 if o else 0.0 for o in outcomes]
        if not xs:
            raise ValueError("cannot fit Platt scaling on zero rows")
        a, b = 1.0, 0.0
        for _ in range(self.max_iter):
            g0 = g1 = h00 = h01 = h11 = 0.0
            for x, y in zip(xs, ys):
                z = a * x + b
                p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
                r = p - y
                w = max(p * (1.0 - p), 1e-9)
                g0 += r * x
                g1 += r
                h00 += w * x * x
                h01 += w * x
                h11 += w
            # Ridge term keeps the 2x2 Hessian invertible when the scores are
            # nearly constant; without it a degenerate field blows up.
            h00 += 1e-8
            h11 += 1e-8
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-14:
                break
            da = (h11 * g0 - h01 * g1) / det
            db = (h00 * g1 - h01 * g0) / det
            a -= da
            b -= db
            if abs(da) + abs(db) < self.tol:
                break
        self.a, self.b = a, b
        return self

    def predict_one(self, score: float) -> float:
        z = self.a * self._logit(score) + self.b
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    def predict(self, scores):
        return [self.predict_one(s) for s in scores]


class PerFieldCalibrator:
    """One calibrator per field, with a hard guard against fitting on eval data.

    The guard is the point. Calibration fitted on the data it is scored against
    is self-fulfilling -- isotonic regression in particular can drive in-sample
    ECE to near zero while learning nothing generalisable. So the calibrator
    remembers the document ids it saw and REFUSES to score any of them.
    """

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "platt"):
            raise ValueError("method must be isotonic or platt")
        self.method = method
        self.models = {}
        self.fit_doc_ids = frozenset()
        self.fallback = {}

    def _new_model(self):
        return IsotonicCalibrator() if self.method == "isotonic" else PlattCalibrator()

    def fit(self, rows):
        rows = [r for r in rows if not r.abstained]
        if not rows:
            raise ValueError("no non-abstained rows to fit on")
        by_field = {}
        for r in rows:
            by_field.setdefault(r.field, []).append(r)
        for name, frows in by_field.items():
            model = self._new_model()
            model.fit([r.raw_confidence for r in frows], [r.correct for r in frows])
            self.models[name] = model
            self.fallback[name] = sum(1.0 for r in frows if r.correct) / len(frows)
        self.fit_doc_ids = frozenset(r.doc_id for r in rows)
        return self

    def assert_held_out(self, rows):
        overlap = self.fit_doc_ids & frozenset(r.doc_id for r in rows)
        if overlap:
            raise LeakageError(
                "calibrator was fitted on %d of these %d documents; "
                "scoring them reports an in-sample number, not a held-out one"
                % (len(overlap), len({r.doc_id for r in rows})))

    def predict(self, rows, check_leakage: bool = True):
        if check_leakage:
            self.assert_held_out(rows)
        out = []
        for r in rows:
            if r.abstained:
                # An abstention is not a low-confidence guess; it is the
                # absence of a value. Probability of being correct is zero, so
                # it routes to a human under every cost model.
                out.append(0.0)
            elif r.field in self.models:
                out.append(self.models[r.field].predict_one(r.raw_confidence))
            else:
                # Unseen field: fall back to its base rate rather than trusting
                # an uncalibrated raw score.
                out.append(self.fallback.get(r.field, 0.5))
        return out


class GlobalCalibrator(PerFieldCalibrator):
    """One calibrator pooled over ALL fields -- the generic mistake, kept
    runnable so the demo can measure how much it leaves on the table."""

    def __init__(self, method: str = "isotonic"):
        super().__init__(method)
        self._pooled = None

    def fit(self, rows):
        rows = [r for r in rows if not r.abstained]
        if not rows:
            raise ValueError("no non-abstained rows to fit on")
        model = self._new_model()
        model.fit([r.raw_confidence for r in rows], [r.correct for r in rows])
        self._pooled = model
        self.fit_doc_ids = frozenset(r.doc_id for r in rows)
        return self

    def predict(self, rows, check_leakage: bool = True):
        if check_leakage:
            self.assert_held_out(rows)
        return [0.0 if r.abstained else self._pooled.predict_one(r.raw_confidence)
                for r in rows]

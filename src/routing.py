"""Routing: who looks at what, and what it costs.

A routing policy is a map from (field, confidence) to REVIEW or AUTO_APPROVE.
Everything here is scored in dollars, because the only defensible way to
compare two policies is the bill they generate, not their accuracy.
"""

from dataclasses import dataclass, field as dc_field

from .fields import FIELD_BY_NAME, REVIEW_COST_USD, cost_optimal_threshold

REVIEW = "REVIEW"
AUTO = "AUTO_APPROVE"


@dataclass
class PolicyResult:
    name: str
    n: int = 0
    n_reviewed: int = 0
    n_errors: int = 0
    n_errors_caught: int = 0
    n_escaped: int = 0
    review_cost: float = 0.0
    escape_cost: float = 0.0
    per_field: dict = dc_field(default_factory=dict)

    @property
    def review_rate(self) -> float:
        return self.n_reviewed / self.n if self.n else 0.0

    @property
    def auto_rate(self) -> float:
        return 1.0 - self.review_rate

    @property
    def escape_rate(self) -> float:
        """Escaped errors as a fraction of ALL fields processed.

        Reported against all fields, not against auto-approved fields only:
        operations cares how many bad values per thousand reach the claim
        system, and dividing by the auto-approved subset flatters a policy that
        simply reviews more.
        """
        return self.n_escaped / self.n if self.n else 0.0

    @property
    def total_cost(self) -> float:
        return self.review_cost + self.escape_cost

    @property
    def recall_on_errors(self) -> float:
        return self.n_errors_caught / self.n_errors if self.n_errors else 0.0


def evaluate(name, rows, scores, threshold_for_field, review_cost=REVIEW_COST_USD) -> PolicyResult:
    """Score one policy. threshold_for_field(field_name) -> float.

    Route to review when score < threshold. Review is assumed to catch the
    error; that assumption is optimistic for BOTH policies compared here, so it
    does not tilt the comparison, and a review-accuracy term would just scale
    both escape costs by the same constant.
    """
    res = PolicyResult(name=name)
    for row, p in zip(rows, scores):
        spec = FIELD_BY_NAME[row.field]
        pf = res.per_field.setdefault(row.field, dict(
            n=0, reviewed=0, errors=0, escaped=0, review_cost=0.0, escape_cost=0.0,
            threshold=threshold_for_field(row.field)))
        res.n += 1
        pf["n"] += 1
        is_error = row.abstained or not row.correct
        if is_error:
            res.n_errors += 1
            pf["errors"] += 1
        if p < threshold_for_field(row.field):
            res.n_reviewed += 1
            pf["reviewed"] += 1
            res.review_cost += review_cost
            pf["review_cost"] += review_cost
            if is_error:
                res.n_errors_caught += 1
        elif is_error:
            res.n_escaped += 1
            pf["escaped"] += 1
            res.escape_cost += spec.escape_cost_usd
            pf["escape_cost"] += spec.escape_cost_usd
    return res


def per_field_thresholds(review_cost=REVIEW_COST_USD) -> dict:
    """The differentiator, in one line each: threshold = 1 - review/escape."""
    return {name: cost_optimal_threshold(spec, review_cost)
            for name, spec in FIELD_BY_NAME.items()}


def global_threshold_for_budget(scores, n_reviewed: int) -> float:
    """Smallest single threshold that sends about n_reviewed items to review.

    Used to hold the review BUDGET equal between policies. Note this tunes the
    baseline on the very data it is then measured on, which is generous to the
    baseline -- deliberately, so the headline result cannot be explained away
    as an unfairly chosen comparison threshold.
    """
    if n_reviewed <= 0:
        return 0.0
    ordered = sorted(scores)
    if n_reviewed >= len(ordered):
        return ordered[-1] + 1e-9
    # Review-if-score-below means the threshold must sit just above the
    # n_reviewed-th smallest score.
    return ordered[n_reviewed - 1] + 1e-12


def sweep_global_threshold(rows, scores, grid=None, review_cost=REVIEW_COST_USD):
    """Best achievable single global threshold, by exhaustive search on cost.

    The strongest form of the baseline: not a threshold someone picked, but the
    cost-minimising one found with hindsight on the evaluation set itself.
    """
    if grid is None:
        grid = [i / 400.0 for i in range(401)]
    best = None
    for t in grid:
        res = evaluate("global t=%.4f" % t, rows, scores, lambda _f, t=t: t, review_cost)
        if best is None or res.total_cost < best[1].total_cost:
            best = (t, res)
    return best


def min_review_for_target_error(rows, probs, target_escape_rate):
    """Fewest reviews whose EXPECTED escape rate is at most the target.

    With calibrated p, auto-approving an item contributes (1 - p) expected
    errors. To hit a count-based error budget with the fewest reviews you
    review lowest-p first (an exchange argument: swapping any reviewed item
    for a lower-p auto item never increases expected escapes). Stop as soon as
    the expected escapes of what remains fit the budget.

    Returns the set of row indices sent to review. Only valid on calibrated
    probabilities -- on raw overconfident scores the expected-escape estimate
    is itself wrong, which the tests demonstrate.
    """
    n = len(rows)
    order = sorted(range(n), key=lambda i: probs[i])
    budget = target_escape_rate * n
    expected = sum(1.0 - p for p in probs)
    reviewed = set()
    for i in order:
        if expected <= budget:
            break
        expected -= 1.0 - probs[i]
        reviewed.add(i)
    return reviewed


def realised(rows, reviewed):
    """(review_rate, realised escape rate over all fields) from planted truth."""
    n = len(rows)
    escaped = sum(1 for i, r in enumerate(rows)
                  if i not in reviewed and (r.abstained or not r.correct))
    return len(reviewed) / n, escaped / n


def validated_cutoff(rows, probs, target_escape_rate):
    """Probability cutoff chosen on a LABELLED validation split.

    Point-estimate budgeting (min_review_for_target_error) overshoots: the
    highest isotonic blocks are the ones most likely to have been fitted
    optimistically, and they are exactly the ones auto-approved. So instead:
    order validation rows by probability, auto-approve from the top while the
    REALISED escapes (counted against planted/reviewed labels) stay within
    target * n, and return the probability at that point. Cuts only at tie
    boundaries so the cutoff reproduces the same set when re-applied.

    Rule applied downstream: auto-approve iff p >= cutoff (cutoff > 1 means
    review everything).
    """
    n = len(rows)
    order = sorted(range(n), key=lambda i: -probs[i])
    budget = target_escape_rate * n
    errors = 0
    best = 1.0 + 1e-9
    k = 0
    while k < n:
        p = probs[order[k]]
        j = k
        group_err = 0
        while j < n and probs[order[j]] == p:
            r = rows[order[j]]
            group_err += 1 if (r.abstained or not r.correct) else 0
            j += 1
        if errors + group_err > budget:
            break
        errors += group_err
        best = p
        k = j
    return best


def apply_cutoff(probs, cutoff):
    return {i for i, p in enumerate(probs) if p < cutoff}

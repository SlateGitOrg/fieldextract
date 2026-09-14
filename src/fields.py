"""Field catalogue and the cost model that drives every routing decision.

The whole project turns on one idea: a confidence threshold is not a taste
decision, it is the solution of a two-cost comparison.  This module holds the
two costs and derives the threshold from them, so that nobody downstream is
tempted to type 0.9.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    """One extractable field, with the economics attached to it.

    escape_cost_usd is the expected cost of ONE wrong value reaching the claims
    system undetected.  These are deliberately spread over almost two orders of
    magnitude: a wrong invoice total mis-pays a claim, a wrong word in a
    free-text damage description costs an adjuster a few seconds of confusion.
    Any design that routes both at the same threshold is asserting these
    numbers are equal, which they are not.
    """

    name: str
    kind: str  # money | id | date | name | text
    escape_cost_usd: float


# Cost of having an operator re-key and confirm one field: ~2 minutes of a
# loaded USD 75/hr claims-intake operator.  This is the ONLY knob that is a
# policy choice; the thresholds below are consequences of it.
REVIEW_COST_USD = 2.50

FIELDS = (
    FieldSpec("invoice_total", "money", 220.00),
    FieldSpec("policy_number", "id", 60.00),
    FieldSpec("incident_date", "date", 45.00),
    FieldSpec("claimant_name", "name", 18.00),
    FieldSpec("damage_description", "text", 6.00),
)

FIELD_BY_NAME = {f.name: f for f in FIELDS}
FIELD_NAMES = tuple(f.name for f in FIELDS)


def cost_optimal_threshold(spec: FieldSpec, review_cost: float = REVIEW_COST_USD) -> float:
    """Calibrated-probability threshold below which review is the cheaper action.

    Let p be the CALIBRATED probability that the extracted value is correct.
    Auto-approving costs (1 - p) * escape_cost in expectation.  Reviewing costs
    review_cost and (by assumption) catches the error.  Review is cheaper when

        (1 - p) * escape_cost > review_cost
        p < 1 - review_cost / escape_cost

    so the threshold is 1 - review_cost/escape_cost.  It is a ratio of two
    business numbers, which is why it never lands on a round value, and why a
    field with a 6 USD error cost gets a far laxer bar than one with a 220 USD
    error cost.

    If review costs at least as much as the error it prevents, the threshold is
    0: never review, because review can only lose money.
    """
    if review_cost >= spec.escape_cost_usd:
        return 0.0
    return 1.0 - review_cost / spec.escape_cost_usd

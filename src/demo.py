"""The 60-second artefact.

Runs the whole pipeline end to end and prints the three things operations
actually needs: is the confidence score a probability, what does each field
cost, and where does the human belong.

ASCII only -- the target console is Windows cp1252.
"""

from .calibrate import (PerFieldCalibrator, brier_score, expected_calibration_error,
                        reliability_diagram)
from .extractor import SimulatedExtractor, extract_all
from .fields import FIELD_BY_NAME, FIELD_NAMES, REVIEW_COST_USD
from .generator import generate_documents, render_text, split_documents
from .redact import naive_redact, redact
from .routing import (apply_cutoff, evaluate, global_threshold_for_budget,
                      min_review_for_target_error, per_field_thresholds, realised,
                      validated_cutoff)

N_DOCS = 4000
UNKNOWN_RATE = 0.02  # a slice of document types the pipeline has never seen


def rule(title=""):
    if title:
        print("\n" + title)
        print("-" * len(title))
    else:
        print("-" * 78)


def money(x):
    return "{:>10,.0f}".format(x).replace(",", ",")


def main():
    print("=" * 78)
    print("fieldextract -- per-field calibrated confidence, routed at a cost-optimal")
    print("               threshold. Insurance claims document intake.")
    print("=" * 78)
    print("NOTE: there is no OCR engine and no vision-language model in this")
    print("environment. The extraction backend is a DETERMINISTIC SIMULATOR behind")
    print("the ExtractorBackend interface, seeded and reproducible, with realistic")
    print("error characteristics (OCR digit substitution, day/month transposition,")
    print("overconfident raw scores). The calibration and routing layers above it")
    print("are real and are what the test suite proves.")

    docs = generate_documents(N_DOCS, seed=7, unknown_rate=UNKNOWN_RATE)
    fit_docs, eval_docs = split_documents(docs, fit_fraction=0.4, seed=8)
    backend = SimulatedExtractor(seed=2024)
    fit_rows = extract_all(backend, fit_docs)
    eval_rows = extract_all(backend, eval_docs)

    rule("1. Corpus")
    print("  documents            : %d (%d fit / %d held-out eval)"
          % (len(docs), len(fit_docs), len(eval_docs)))
    print("  field extractions    : %d fit / %d eval" % (len(fit_rows), len(eval_rows)))
    print("  unrecognised type    : %d eval documents took the abstain path"
          % sum(1 for d in eval_docs if d.doc_type not in
                ("repair_invoice", "police_report", "medical_summary")))
    scored = [r for r in eval_rows if not r.abstained]
    truth = [r.correct for r in scored]
    raw = [r.raw_confidence for r in scored]
    print("  overall accuracy     : %.4f  (mean raw confidence %.4f)"
          % (sum(truth) / len(truth), sum(raw) / len(raw)))

    rule("2. What does a confidence of 0.9 mean? (raw scores, held-out)")
    raw_ece, raw_table = expected_calibration_error(raw, truth)
    for line in reliability_diagram(raw_table):
        print(line)
    print("  ECE %.4f   Brier %.4f" % (raw_ece, brier_score(raw, truth)))
    top = [(c, t) for c, t in zip(raw, truth) if c >= 0.99]
    print("  Fields the extractor scored 0.99 or above are correct %.1f%% of the"
          % (100.0 * sum(1 for _, t in top if t) / len(top)))
    print("  time (n=%d). The score is a ranking, not a probability." % len(top))

    calibrator = PerFieldCalibrator("isotonic").fit(fit_rows)
    cal = calibrator.predict(eval_rows)          # raises if the split leaked
    cal_scored = [p for p, r in zip(cal, eval_rows) if not r.abstained]

    rule("3. After per-field isotonic calibration (fitted on the fit split only)")
    cal_ece, cal_table = expected_calibration_error(cal_scored, truth)
    for line in reliability_diagram(cal_table):
        print(line)
    print("  ECE %.4f   Brier %.4f" % (cal_ece, brier_score(cal_scored, truth)))
    print("  ECE reduced %.1fx." % (raw_ece / cal_ece))

    rule("4. Per-field error profiles -- the reason one threshold cannot serve")
    print("  field                 acc    raw ECE   cal ECE   raw Brier  cal Brier")
    for name in FIELD_NAMES:
        idx = [i for i, r in enumerate(eval_rows) if r.field == name and not r.abstained]
        y = [eval_rows[i].correct for i in idx]
        rr = [eval_rows[i].raw_confidence for i in idx]
        cc = [cal[i] for i in idx]
        print("  %-20s %.3f    %.4f    %.4f     %.4f     %.4f"
              % (name, sum(y) / len(y), expected_calibration_error(rr, y)[0],
                 expected_calibration_error(cc, y)[0], brier_score(rr, y),
                 brier_score(cc, y)))

    rule("5. The cost model, and the thresholds it forces")
    print("  review cost per field : USD %.2f" % REVIEW_COST_USD)
    print("  threshold = 1 - review_cost / escape_cost. Nobody picks these.")
    print()
    print("  field                 escape cost   threshold")
    thresholds = per_field_thresholds()
    for name in FIELD_NAMES:
        print("  %-20s   USD %6.2f     %.5f"
              % (name, FIELD_BY_NAME[name].escape_cost_usd, thresholds[name]))

    result = evaluate("per-field calibrated", eval_rows, cal, lambda f: thresholds[f])

    rule("6. Operating point (held-out eval set)")
    print("  field                  auto%   review%   escaped   escaped cost")
    for name in FIELD_NAMES:
        d = result.per_field[name]
        print("  %-20s  %5.1f     %5.1f   %7d   USD %8.0f"
              % (name, 100.0 * (1 - d["reviewed"] / d["n"]),
                 100.0 * d["reviewed"] / d["n"], d["escaped"], d["escape_cost"]))
    print("  %-20s  %5.1f     %5.1f   %7d   USD %8.0f"
          % ("ALL", 100.0 * result.auto_rate, 100.0 * result.review_rate,
             result.n_escaped, result.escape_cost))
    print()
    print("  auto-approved  : %.1f%% of fields" % (100.0 * result.auto_rate))
    print("  sent to review : %.1f%% of fields" % (100.0 * result.review_rate))
    print("  escaped        : %.2f%% of all fields carried a wrong value into the"
          % (100.0 * result.escape_rate))
    print("                   claim system, costing USD %.0f" % result.escape_cost)

    rule("7. HEADLINE -- one global threshold vs per-field, at EQUAL review budget")
    t_raw = global_threshold_for_budget([r.raw_confidence for r in eval_rows],
                                        result.n_reviewed)
    t_cal = global_threshold_for_budget(cal, result.n_reviewed)
    a = evaluate("A", eval_rows, [r.raw_confidence for r in eval_rows], lambda f: t_raw)
    b = evaluate("B", eval_rows, cal, lambda f: t_cal)
    print("  Both baselines get their single threshold tuned on the EVALUATION set")
    print("  to match the per-field review volume exactly -- a hindsight advantage")
    print("  no production baseline would have.")
    print()
    print("  policy                              review%  escape%  escape USD  total USD")
    for label, res in (("A one global threshold on RAW conf", a),
                       ("B one global threshold on CALIB.  ", b),
                       ("C per-field cost-optimal (ours)   ", result)):
        print("  %s  %6.1f   %6.2f  %s %s"
              % (label, 100.0 * res.review_rate, 100.0 * res.escape_rate,
                 money(res.escape_cost), money(res.total_cost)))
    print()
    print("  C costs %.2fx less than A and %.2fx less than B for the same number"
          % (a.total_cost / result.total_cost, b.total_cost / result.total_cost))
    print("  of human reviews (%d)." % result.n_reviewed)
    print()
    print("  HONEST CAVEAT: C lets MORE errors through by COUNT (%d vs %d) and less"
          % (result.n_escaped, a.n_escaped))
    print("  through by COST (USD %.0f vs %.0f). That is the design, not a defect:"
          % (result.escape_cost, a.escape_cost))
    print("  a dropped word in a damage description costs USD %.2f and is not worth"
          % FIELD_BY_NAME["damage_description"].escape_cost_usd)
    print("  USD %.2f of an operator's attention. Counting errors is the wrong" % REVIEW_COST_USD)
    print("  metric; that is the whole argument.")
    expensive = ("invoice_total", "policy_number")
    caught_c = sum(result.per_field[f]["errors"] - result.per_field[f]["escaped"] for f in expensive)
    caught_a = sum(a.per_field[f]["errors"] - a.per_field[f]["escaped"] for f in expensive)
    print("  On the two costliest fields C catches %d errors to A's %d."
          % (caught_c, caught_a))

    rule("8. Target error rate at minimum review (fit / validation / test)")
    print("  Cutoff chosen on a labelled validation split, then applied blind to")
    print("  a held-out test split. Point-estimate budgeting on raw scores shown")
    print("  as the generic mistake.")
    print()
    print("  target   auto%(test)  escape%(test)   raw-as-probability escape%")
    fit3, rest = split_documents(docs, fit_fraction=1 / 3, seed=8)
    val3, test3 = split_documents(rest, fit_fraction=0.5, seed=9)
    fr3, vr3, tr3 = (extract_all(backend, fit3), extract_all(backend, val3),
                     extract_all(backend, test3))
    cal3 = PerFieldCalibrator("isotonic").fit(fr3)
    pv3, pt3 = cal3.predict(vr3), cal3.predict(tr3)
    raw3 = [r.raw_confidence for r in tr3]
    for target in (0.003, 0.01, 0.02):
        rev_rate, esc = realised(tr3, apply_cutoff(pt3, validated_cutoff(vr3, pv3, target)))
        naive_esc = realised(tr3, min_review_for_target_error(tr3, raw3, target))[1]
        print("  %5.1f%%     %5.1f          %5.2f            %5.2f"
              % (100 * target, 100 * (1 - rev_rate), 100 * esc, 100 * naive_esc))

    rule("9. PII redaction before the model call (planted PII)")
    leaks = destroyed = naive_destroyed = 0
    sample = eval_docs[:500]
    for d in sample:
        text, planted = render_text(d)
        clean, _ = redact(text)
        leaks += sum(1 for vs in planted.values() for v in vs if v in clean)
        destroyed += d.truth["policy_number"] not in clean
        naive_destroyed += d.truth["policy_number"] not in naive_redact(text)
    print("  documents: %d   PII strings leaked: %d   policy numbers destroyed: %d"
          % (len(sample), leaks, destroyed))
    print("  naive digit-run redaction destroys %d/%d policy numbers."
          % (naive_destroyed, len(sample)))
    print("=" * 78)


if __name__ == "__main__":
    main()

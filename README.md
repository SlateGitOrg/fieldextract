# fieldextract

> Document extraction with per-field calibrated confidence routed at a cost-optimal threshold, so operations knows where to put the human.

> **Implementation note.** No OCR engine, vision-language model, Ollama or paid
> API is used. Extraction runs behind the `ExtractorBackend` interface
> (`src/extractor.py`), and the shipped backend is a **deterministic, seeded
> simulator**. It has planted ground truth, field-specific error profiles (OCR
> digit swaps, day/month transposition, dropped tokens), worse accuracy on
> poor scans, and deliberately overconfident raw scores. Calibration, cost
> routing, target-error routing, PII redaction and the abstain path are real,
> and the tests check them. A real PaddleOCR + Qwen-VL stack plugs in by
> implementing `extract(document) -> [Extraction(doc_id, field, value,
> raw_confidence, ...)]`. Nothing above that interface changes. PostgreSQL,
> FastAPI, the React review UI and the public datasets (FUNSD/CORD/DocILE)
> remain targets. Every number below is measured on the simulated backend, not
> on a real model.


`FLAGSHIP` · **AI / ML Engineering** · Advanced · ~5 weeks · Insurance - claims document intake

**Primary language:** Python
**Tags:** `document-ai`, `calibration`, `ocr`, `human-in-the-loop`, `local-llm`, `fastapi`

---

## The problem

An insurer receives repair invoices, police reports and medical summaries as scanned PDFs. Full manual keying costs a fortune; full automation puts wrong numbers into claim payments. The real question is not how accurate the model is - it is which fields can be trusted and where exactly the human belongs, and a single accuracy figure cannot answer that.

## ⭐ The differentiator

Produces **per-field calibrated confidence and routes to human review at a cost-optimal threshold derived from the actual cost of an error against the cost of a review** - with calibration *verified* by reliability diagrams and expected calibration error, not assumed. A generic extraction project reports 94% field accuracy and gives operations no basis for deciding what to check, so operations checks everything and the automation saves nothing.

This is the sentence to lead with when someone asks you to walk through the
project. Everything else in this repo exists to make it true and to prove it.

## Data

Public document datasets - **FUNSD, CORD, DocILE** (all openly licensed) - plus a documented synthetic generator producing insurance-shaped documents with realistic scan noise and known field ground truth.

> No paid API key is required to run or demo this project. Where a paid
> service would add value it is wired as an optional enhancement behind an
> interface with an offline mock as the default implementation.

## Stack

- Python
- PaddleOCR / Tesseract for the OCR layer
- A local open-weight vision-language model via Ollama (LLaVA / Qwen-VL) - no paid key required; a hosted model is an optional comparison
- PostgreSQL, FastAPI
- React review UI
- Docker, CI, pytest

## Core capabilities

- Layout-aware extraction with per-field bounding-box provenance, so every value points at where it came from
- Per-field confidence calibration (temperature and isotonic) with reliability reporting
- Cost-optimal routing threshold derived from a configurable error/review cost matrix
- Human review UI showing the source region for every low-confidence field
- Review outcomes fed back as labelled data, with a drift monitor on the input distribution

## Repository layout

```
src/extract/
src/calibrate/
src/route/
service/
apps/review/
test/calibration/
```

## Build plan

1. OCR and extraction with provenance first. Provenance is what makes review fast enough to be worth doing.
2. Measure raw confidence calibration. It will be bad. That is the finding.
3. Calibrate, then derive the routing threshold from the cost matrix rather than picking 0.9 because it looks confident.
4. Review UI and the feedback loop last.

## Testing strategy

Assert **expected calibration error below a stated bound** on held-out data - a model whose 0.9 confidence is right 62% of the time makes the routing threshold meaningless. Assert the routing policy achieves the target error rate at minimum review volume. Golden-file extraction tests per document type guard against regression.

Tests assert **correctness**, not merely that the code runs. A green suite on
this repo is a claim about behaviour under adversarial conditions; treat any
test that would pass against a deliberately broken implementation as a bug in
the test.

## Quality & safety layer

PII is redacted before any model call. An unrecognised document type takes an explicit abstain path rather than producing a best guess, because a confidently wrong figure on a claim payment is the expensive failure.

## Measurable outcome

> On the simulated backend, per-field cost-optimal routing costs 1.78x less than one tuned global threshold for the same 4,817 human reviews. A validated cutoff meeting a 1% field-error target auto-approves 43.8% of fields at a realised 0.87% error rate on held-out test data.

(The original target read "68% auto-approved at 0.3% error". That is not what we measured: at a 0.3% target the simulator auto-approves 15.1%. 65.6% auto-approval corresponds to a 2% target.)

State it in these terms — business units, not technical ones — in your CV
bullet and in the first thirty seconds of describing the project.

## Measured results

All figures come from `python -m src.demo` (4,000 synthetic documents, seeded)
on the **simulated backend**. They show the engineering works; they are not
model benchmarks.

| Measure | Value |
|---|---|
| Raw-confidence ECE (held-out) | 0.0290 |
| ECE after per-field isotonic calibration (fitted on the fit split only) | 0.0048 (6.0x lower) |
| Cost-derived thresholds (review USD 2.50) | invoice_total 0.98864, policy_number 0.95833, incident_date 0.94444, claimant_name 0.86111, damage_description 0.58333 |
| Total cost at equal review budget (4,817 reviews): global raw / global calibrated / per-field | USD 32,896 / 28,762 / 18,518 |
| Errors caught on the two costliest fields: global raw vs per-field | 112 vs 207 |
| Validated-cutoff routing, target 0.3% / 1% / 2%: auto-approved (test) | 15.1% / 43.8% / 65.6% |
| ... realised escape rate (test) | 0.17% / 0.87% / 2.15% |
| ... the same targets with raw scores treated as probabilities | 0.84% / 2.90% / 4.47% |
| Planted PII strings leaked after redaction (500 docs) | 0 (naive digit-run rule destroys 500/500 policy numbers) |

What the tests caught along the way:
- The original "68% auto-approved at 0.3% error" outcome is false on this backend and has been replaced.
- Budgeting to a target error rate using calibrated *point estimates* still overshot: 0.3% requested gave about 0.48% realised, because the top isotonic blocks fit optimistically. The target is now enforced with a cutoff chosen on a labelled validation split.
- An early draft claimed Platt scaling fails where isotonic works. Measured, the two are indistinguishable here (`test_isotonic_and_platt_are_indistinguishable_here`).

### Limitations
- The simulator decides both the error process and how miscalibrated the scores are. How well calibration and routing transfer to a real OCR/VLM model is untested.
- Per-field routing lets *more* errors through by count (450 vs 213) and fewer by cost. The trade depends on the escape-cost table in `src/fields.py`, which is illustrative, not actuarial.
- Review is assumed to catch every error it sees.
- At a 0.3% target the test split has only a few dozen errors, so the realised rate is noisy from seed to seed (0.19-0.50% across 8 seeds in development).
- A validated cutoff only needs a good ranking, so for a count-based target raw scores with a validated cutoff do about as well as calibrated ones. Calibration matters for cost-based thresholds and for reading the score as a probability.
- There is no drift monitor, feedback loop, review UI, service layer or database yet. Redaction is pattern-based and covers SSN, phone, email and Luhn-valid card numbers only.

## Interview questions this project answers

- **What does a confidence score of 0.9 mean, and how do you know?**
- **How did you choose the review threshold?**
- **What does your system do with a document type it has never seen?**

## What this deliberately is *not*

- Not a chat-with-your-PDF demo. The output is structured fields with calibrated confidence and a routing decision.
- Not dependent on any paid API - the default path runs locally.


## Run it now

```bash
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Requires Python 3.11+. The runnable core uses **only the standard
library** (including `sqlite3`), so there is nothing to install.

## Getting started

```bash
cd fieldextract
python -m unittest discover -s tests -v
python -m src.demo
```

Nothing to install. The Docker/Ollama/PostgreSQL path described under Stack is
the target, not yet implemented.

## Definition of done

- [ ] The differentiator above is implemented, and a test proves it
- [ ] The measurable outcome is produced by a command anyone can run
- [ ] `README` explains the one decision a generic version gets wrong
- [ ] CI runs the full suite on every push and is green on `main`
- [ ] A recruiter can see the headline artefact in under 60 seconds

## Licence

MIT — see [LICENSE](LICENSE).

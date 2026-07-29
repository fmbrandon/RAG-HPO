# Phase 3 findings: precision cascade and recall-preserving verification

## Decision

The deterministic lexical cascade is **not** suitable as the final annotator.
It is a high-confidence evidence lane only. On all 112 eligible CSC cases it
reached 0.816 precision but only 0.208 recall. The same fixed policy reached
0.923 precision and 0.346 recall on 114 GSC cases. This supports its
generalizability as a precision filter, not as a complete phenotype extractor.

The retained Phase 3 policy starts with the rebuilt RAG-HPO predictions:

1. model/FastHPOCR agreement is accepted without another model call;
2. predictions not found verbatim in the note are retained, not rejected;
3. local negation, family-history, and uncertainty rules flag a result but do
   not delete it;
4. remaining predictions are reviewed together in one model call per case;
5. only an `unsupported` verdict with `high` confidence removes a prediction;
6. every ambiguous or lower-confidence verdict remains in the output.

This policy is provider-neutral and compatible with a loopback
OpenAI-compatible local model. It does not contain case IDs, benchmark phrases,
or gold-standard answers.

## Why the local context filter was changed

The first staged run allowed deterministic context rules to delete findings.
Discovery already showed that this was harmful: those rules removed four true
positives and three false positives. The locked confirmation audit showed the
same problem more clearly: they removed 15 true positives and eight false
positives.

The corrected policy therefore treats local assertion output as review
evidence. It does not make a final deletion. This is a general safety repair,
not a case-specific threshold.

## Accuracy

The confirmation cohort was fixed before live inference and contains the 82
CSC cases not used for discovery.

| Cohort and policy | TP | FP | FN | Precision | Recall | F1 | F0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Discovery base RAG-HPO | 299 | 124 | 205 | 0.707 | 0.593 | 0.645 | 0.681 |
| Discovery corrected hybrid | 296 | 103 | 208 | 0.742 | 0.587 | 0.656 | 0.705 |
| Confirmation base RAG-HPO | 789 | 332 | 496 | 0.704 | 0.614 | 0.656 | 0.684 |
| Confirmation corrected hybrid | 774 | 260 | 511 | 0.749 | 0.602 | 0.668 | 0.714 |
| All 112 CSC, descriptive base | 1,088 | 456 | 701 | 0.705 | 0.608 | 0.653 | 0.683 |
| All 112 CSC, descriptive hybrid | 1,070 | 363 | 719 | 0.747 | 0.598 | 0.664 | 0.711 |

On confirmation, the verifier removed 72 false positives and 15 true
positives. The mean paired per-case F0.5 improvement was 0.0348, with a 95%
bootstrap interval from 0.0220 to 0.0492 and a one-sided sign-flip
`p = 0.00001`. Mean paired per-case F1 also improved by 0.0129, with a 95%
interval from 0.0045 to 0.0216.

Recall passed the strengthened gate: it remained above 0.58 and declined by
less than 0.02 from the base model. The hybrid did **not** pass the predeclared
0.80 precision gate. It must therefore remain benchmark-only or explicitly
experimental; it is not promoted to the default CLI.

## Cost and privacy

The confirmation verifier reviewed 543 items in 82 batched calls. It added
approximately 70,478 input tokens: median 715 and p95 1,737 per case. This is
about 12% of the current pipeline's conservative input-token lower bound,
because the current mapper repeats the full note for each extracted phenotype.
Median added runtime was represented by an aggregate 2.66 seconds per case on
the validated Mac/Groq run. This is not an orders-of-magnitude increase and is
well below manual review time.

Only the extracted phrase, containing sentence, proposed HPO concept,
definition, parents, and score were submitted for verification. The API key,
raw provider responses, and complete notes were not written to tracked files.

## Remaining limitations

- The live hybrid was tested only on authorized CSC text. GSC independently
  validates the offline confidence lane, not the model verifier.
- The current verifier is an add-on to stored RAG-HPO predictions. Phase 4 must
  integrate it into the pipeline and measure end-to-end tokens and latency.
- Groq temperature zero is not a bitwise reproducibility guarantee.
- Exact-set scoring still treats clinically plausible ancestor/descendant
  mappings as wrong. Human adjudication remains necessary after machine-side
  normalization and hierarchy reports.
- Precision remains below the 0.80 promotion threshold. Raising it by deleting
  more predictions would conflict with the recall constraint and is not
  justified by this evidence.

Detailed decision rows and provider-derived outputs remain private under
`RAG-HPO_AUDIT_ARTIFACTS/phase3-hybrid/`.

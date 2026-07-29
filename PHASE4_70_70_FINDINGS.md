# Staged 70/70 remediation findings

## Outcome

The staged `balanced` pipeline is functional and substantially more
precision-oriented, but it is not promoted to the default.

- CSC passed the predeclared point-estimate target: precision 0.727, recall
  0.711, F1 0.719.
- GSC precision passed at 0.858, but recall was 0.698. This is one additional
  true positive short of the 0.70 gate.
- Counting all mapped review findings would raise GSC recall to 0.705, but
  review findings are not accepted answers and cannot be used to claim a pass.
- The case-bootstrap intervals cross 0.70, so the samples do not prove that
  corpus-wide precision and recall exceed 0.70.

No setting was changed after looking at CSC confirmation or GSC results.

## Evaluation boundary

Full-corpus live runs were stopped. Future provider evaluation uses fixed,
performance-blind subsets to save time and tokens:

- CSC: 30 cases sampled only from the 82 untouched confirmation cases,
  containing 446 manual-reference findings.
- GSC: 30 cases sampled from 114 eligible cases, containing 268
  manual-reference findings.
- Seed: `20260728`.
- Strata: note character count crossed with manual-reference count in a 3x3
  grid.

The selected samples closely resemble their eligible populations in average
note length and annotation count. Selection manifests contain IDs, strata,
seed, and hashes but no note text. The private CSV files are owner-only and
outside Git.

These are subset estimates, not complete-corpus results. Case-level
nonparametric bootstrap intervals quantify sampling uncertainty but do not
cover corpus shift, model-run variability, or gold-standard error.

## Strict accepted-only results

| Corpus | TP | FP | FN | Precision | Recall | F1 | 95% precision interval | 95% recall interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CSC | 317 | 119 | 129 | 0.727 | 0.711 | 0.719 | 0.674–0.783 | 0.648–0.770 |
| GSC | 187 | 31 | 81 | 0.858 | 0.698 | 0.770 | 0.804–0.904 | 0.628–0.762 |

The all-mapped sensitivity results were:

| Corpus | Precision | Recall | F1 |
|---|---:|---:|---:|
| CSC | 0.724 | 0.722 | 0.723 |
| GSC | 0.851 | 0.705 | 0.771 |

This shows that the review queue contains enough signal to cross the GSC point
threshold, but accepting it wholesale is not justified.

## Comparison with the historical system

On the same selected cases, historical LLaMa 4-Scout workbook counts were:

| Corpus | System | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| CSC | Historical | 0.695 | 0.741 | 0.717 |
| CSC | Rebuilt accepted | 0.727 | 0.711 | 0.719 |
| GSC | Historical | 0.623 | 0.816 | 0.707 |
| GSC | Rebuilt accepted | 0.858 | 0.698 | 0.770 |

The rebuild therefore moved in the requested precision-first direction. CSC
F1 was essentially unchanged, while GSC F1 increased. Recall remains the
limiting metric. Historical predicted IDs are unavailable, so disagreement
cannot be compared ID by ID.

## Reliability, time, and tokens

- Both 30-case subsets ultimately completed without row errors.
- GSC completed in 569 seconds using 134 provider requests and 856,905 total
  tokens.
- Four CSC rows encountered transient provider network failures. SQLite resume
  retried only those rows and recovered all four.
- The old resume manifest behavior overwrote earlier attempt cost totals, so
  the exact cumulative CSC token/runtime total cannot be reconstructed.
- Future run manifests now retain every attempt and cumulative provider usage
  and elapsed time. This accounting repair does not change predictions.

Using 30 rather than all eligible cases reduced live evaluation volume by
63% for the untouched CSC pool and 74% for GSC.

## Decision

`balanced` remains experimental because the independent GSC accepted-only
recall gate failed, even though only narrowly. `model` remains the documented
default. The completed machine work leaves two distinct next questions:

1. Whether a general, predeclared verification policy can safely promote a
   small part of the review queue without losing the precision advantage.
2. Whether remaining exact-ID disagreements are clinically acceptable
   specificity differences or gold-standard omissions. That requires blinded
   human review, not case-specific production rules.

The machine-readable aggregate evidence is in
`benchmarks/results/staged-70-70-subset-summary.json`. Raw notes, predictions,
provider-derived rows, and detailed score files remain in the private audit
artifact directory.

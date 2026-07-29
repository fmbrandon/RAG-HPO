# RAG-HPO Accuracy Investigation

## Bottom line

The rebuilt RAG-HPO is more reliable, private, reproducible, and easier to run,
but it does **not** currently have better exact-set accuracy than the historical
LLaMa 4-Scout result.

On the 82 untouched confirmation cases:

| Measure | Rebuild | Historical | Difference |
| --- | ---: | ---: | ---: |
| TP / FP / FN | 786 / 335 / 499 | 926 / 429 / 364 | −140 / −94 / +135 |
| Micro precision | 0.7012 | 0.6834 | +0.0178 |
| Micro recall | 0.6117 | 0.7178 | −0.1062 |
| Micro F1 | 0.6534 | 0.7002 | −0.0468 |
| Macro F1 | 0.6469 | 0.6938 | −0.0469 |

The rebuild emitted 234 fewer distinct positive predictions. It avoided 94
historical false positives but also lost 140 historical true positives. That
more conservative output directly explains why the aggregate precision is a
little higher while recall and F1 are lower.

The paired per-case precision difference was not established: mean `+0.0028`,
95% interval `[-0.0338, 0.0400]`. Recall was lower by `−0.1005`, interval
`[-0.1396, −0.0625]`, Holm-adjusted two-sided p `< 0.0001`. F1 was lower by
`−0.0469`, interval `[-0.0827, −0.0116]`, adjusted p `0.0230`.

The complete statistical record is in
[`csc-confirmation-82-analysis.json`](benchmarks/results/csc-confirmation-82-analysis.json).

## Study design

- The existing fixed 30-case sample was used only for discovery.
- Every remaining eligible case—82 cases—was locked before further inference.
- A 20-case subset was fixed in advance and observed three times for
  run-to-run variability.
- The current HPO release, two historical releases, the embedding model,
  prompt files, references, input, predictions, and vectors are identified by
  SHA-256.
- Exact normalized HPO sets remain the primary metric. Hierarchy-aware and
  compound-reference scores are explicitly labeled sensitivity analyses.
- Raw notes, provider responses, and review material remain outside Git.

The cohort definition is in
[`csc-confirmation-82-20260728-selection.json`](benchmarks/results/csc-confirmation-82-20260728-selection.json).

## Where recall is lost

The automated stage ledger assigns every one of the 499 confirmation false
negatives a primary, mutually exclusive stage:

| Provisional stage | FN | Share |
| --- | ---: | ---: |
| Exact-scoring hierarchy/manual-standard disagreement | 157 | 31.5% |
| Finding absent from extracted phrases | 93 | 18.6% |
| Correct ID beyond current raw-eight candidates | 91 | 18.2% |
| Correct candidate supplied but mapper rejected/chose another | 91 | 18.2% |
| Correct ID remained outside deeper embedding retrieval | 50 | 10.0% |
| Extracted but assigned a non-Abnormal category | 17 | 3.4% |

These labels use phrase-alignment heuristics and ontology relationships. They
are traceable but not equivalent to expert clinical adjudication. The private
confirmation ledger has 2,406 rows, and a blinded 232-row disagreement packet
is ready for lab review.

The sanitized evidence is in
[`csc-confirmation-82-diagnostic-summary.json`](benchmarks/results/csc-confirmation-82-diagnostic-summary.json).

## Is periodic HPO updating the main problem?

No—not at the population level observed here.

The historical vector notebook printed 19,650 HPO terms. The official
`2025-05-06` release contains exactly 19,650 terms, and it is also the last
release before the workbook was created on June 24, 2025. This makes it the
likely historical vector ontology. The current release is `2026-06-23` with
20,413 terms.

Holding the embedding model, add-ons, query text, and vector-building rules
constant, manual-ID raw-eight retrieval was:

| HPO snapshot | Raw-eight oracle recall |
| --- | ---: |
| Current 2026-06-23 | 0.8671 |
| Likely historical 2025-05-06 | 0.8690 |
| Original-code-era 2024-12-12 | 0.8690 |

That approximately 0.2 percentage-point difference is far too small to explain
the observed recall deficit.

Ontology drift still matters for individual cases. Among the 1,316 distinct
reference/prediction identifiers audited, the 2025 versus current comparison
found 25 label, 58 synonym, 39 parent, and 55 definition changes. The manual
reference contains eight obsolete IDs with replacements and one obsolete ID
without a replacement. Benchmark normalization was corrected to follow
unambiguous `replaced_by` targets; this changed two discovery FP/FN pairs into
true positives.

See
[`hpo-snapshot-comparison.json`](benchmarks/results/hpo-snapshot-comparison.json)
and
[`csc-reference-quality-audit.json`](benchmarks/results/csc-reference-quality-audit.json).
Official release artifacts come from the
[HPO GitHub releases](https://github.com/obophenotype/human-phenotype-ontology/releases).

## Similar, equivocal, and obsolete HPO entries

This is a genuine limitation, although exact duplicate wording is not the
largest part of it.

- Only 13 normalized vector phrases map exactly to more than one HPO ID.
- The current vector bundle nevertheless includes 669 entries tied to obsolete
  IDs: 585 with replacements and 84 without.
- Because retrieval asks for eight raw metadata rows before deduplicating HPO
  IDs, synonyms and obsolete entries consume slots. Across extracted phrases,
  the raw eight contain a median of only six distinct HPO IDs.
- In 197 of 704 discovery extractions, at least two distinct IDs were within
  0.02 cosine similarity of the top result.
- Of the 335 confirmation false positives, 77 were more specific descendants,
  46 were less specific ancestors, and 51 were siblings of a manual reference.

With deterministic one-to-one parent/child matching, confirmation F1 rises
from 0.6534 to 0.7140 at one edge and 0.7273 at two edges. This does **not**
prove those predictions are clinically equivalent. It shows that strict exact
scoring treats many close ontology disagreements as fully wrong.

The workbook has a separate reference defect: six cells in Cases 8, 9, 30,
and 100 contain two comma-separated IDs. The strict scorer treats each entire
cell as an impossible identifier. Treating the IDs as alternatives raises
confirmation F1 only to 0.6559, so it is important but not the main deficit.
The lab owner subsequently confirmed that either ID is acceptable for one
finding. Alternative-group scoring is now the primary policy; the source
workbook remains unchanged. Values in the original investigation tables are
retained as the pre-adjudication record unless explicitly labeled corrected.

See
[`current-vector-metadata-audit.json`](benchmarks/results/current-vector-metadata-audit.json)
and
[`csc-confirmation-82-reference-sensitivity.json`](benchmarks/results/csc-confirmation-82-reference-sensitivity.json).

## Candidate retrieval is a material limitation

The original program searched 500 raw vectors and retained approximately 15–20
unique candidates. The rebuild asks FAISS for eight rows and only then removes
duplicate IDs.

Using manual descriptions as an offline candidate-coverage oracle on the
discovery cohort:

| Policy | Correct ID available |
| --- | ---: |
| Current raw eight | 0.8671 |
| Eight distinct IDs | 0.8909 |
| Sixteen distinct IDs | 0.9286 |
| Thirty-two distinct IDs | 0.9524 |
| Reconstructed legacy retrieval | 0.9425 |

Against raw eight, the mean per-case oracle-recall gain was 0.0166 for eight
distinct IDs, 0.0517 for sixteen, 0.0769 for thirty-two, and 0.0682 for the
legacy policy. All remained positive after Holm correction.

This is strong evidence that the current candidate policy limits recall.
However, it is an oracle experiment: it proves the correct manual ID becomes
available, not that the language model will select it without reducing
precision.

The original 245,916-entry lineage-expanded vector corpus was reproduced
exactly and stored efficiently as 42,384 collapsed entries with repetition
counts. See
[`csc-sample-30-retrieval-policy-comparison.json`](benchmarks/results/csc-sample-30-retrieval-policy-comparison.json)
and
[`legacy-vector-corpus-profile.json`](benchmarks/results/legacy-vector-corpus-profile.json).

## Other tested explanations

### Score threshold

A discovery-only threshold of 0.42 appeared to remove 14 false positives
without losing a true positive. Applied unchanged to confirmation, it removed
27 false positives but also 12 true positives. Precision rose to 0.7153,
recall fell to 0.6023, and F1 remained effectively unchanged at 0.6540. The
paired F1 interval crossed zero. A global threshold is therefore not supported
as the main fix.

### Mapping Suspected findings

Counterfactually mapping all 15 discovery “Suspected” rows added six true
positives and nine false positives. F1 moved from 0.6451 to 0.6476; the paired
interval crossed zero. Broadly mapping suspected findings is not supported.

### Historical prompt

The July 2025 prompt was run with the current Groq model, ontology, vectors,
and strict schema. It failed Cases 61 and 65 on three consecutive attempts.
On the 28 successful paired cases, micro F1 fell from 0.6512 to 0.5973. The
historical prompt neither recovered recall nor matched the current provider
reliability.

### Model variability and provider reliability

Across three observations of the fixed 20-case subset:

- mean within-case F1 range: 0.0697;
- maximum within-case F1 range: 0.1750;
- mean within-case F1 standard deviation: 0.0376;
- mean pairwise predicted-ID Jaccard overlap: 0.8006.

Variability is material, but the confirmation F1 deficit remains after paired
inference. Intermittent Groq HTTP 400 responses affected different cases and
were recoverable through resume with the current prompt. Two cases remained
incompatible with the historical prompt. See
[`provider-reliability-summary.json`](benchmarks/results/provider-reliability-summary.json).

## Additional limitations

- Historical files contain per-case TP/FP/FN counts, not the historical
  predicted IDs or phrases. The exact historical errors cannot be reconstructed.
- The deprecated historical model cannot be rerun under identical conditions.
- The manual annotations may omit plausible findings or prefer a different
  specificity. The blinded lab review has not yet been completed.
- Exact-set HPO scoring does not grant partial credit for clinically related
  nodes.
- The extraction-to-reference alignment is automated and sometimes one
  extracted phrase corresponds to multiple manual concepts.
- `HPO_addons.csv` has not changed since December 2024. It contains 81 IDs now
  obsolete with replacements and three obsolete without replacements.
- Repository cases are published case-report prose and may not represent
  typical clinical notes.
- Results cover one provider/model and one embedding model.

## Recommended remediation order

1. **Fix candidate construction.** Filter or retarget obsolete IDs first, fetch
   enough raw rows to supply at least 16 distinct HPO IDs, and record both raw
   and distinct rank. Validate final—not oracle—precision and recall on the
   locked benchmark.
2. **Complete blinded lab adjudication.** Resolve the 232 confirmation
   disagreements, especially ancestor/descendant pairs and compound reference
   cells. Preserve strict metrics and add a separately labeled adjudicated
   metric.
3. **Improve extraction recall.** Add explicit one-to-many phenotype
   segmentation and a validation pass. Test changes on a new held-out or
   cross-validated cohort rather than tuning the confirmation cases.
4. **Clean vector inputs.** Exclude obsolete terms without replacements,
   redirect unambiguous replacements, update `HPO_addons.csv`, and version its
   clinical review.
5. **Harden provider recovery.** Capture a sanitized provider error code/body
   category and permit bounded retries for Groq structured-output validation
   failures. Never silently change response mode.
6. **Do not adopt the global 0.42 threshold or map all Suspected rows.** Neither
   generalized into a meaningful F1 improvement.

No production CLI or annotation behavior was changed by this investigation.
All implementation is confined to benchmark tooling, scoring normalization,
tests, and sanitized reports.

## Verification status

- Exact 30-case and 82-case scores reproduce from stored predictions.
- Every discovery and confirmation FP/FN has a private trace record.
- Current and historical ontology vectors were rebuilt with the same pinned
  SapBERT revision.
- Candidate policies operate on distinct HPO IDs.
- The 82-case cohort and repeat subset were locked before inference.
- Raw notes, credentials, and provider responses are absent from tracked files.
- Lab review remains the only incomplete interpretive step; the blinded packet
  is ready outside Git.

# Benchmark reproduction

## Corpus boundary

CSC and GSC are distinct input/reference pairs:

- `CSC Input` / `Test_Cases.csv` → `CSC Manual Annotations`
- `GSC Input` → `GSC Manual Annotations`

Patient numbers are local to a corpus. Do not compare Test Case 1 with GSC Case
1 and do not join CSC to GSC only by patient number. Immutable CSV exports and
their source hashes are in [references/SOURCE.md](references/SOURCE.md).

Cases 67 and 68 are separate CSC records. Case 68 is synchronized from the
workbook-derived `csc_input.csv` export and has ten manual reference IDs.

## Score current predictions

`run_benchmark.py` evaluates actual normalized HPO sets:

```bash
python benchmarks/run_benchmark.py \
  --predictions rag_hpo_output/rag_hpo_results.json \
  --input Test_Cases.csv \
  --references benchmarks/references/csc_manual_annotations.csv \
  --ontology /path/to/hp.obo \
  --prompt-file src/rag_hpo/data/system_prompts.json \
  --vector-manifest /path/to/hpo_manifest.json \
  --case-id 1 \
  --model openai/gpt-oss-120b \
  --prompt-version 1.0 \
  --output-dir benchmark-results/case-1
```

The runner:

- maps current and alternate HPO IDs through the supplied ontology;
- treats each bounded `candidate_hpo_ids` set as one predicted phenotype by
  default and rejects sets larger than three;
- uses deterministic one-to-one matching, so any prediction/reference
  alternative intersection earns one TP and unused alternatives add no FP;
- supports `--prediction-unit selected-id` only for reproducing legacy
  document-ID analyses;
- writes the exact predicted, reference, TP, FP, and FN sets;
- writes per-case CSV plus micro/macro JSON metrics;
- records hashes for predictions, references, ontology, and the supplied vector
  manifest identity;
- produces byte-identical output for identical files and metadata.

Run Case 1 before an authorized subset. Live provider evaluation defaults to
the locked subsets described below; do not run a complete corpus merely for
routine development. Repository examples are published but are not
automatically approved for external transmission.

## Historical workbook arithmetic

The published workbook stores TP, FP, and FN counts as static values.
`recompute_metrics.py` checks their arithmetic:

```bash
python benchmarks/recompute_metrics.py \
  "RAG-HPO Tests and Data Analysis copy.xlsx" \
  --output benchmark-results/historical-metrics.csv
```

This does not reconstruct historical predictions. Premium, CSC, and GSC metrics
remain separately labeled and are not interchangeable gold standards.

For CSC Case 1, the historical LLaMa 4-Scout row is TP 3, FP 8, FN 5, F1
0.315789. It is a historical baseline, not a required performance guarantee for
the new provider/model.

## Paired accuracy comparison

The checked-in 30-case selection and reports are:

- `results/csc-sample-30-20260728-selection.json`
- `results/csc-sample-30-groq-gpt-oss-120b.json`
- `results/csc-sample-30-current-vs-llama4-scout.json`

The sample was fixed before live inference. Case 68 was required, and 29 other
eligible cases were selected without replacement using seed `20260728`.
`prepare_csc_sample.py` reproduces the selection; `compare_paired_models.py`
reproduces the paired bootstrap interval and sign-flip test. The raw sample
notes and provider predictions are deliberately kept outside the repository.

The result does not show an accuracy improvement over historical LLaMa
4-Scout: micro F1 was 0.6451 versus 0.6653, macro F1 was 0.6449 versus 0.6554,
and the paired 95% interval included zero. See [VALIDATION.md](../VALIDATION.md)
for the interpretation and retry record.

## Accuracy investigation

The checked-in investigation tools keep production annotation behavior
unchanged. They:

- trace each exact-set false negative through extraction, category assignment,
  candidate retrieval, and final mapping;
- compare exact scoring with one-to-one parent/child sensitivity scoring;
- compare official HPO snapshots and hash every ontology/vector input;
- reconstruct the original lineage-expanded 245,916-row vector corpus;
- lock an untouched confirmation cohort before any additional live inference;
- generate note-level ledgers and a blinded review packet only in the private
  audit-artifact directory.

Run the offline discovery diagnostics with `diagnose_pipeline.py`,
`compare_ontologies.py`, `compare_vector_snapshots.py`, and
`compare_retrieval_policies.py`. The sanitized outputs live under `results/`;
the detailed rows deliberately remain outside Git.

`audit_reference_quality.py` must be reviewed before interpreting a complete
CSC benchmark. Six source cells contain two comma-separated HPO IDs. The
lab owner confirmed that these are alternative IDs for one finding. The primary
benchmark now uses deterministic one-to-one alternative-group matching.
`score_reference_sensitivity.py` retains the former strict and all-required
interpretations only as historical sensitivity bounds. The source file remains
unchanged.

The 30-case cohort is discovery-only. The confirmation manifest contains all
remaining eligible cases and a predeclared 20-case repeat subset. Do not tune
prompts, thresholds, or candidate counts against those confirmation results.
The locked 82-case confirmation found rebuild micro F1 0.6534 versus
historical 0.7002. The paired F1 interval excluded zero in the negative
direction; see `results/csc-confirmation-82-analysis.json` and
[the investigation report](../RAG-HPO_ACCURACY_INVESTIGATION.md).

## Confidence calibration and cutoff selection

Do not translate model-provided high/medium/low labels directly into numeric
probabilities. After producing a bounded-candidate discovery run, calibrate
evidence strata without new provider inference:

```bash
python benchmarks/calibrate_candidate_confidence.py \
  --predictions /private/discovery/rag_hpo_results.json \
  --references references/csc_manual_annotations.csv \
  --ontology /private/artifacts/hp.obo \
  --prompt-file ../src/rag_hpo/data/system_prompts.json \
  --vector-manifest /private/artifacts/hpo_manifest.json \
  --selection-manifest results/csc-sample-30-20260728-selection.json \
  --minimum-precision 0.70 \
  --precision-criterion point \
  --output /private/discovery/confidence_calibration.json
```

For calibration only, a prediction candidate set that intersects its matched
gold alternative group receives `gold_match=1`; otherwise it receives zero.
The locked selection manifest is required so cases with zero predictions
remain in the recall denominator.
The default selects the highest-recall cumulative evidence policy whose
precision point estimate meets the frozen floor and reports its Wilson 95%
lower bound. Use `--precision-criterion lower95` for a much more conservative
policy. Neither discovery policy should be promoted without held-out
confirmation. The file contains no note text and can be applied only when its
prompt and artifact hashes match:

```bash
rag-hpo annotate ... \
  --mode balanced \
  --confidence-calibration /private/discovery/confidence_calibration.json
```

The existing references cannot separately identify “real phenotype but wrong
mapping” from “spurious phenotype.” The first calibration is therefore an
overall correctness probability. Separate phenotype and mapping-set
probabilities require span-level human labels and remain null until those
labels exist.

## FastHPOCR Phase 1

Install the optional research dependency with:

```bash
pip install -e '.[fasthpocr]'
```

`run_fasthpocr.py` builds or validates a private, provenance-recorded index and
emits span-level predictions plus alternative-aware metrics. The index and
note-level results belong outside Git. `summarize_historical_metrics.py`
creates the corpus-aware historical comparison, and
`compare_recognizer_predictions.py` measures overlap without mixing CSC and
GSC identifiers.

Phase 1 found that longest-span filtering improved precision-weighted and F1
results on both CSC and GSC. A raw union with RAG-HPO increased recall but
reduced precision and F1, while exact-ID agreement reached 0.937 precision.
See [the Phase 1 report](../PHASE1_FASTHPOCR_FINDINGS.md) and
[`fasthpocr-phase1-summary.json`](results/fasthpocr-phase1-summary.json).

## Canonical registry and Phase 2

Build the terminology registry without loading an embedding model:

```bash
python benchmarks/build_registry.py \
  --ontology /path/to/pinned/hp.obo \
  --addons HPO_addons.csv \
  --output-dir /private/path/to/registry
```

The builder writes deterministic registry and lexical artifacts with their
manifests. Full Phase 2 evidence is summarized in
[`phase2-registry-summary.json`](results/phase2-registry-summary.json).
FastHPOCR benchmarks now derive their external-synonym view from this registry
and retain exact or morphological multi-ID matches as ambiguity sets.

## Phase 3 cascade and verifier

`run_phase3_cascade.py` evaluates the deterministic confidence lane.
`run_phase3_hybrid.py` performs one optional batched verification call per case,
and `compare_phase3_hybrid.py` replays the retained policy and performs paired
inference. Note-level decisions and provider-derived predictions must be
written outside Git.

The deterministic lane is not a complete annotator: its CSC recall was 0.208.
The corrected confirmation hybrid retained 0.602 recall while increasing
precision from 0.704 to 0.749. It did not reach the 0.80 precision requirement
for default promotion. See [the Phase 3 report](../PHASE3_CASCADE_FINDINGS.md)
and the sanitized
[`phase3-hybrid-summary.json`](results/phase3-hybrid-summary.json).

## Staged 70/70 remediation

`evaluate_hybrid_retrieval.py` measures active canonical-ID coverage at 16 and
32 candidates. `analyze_staged_policies.py` compares only general
confidence/method policies on the fixed discovery cohort and emits aggregate
results; it does not add case-specific logic.

Primary staged scoring uses:

```bash
python benchmarks/run_benchmark.py \
  --predictions /private/path/rag_hpo_results.json \
  --input benchmarks/references/csc_input.csv \
  --references benchmarks/references/csc_manual_annotations.csv \
  --ontology /private/path/hp.obo \
  --prompt-file src/rag_hpo/data/system_prompts.json \
  --vector-manifest /private/path/hpo_manifest.json \
  --selection-manifest results/csc-confirmation-stratified-30-20260728-selection.json \
  --accepted-only \
  --model openai/gpt-oss-120b \
  --prompt-version staged-2.0 \
  --output-dir /private/path/score
```

`--accepted-only` excludes review/rejected rows. Omitting it is a high-recall
sensitivity analysis, not the primary result. Prediction-only cases with no
manual reference findings are excluded from the denominator; reference-only
cases remain and count as false negatives.

The post-discovery settings and source/artifact hashes are frozen in
[`70-70-frozen-config.json`](provenance/70-70-frozen-config.json). Raw notes,
provider responses, prediction rows, and adjudication packets remain in the
owner-only audit-artifact directory.

Live evaluation is subset-only by default. `prepare_stratified_subset.py`
selects 30 cases per corpus without reading predictions or scores. It uses a
fixed seed and 3x3 strata over note length and manual-reference count. CSC
sampling excludes the 30 discovery cases; GSC is sampled independently. The
locked manifests are:

- `results/csc-confirmation-stratified-30-20260728-selection.json`
- `results/gsc-stratified-30-20260728-selection.json`

These samples contain 446 CSC and 268 GSC manual-reference findings. Report
bootstrap intervals with their metrics by running
`bootstrap_subset_metrics.py` against the accepted-only benchmark report and
its selection manifest. Label the results as subset estimates, not full-corpus
results. Neither selected subset may be used to retune the frozen
configuration. Full-corpus provider runs require an explicit, documented
reason; routine development and regression work must use the fixed subsets or
smaller synthetic fixtures.

`run_corpus_evaluation.py` is the resumable end-to-end wrapper for authorized
live evaluation. Exactly one of `--case-id`, `--selection-manifest`, or
`--all` is required. Remote runs require
`--confirm-external-transmission`; the API key is read only from the
environment. The wrapper:

- writes the selected note input and all outputs outside Git with private
  permissions;
- retries only unfinished/error rows up to `--max-attempts`;
- preserves a compatible SQLite checkpoint so the identical command can be
  rerun safely;
- produces accepted-only alternative-set exact metrics; and
- produces distinct one- and two-edge DAG sensitivity reports without new
  provider calls.

```bash
python benchmarks/run_corpus_evaluation.py \
  --corpus csc \
  --vector-dir /private/artifacts/hpo \
  --ontology /private/artifacts/hp.obo \
  --output-dir /private/results/csc-full \
  --all \
  --confirm-external-transmission
```

The locked subset result is summarized in
[`staged-70-70-subset-summary.json`](results/staged-70-70-subset-summary.json)
and [the findings report](../PHASE4_70_70_FINDINGS.md). CSC passed the
accepted-only 0.70/0.70 point gate. GSC precision was 0.858 and recall was
0.698, one true positive below the recall gate, so `balanced` was not promoted
to the default.

## Context, hierarchy, and consistency evaluation

Context-aware zero-shot and fixed synthetic one-shot mapping use the same
extraction and retrieval calls. Select them with `--mapping-prompt`; do not
choose a prompt on the locked confirmation subsets.

The completed prompt screen used `run_context_mapping_ablation.py` to reuse
fixed discovery extraction and candidates. Zero-shot reached precision 0.648
and recall 0.603; one-shot reached precision 0.656 and recall 0.615. Neither
met the predeclared 0.70 precision floor, so no prompt advanced to expensive
three-run confirmation. See
[`context-prompt-screen-summary.json`](results/context-prompt-screen-summary.json).

Keep strict alternative-aware exact scoring as the primary result. Produce
separate one- and two-edge sensitivity results without new inference:

```bash
python benchmarks/score_layered.py \
  --predictions /private/run/rag_hpo_results.json \
  --references references/csc_manual_annotations.csv \
  --ontology /private/artifacts/hp.obo \
  --selection-manifest results/csc-confirmation-stratified-30-20260728-selection.json \
  --accepted-only \
  --output /private/score/layered.json
```

The scorer matches one prediction to one alternative-aware reference finding,
prioritizing exact IDs before one-edge and then two-edge parent, child, or
sibling relationships. Relaxed results are sensitivity analyses, not extra
accepted predictions.

Measure repeatability from three already-produced runs of a fixed 10–20 case
subset:

```bash
python benchmarks/evaluate_consistency.py \
  --predictions /private/run-1/rag_hpo_results.json \
  --predictions /private/run-2/rag_hpo_results.json \
  --predictions /private/run-3/rag_hpo_results.json \
  --references references/csc_manual_annotations.csv \
  --ontology /private/artifacts/hp.obo \
  --selection-manifest /private/repeat-subset-selection.json \
  --output /private/score/consistency.json
```

Predeclared defaults require micro precision/recall/F1 ranges no larger than
0.02, mean pairwise case Jaccard at least 0.85, and at least 90% of accepted
IDs to recur in all runs. The report also records disposition agreement,
per-case F1 ranges, input hashes, retry/usage data, model configuration, and
runtime identity when the run manifests are available.

The no-new-inference hierarchy results and the historical 20-case consistency
baseline are summarized in
[CONTEXT_CONSISTENCY_FINDINGS.md](../CONTEXT_CONSISTENCY_FINDINGS.md).

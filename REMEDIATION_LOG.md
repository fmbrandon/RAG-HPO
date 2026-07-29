# RAG-HPO remediation log

This log records sanitized implementation actions. API keys, complete notes,
and raw provider responses are excluded.

## 2026-07-28 — Phase 1: FastHPOCR licensing and evidence

- Confirmed a clean starting tree on `codex/rebuild-rag-hpo-v0.2.0`.
- Verified the workbook and immutable CSC/GSC exports against their recorded
  SHA-256 hashes.
- Installed FastHPOCR 0.1.4 into the project `.venv` for the audit.
- Inspected its source distribution, installed resources, package metadata,
  imports, public indexer, annotator, and MIT license.
- Added `rag-hpo[fasthpocr]` as a pinned optional extra and declared the missing
  Pronto runtime dependency explicitly.
- Added third-party notices, package/resource hashes, and a no-vendoring rule.
- Added alternative-group scoring so a comma-delimited reference cell counts
  as one finding satisfied by either ID.
- Corrected benchmark selection to include reference-only cases and to exclude
  input cases with no manual findings from accuracy denominators.
- Added a private FastHPOCR index/benchmark runner retaining IDs, spans,
  versions, hashes, configuration, timing, and row-level failures.
- Added a corpus-aware historical summary tool and a recognizer overlap tool.
- Reproduced all 1,730 historical metric rows byte-for-byte.
- Built current HPO indexes with and without 3,079 RAG-HPO add-on phrases.
- Ran default and longest-match configurations on all eligible CSC and GSC
  cases with zero annotation errors.
- Rebuilt the official-only index twice and demonstrated upstream output
  nondeterminism affecting one selected HPO ID.
- Compared rebuilt RAG-HPO with FastHPOCR on 112 shared CSC cases.
- Recorded the findings and revised Phases 2–4 accordingly.

Validation performed:

- Pytest: 73 tests passed with 87.54% branch-aware package coverage
  (85% required).
- Ruff on the complete repository: passed.
- Mypy on all 21 package source files: passed.
- `pip check`: passed.
- `pip-audit`: no known vulnerabilities.
- Wheel and source-distribution builds: passed.
- FastHPOCR optional-extra installation and import: passed.
- New JSON provenance and result summaries: syntactically valid.
- Git diff whitespace validation: passed.
- Private artifact permissions: directories `0700`, sampled files `0600`.

## 2026-07-28 — Phase 2: canonical registry and deterministic views

- Added a versioned canonical HPO registry retaining concept status, alternate
  IDs, direct parents, definitions, synonym scopes, extension provenance,
  normalized phrases, ambiguity groups, and stable phrase keys.
- Added a deterministic native lexical artifact and independent manifests for
  the registry and lexical view.
- Linked dense metadata and vectors to all four registry artifacts by SHA-256.
- Changed vector generation to derive metadata exclusively from the registry.
- Excluded obsolete concepts from new mapping artifacts while retaining them
  in the registry for normalization and audit.
- Normalized add-on rows using official alternate IDs; unknown IDs now fail
  instead of being silently ignored.
- Added UTF-8 validation and rejection of replacement characters, NULs, and
  known mojibake sequences.
- Changed the FastHPOCR adapter to reconcile every ID through the registry.
- Added token-signature reconstruction so exact and morphological multi-ID
  matches are emitted as ordered ambiguity sets.
- Added byte, semantic, and fixed-probe output fingerprints to FastHPOCR
  manifests.
- Changed the FastHPOCR external-synonym input to be derived from canonical
  extension records rather than directly from the CSV.
- Added a registry-only benchmark builder and machine-readable Phase 2 result
  summary.
- Built the complete registry twice; all registry and lexical hashes matched.
- Built the complete 47,859-row SapBERT view twice; metadata and vector hashes
  matched.
- Built and revalidated a full canonical FastHPOCR add-on index.
- Normalized both prior byte-different indexes across all 116 CSC inputs; their
  complete normalized predictions matched.

Phase 2 validation performed:

- Pytest: 81 tests passed with 86.04% branch-aware package coverage
  (85% required).
- Ruff on the complete repository: passed.
- Mypy on all 22 package source files: passed.
- Bandit: no medium- or high-severity findings.
- `pip check`: passed.
- `pip-audit`: no known vulnerabilities.
- Detect-secrets credential detectors: no findings. Generic high-entropy
  findings were recorded hashes and immutable revisions, not credentials.
- Wheel and source-distribution builds: passed and contain the registry module.
- Built packages contain no private audit-artifact paths.
- `rag-hpo doctor` validated the complete 47,859-row, 768-dimensional artifact.
- New machine-readable JSON: syntactically valid.
- Git diff whitespace validation: passed.
- Direct Groq-key-pattern scan: no findings.
- Private Phase 2 directory permissions: `0700`; registry, vectors, and
  FastHPOCR index: `0600`.

## 2026-07-28 — Phase 3: precision cascade and recall repair

- Added independent native lexical recognition with deterministic exact and
  morphological token signatures, acronym exclusion, longest spans, and
  retained multi-ID ambiguity.
- Added sentence/section assertion analysis for affirmation, negation, family
  history, uncertainty, resolution, and hypothetical mentions.
- Added a deterministic confidence cascade plus strict Pydantic schemas for
  batched candidate and prediction verification.
- Added two versioned prompts without changing the existing extraction,
  mapping, or validation prompt text.
- Added 22 focused Phase 3 tests, including ambiguity, out-of-set response,
  incomplete response, high-confidence rejection, and recall-preserving
  abstention cases.
- Ran the offline confidence lane on all eligible CSC and GSC cases. It
  generalized in precision but had unacceptable recall and was rejected as a
  final annotator.
- Ran live batched verification on the fixed 30-case discovery and locked
  82-case confirmation cohorts.
- Audited every rejection source. Local assertion deletion removed more true
  positives than false positives and was replaced with a non-deleting flag.
- Replayed the corrected policy: only model-reviewed
  unsupported/high-confidence findings are removed.
- Strengthened the recall gate to require at least 0.58 and no more than a 0.02
  decline from the base model.
- Confirmation precision improved from 0.704 to 0.749; recall changed from
  0.614 to 0.602; F1 improved from 0.656 to 0.668; F0.5 improved from 0.684
  to 0.714.
- The paired confirmation F0.5 interval was fully positive. The 0.80 precision
  promotion gate failed, so no default CLI behavior was changed.
- Measured verifier overhead at approximately 70,478 input tokens and 2.66
  seconds per confirmation case; raw notes, responses, and credentials remain
  outside Git.

Phase 3 validation performed:

- Pytest: 103 tests passed with 86.94% branch-aware package coverage
  (85% required).
- Ruff lint and formatting checks: passed across all 91 checked files.
- Mypy: passed across all 26 package source files.
- Bandit: no medium- or high-severity findings.
- `pip check`: passed.
- `pip-audit`: no known vulnerabilities.
- Wheel and source-distribution builds: passed; the wheel contains the new
  assertion, lexical, hybrid, and prompt resources.
- Direct Groq-key-pattern scan: no findings.
- Git diff whitespace and tracked JSON validation: passed.

## 2026-07-28 — Two-phase 70/70 remediation

- Added five public annotation modes: compatibility model, balanced staged,
  high-recall staged, native recognizer-only, and optional FastHPOCR
  recognizer-only.
- Added repeatable recognizer selection, offline network enforcement,
  no-model execution, and opt-in evidence-text retention with a privacy
  warning.
- Added backward-compatible evidence offsets, assertion status, confidence,
  review disposition, contributing methods, and candidate IDs to CSV/JSON
  results.
- Added active-ID candidate retrieval combining pinned SapBERT, character
  n-gram TF-IDF, RapidFuzz, exact labels, synonyms, add-ons, and deterministic
  canonical-ID rank fusion.
- Added strict two-pass exact-phrase extraction. Numeric offsets are recovered
  locally after Groq strict mode proved unreliable when asked to generate many
  offsets in one response.
- Added targeted sentence auditing for unmatched recognizer findings,
  abnormal measurements, lists, and possible compound findings.
- Replaced per-phenotype full-note mapping with compact sentence-local batches.
  Incomplete and duplicate responses fail safely; strict HTTP 400 batches
  split recursively; out-of-candidate IDs cannot become accepted answers.
- Kept local assertion analysis non-deleting. Negated, uncertain, resolved,
  hypothetical, and family-history findings move to review.
- Added accepted-only primary scoring and aggregate discovery-policy analysis.
- Added a redacted run manifest with input/artifact/prompt hashes, elapsed
  time, and provider token counts.
- Documented every new prompt with its purpose and SHA-256 in
  `PROMPT_CHANGELOG.md`.
- Measured discovery description retrieval at 0.9345 within 16 distinct IDs
  and 0.9544 within 32. The 32-ID gate passed; the 16-ID gate remains three
  reference findings short.
- Completed a live staged Case 1 smoke with no row errors: 5 TP, 8 FP, 3 FN,
  precision 0.385, recall 0.625, and F1 0.476. This is diagnostic and does not
  satisfy the 70/70 promotion target.

## 2026-07-29 — Context and consistency hardening

- Added deterministic context packets to the existing batched mapping call.
  Adjacent sentences are bounded and included only for an assertion,
  short/ambiguous phrase, context cue, or close top-candidate scores.
- Added explicit zero-shot and fixed synthetic one-shot prompt modes. Zero-shot
  remains the staged default; no benchmark example is inserted into a prompt.
- Added alternative-aware exact, one-edge, and two-edge one-to-one scoring.
  Exact remains primary and hierarchy-aware results remain sensitivity
  analyses.
- Added a repeat-run consistency evaluator with predeclared metric, Jaccard,
  ID-recurrence, and disposition-agreement outputs.
- Extended private run manifests with platform, embedding identity, mapping
  mode, temperatures, batch size, and provider determinism limitations.
- Performed no live provider run, full-corpus evaluation, vector rebuild, or
  prompt selection. Validation used synthetic fixtures and existing artifacts.
- Stopped the in-progress full CSC and GSC provider runs after the evaluation
  policy was narrowed to save time and tokens. Their partial outputs remain
  private and are not used as final evidence.

## 2026-07-29 — CI repair and bounded context-prompt screen

- Replaced four runtime `assert` statements in the diagnostic OBO parser with
  direct typed-container operations. Full Bandit now reports no findings.
- Added linked-artifact failure tests for incomplete provenance, unsupported
  registry schema, and loaded registry/lexical identity mismatches.
- Reached 100% branch coverage in every safety-critical target and 85.80%
  overall branch-aware coverage; 124 tests pass.
- Added a mapping-only ablation runner so extraction, embedding, and candidate
  retrieval are reused rather than recomputed.
- Corrected an initial resource pilot that over-supplied candidate metadata,
  then corrected a policy pilot that incorrectly remapped review-only
  recognizer findings. Neither pilot was used as accuracy evidence.
- Final zero-shot and one-shot arms each used 37 requests. Together they used
  613,573 tokens, versus repeating complete extraction and retrieval.
- Neither context prompt met the predeclared 0.70 precision screening floor,
  so no multi-run confirmation was launched.
- Moved the CI secret-scan report outside the checkout so the detector cannot
  scan its own partially written high-entropy JSON output. Failure diagnostics
  now expose only filename, line number, and detector type.
- Narrowly excluded benchmark provenance JSON and the archived historical
  prompt record from secret scanning. Their flagged values are documented Git
  commit and SHA-256 identifiers; executable source and current prompts remain
  scanned.

## 2026-07-29 — three-bin categorization and bounded-set scoring

- Changed staged extraction to return source spans without assigning a
  category. Native, FastHPOCR, first-pass, and selective audit spans merge
  before categorization.
- Reduced phenotype categories to Abnormal, Normal, and Family History.
  Suspected, historical, resolved, and hypothetical status remains assertion
  metadata; non-phenotype errors use a null category.
- Added deterministic early routing for explicit normal/negated and
  family-history findings, plus one consolidated whole-note final category
  request for abnormal and ambiguous patient findings after mapping. Routing
  is not serialized as a category, so each output finding receives one
  definitive category.
- Added telegraphic headings, bullets, observation predicates, lists,
  measurements, and coordinated structures as selective second-pass cues.
- Changed staged mapping to return a clinically bounded alternative set of at
  most three supplied IDs. The full retrieval pool is retained in a separate
  audit field.
- Added one-to-one prediction-set/reference-set scoring. One intersecting ID
  yields one true positive; unused alternatives are not false positives and a
  single prediction cannot satisfy multiple reference findings.
- Added optional held-out confidence calibration. Gold overlap is stored as a
  binary calibration outcome, while runtime numeric confidence remains absent
  unless the prompt and artifact hashes match the calibration file.
- The cutoff selector maximizes recall subject to a frozen precision floor.
  It supports a practical point-estimate criterion and a stricter Wilson 95%
  lower-bound criterion; both are reported, and unseen evidence remains in
  review rather than being rejected.
- Added performance-blind subset selection stratified by note length and
  manual-reference count. The fixed seed is 20260728. The 30-case untouched
  CSC confirmation subset contains 446 references; the independent 30-case
  GSC subset contains 268. Manifests contain case IDs and hashes but no note
  text.

Validation performed so far:

- Pytest: 146 tests passed with 85.86% branch-aware package coverage.
- Ruff and mypy: passed for package, tests, and new benchmark tools.
- Two synthetic live provider schema smokes were used as a gate; the corrected
  run returned five nonduplicated findings without errors in four requests and
  8,766 total provider tokens.

Discovery evaluation:

- Ran the fixed 30-case CSC discovery cohort once with prompt schema 1.4.
  Malformed provider decisions initially caused 13 case-level failures.
- Changed incomplete, duplicate, out-of-candidate, and unsplittable HTTP 400
  decisions into explicit finding-level review flags. Out-of-candidate IDs are
  discarded and can never be accepted.
- Retried only the 13 failed cases, then only the single remaining failed case.
  The assembled 30-case artifact has zero case-level errors. Across all
  attempts it used 1,079 requests and 3,717,562 provider tokens; this is too
  costly for routine use and is a measured limitation.
- Strict candidate-set discovery metrics were:
  - qualitative accepted policy: precision 0.6913, recall 0.5198, F1 0.5934;
  - all mapped candidates: precision 0.6800, recall 0.6071, F1 0.6415;
  - point-floor calibration: precision 0.7011, recall 0.6052, F1 0.6496.
- The stricter Wilson-lower-bound policy produced precision 0.7722 but recall
  0.2421. It is too conservative for the stated recall goal.
- Because even the all-mapped discovery ceiling was only 0.6071 recall, no GSC
  inference was run and no revised policy was promoted. Further token spend
  would not validate a configuration that already failed its discovery gate.
- Mypy: passed for all 28 package source files.
- `pip check`: passed.
- `pip-audit`: no known dependency vulnerabilities.
- Bandit: no medium- or high-severity findings; four existing low-severity
  assertion findings remain in diagnostic code.
- Wheel and source distribution: built successfully.
- The CSC subset reproduced byte-for-byte from the recorded seed and source
  files.
- Live strict extraction, targeted audit, bounded mapping, normalized export,
  and Case 1 scoring completed.
- Locked CSC/GSC subset evaluation, repeated-run variability, and final release
  checks remain recorded as open gates until their runs finish. Full-corpus
  live runs are no longer a routine gate.
- Completed both locked 30-case subset runs. CSC accepted-only precision,
  recall, and F1 were 0.727, 0.711, and 0.719. GSC values were 0.858, 0.698,
  and 0.770. CSC passed the point gate; GSC recall failed by one true
  positive. `balanced` was not promoted.
- Added deterministic 20,000-iteration case-bootstrap intervals. Both corpora
  have intervals crossing 0.70, so the point estimates are not described as
  proof of corpus-wide performance.
- Verified SQLite recovery in the live CSC run: four transient provider
  failures were retried without rerunning completed rows, and all recovered.
- Fixed resume cost accounting so future manifests retain per-attempt and
  cumulative token/runtime totals. The evaluation behavior and scores were not
  changed.

## 2026-07-29 — stage-loss remediation and five-case smoke

- Removed balanced mode's pre-mapper deferral. Every abnormal mention with a
  nonempty bounded candidate pool now reaches the batched mapper, regardless
  of whether it came from pass one, pass two, native recognition, or optional
  FastHPOCR.
- Changed nested-span consolidation to preserve the broader source phrase and
  its modifiers. Narrow recognizer IDs and methods are attached without
  replacing clinical meaning.
- Added ordered non-contiguous evidence alignment for coordinated phrases.
  Exact source segments are exported separately; no false contiguous offset is
  invented.
- Added clause-local assertion boundaries for a new patient, examination, or
  imaging clause after “and.” This prevents an earlier normal result from
  making an independent later finding normal.
- Treats `HP:0012823 Clinical modifier` and all ontology descendants as
  attributes rather than standalone phenotype mappings. Modifier IDs remain
  auditable in `modifier_hpo_ids`.
- Added one batched context-enriched SapBERT query for short phrases. The
  original phrase query and lexical query remain independent. A context-only
  rank-one candidate has enough reciprocal-rank weight to survive the
  16-candidate balanced bound.
- Versioned the staged pipeline as 3.2 and retrieval policy as 2.0 so old
  checkpoints/calibrations cannot be mixed with the changed behavior.
- Added a resumable corpus wrapper that annotates, retries only failed rows,
  scores bounded alternative sets exactly, and emits one-/two-edge hierarchy
  sensitivity without additional inference.

Five-case CSC smoke (Cases 1, 4, 34, 67, and 99):

- strict accepted-only micro: TP 41, FP 21, FN 18; precision 0.6613, recall
  0.6949, F1 0.6777;
- strict macro: precision 0.7033, recall 0.7131, F1 0.7034;
- one-/two-edge sensitivity: precision 0.7258, recall 0.7627, F1 0.7438,
  including four one-edge ancestor/descendant matches;
- Case 99: precision 1.0, recall 0.8571, F1 0.9231; standalone laterality and
  spatial-modifier IDs were not counted as phenotypes;
- the context lane placed `HP:0005227` inside the bounded candidate set for the
  short “tubular adenoma” mention and it was recovered elsewhere in Case 1;
- the final run completed with zero row errors in 64 provider requests and
  420,564 reported tokens.

This five-case cohort is a smoke comparison, not a statistically conclusive
accuracy estimate. Strict micro recall was within one true positive of 0.70,
but strict micro precision remained below 0.70. The hierarchy sensitivity
passed both thresholds and must remain separately labeled.

## 2026-07-29 — observable full-corpus launchers

- Added a terminal progress bar backed by the SQLite checkpoint rather than
  inferred provider activity. It refreshes every second and reports successful
  cases, elapsed time, retry attempt, and row errors.
- Progress is enabled by default for corpus evaluation and can be disabled
  with `--no-progress` for automation.
- Split the private complete evaluation into independent CSC and GSC shell
  launchers. Either command can be stopped and rerun without starting or
  repeating the other corpus.
- The local launchers force the already-cached Hugging Face model into offline
  mode, removing unauthenticated Hub checks while leaving Groq inference
  unchanged.

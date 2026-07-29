# RAG-HPO remediation plan after Phase 3

## Phase 1 — Complete: licensing, evidence, and benchmark boundary

Phase 1 approved FastHPOCR 0.1.4 as an attributed optional dependency, made
alternative-ID scoring primary, reproduced the historical workbook arithmetic,
ran current FastHPOCR across CSC and GSC, and measured its overlap with rebuilt
RAG-HPO. See [PHASE1_FASTHPOCR_FINDINGS.md](PHASE1_FASTHPOCR_FINDINGS.md).

The evidence changed the remaining plan in four important ways:

1. Use longest-span matching as the precision-first FastHPOCR configuration.
2. Treat exact-ID agreement as a high-confidence signal.
3. Never use a raw union as final output; recognizer-unique findings require
   verification.
4. Do not trust FastHPOCR's serialized index hash alone because identical
   builds can select different ambiguous IDs.

## Phase 2 — Complete: canonical registry and deterministic derived indexes

Phase 2 created the versioned concept registry, separate extension records,
native lexical view, registry-linked SapBERT artifacts, and a deterministic
FastHPOCR adapter. It also added UTF-8/corruption validation and byte,
semantic, and output fingerprints. See
[PHASE2_REGISTRY_FINDINGS.md](PHASE2_REGISTRY_FINDINGS.md).

The complete native artifacts reproduced byte-for-byte. Two previously
byte-different FastHPOCR indexes also produced identical normalized predictions
across all 116 CSC inputs. Morphological equivalence is now retained as an
ordered candidate set instead of being resolved by iteration order.

Phase 2 supplied two new constraints for Phase 3:

1. The 15 exact lexical ambiguity groups and 53 morphologically ambiguous CSC
   mentions must enter verification or abstention; they cannot be treated as
   ordinary single-ID recognizer output.
2. Obsolete concepts are available for normalization and audit, but only
   active concepts may enter new mapping artifacts.

## Phase 3 — Complete: native cascade and recall-preserving verification

Phase 3 independently implemented morphological/token-signature recognition,
acronym exclusion, ambiguity preservation, longest-span selection, assertion
flags, and batched verification. No FastHPOCR source or resource was copied.

The evidence rejected the deterministic cascade as a final annotator because
CSC recall was 0.208. It remains a high-confidence lane. The corrected hybrid
retains local assertion flags and ambiguous model reviews, deleting only
high-confidence unsupported model predictions.

On the locked 82-case confirmation cohort, precision increased from 0.704 to
0.749, recall changed from 0.614 to 0.602, F1 increased from 0.656 to 0.668,
and F0.5 increased from 0.684 to 0.714. The paired F0.5 95% interval was
0.0220 to 0.0492. Added verifier input was approximately 12% of the current
pipeline's conservative token lower bound.

The recall and paired-F0.5 gates passed. The 0.80 default-precision gate did
not pass, so the hybrid remains benchmark-only/experimental. See
[PHASE3_CASCADE_FINDINGS.md](PHASE3_CASCADE_FINDINGS.md).

## Phase 4A — Recall expansion and better mapping

Status: implemented; discovery validation is in progress.

- Added `model`, `balanced`, `high-recall`, `native`, and `fasthpocr` modes,
  repeatable recognizer selection, `--offline`, `--no-model`, and opt-in
  evidence text.
- Added active-ID hybrid retrieval using exact labels, synonyms, add-ons,
  RapidFuzz, character n-gram TF-IDF, and pinned SapBERT embeddings. Alternate
  and obsolete IDs normalize before deterministic rank fusion.
- Added distinct-candidate limits of 16 for balanced mode and 32 for
  high-recall mode. The fixed 30-case discovery descriptions currently reach
  0.9345 recall at 16 and 0.9544 at 32. The 32-candidate gate passes; the
  16-candidate 0.94 gate remains narrowly unmet without case-specific rules.
- Added exact-source phrase extraction, deterministic local span recovery,
  compound/list audit cues, and a targeted second extraction pass.
- Added sentence-local mapping with compact candidate metadata and stable
  eight-item batches. Strict-output failures recursively split into smaller
  batches; incomplete, duplicate, and out-of-set decisions cannot become
  accepted mappings.
- Added provider-neutral local-model support. Offline mode rejects remote
  endpoints and model downloads but permits a loopback OpenAI-compatible
  server with cached artifacts.
- Added run manifests containing hashes, mode, redacted provider
  configuration, elapsed time, and token totals without note text or secrets.

Remaining acceptance work:

- Complete the discovery run and quantify pre-verification candidate recall,
  precision, tokens, and runtime.
- Freeze the general ranking and extraction configuration before any
  independent evaluation.
- Do not add any case ID, reference answer, or case-specific phrase to
  production logic or prompts.

## Phase 4B — Recall-preserving verification and final validation

Status: implemented and evaluated; default promotion did not pass.

- Added evidence-span assertion status, confidence, accepted/review/rejected
  disposition, contributing methods, and retained candidate IDs to normalized
  CSV/JSON output.
- Direct model/recognizer agreement is accepted. Supported additions may be
  accepted; ambiguous additions remain in review; only high-confidence
  unsupported additions are rejected.
- Negation, uncertainty, resolution, hypothetical use, and family history are
  review flags. They cannot independently delete a finding.
- Existing first-pass model findings are not subjected to a broad deletion
  pass. Added recognition/audit lanes receive the extra verification scrutiny.
- Added strict accepted-only benchmark scoring while retaining an all-mapped
  high-recall sensitivity analysis. Alternative HPO IDs in one reference cell
  continue to count as one finding satisfied by any listed ID.
- Added aggregate discovery-policy analysis so confidence and method policies
  can be selected without case-specific tuning.

Current evidence:

- The first complete staged Case 1 run finished without row errors and scored
  5 TP, 8 FP, and 3 FN (precision 0.385, recall 0.625, F1 0.476). It improved
  recall over the historical Case 1 result but does not meet the precision or
  recall target.
- The 30-case discovery run is used only for configuration selection. CSC/GSC
  configuration was frozen before subset inference.
- Future live evaluation is limited to performance-blind, stratified
  30-case subsets. The CSC subset excludes all discovery cases and contains
  446 reference findings; the independent GSC subset contains 268. Selection
  manifests record the seed, IDs, strata, and hashes. Earlier partial
  full-corpus runs are aborted and are not final evidence.
- On the locked CSC subset, accepted findings reached precision 0.727, recall
  0.711, and F1 0.719, passing the point-estimate target.
- On the independent locked GSC subset, accepted findings reached precision
  0.858, recall 0.698, and F1 0.770. Recall missed the point-estimate target by
  one true positive. The all-mapped review sensitivity reached 0.705 recall,
  but review findings are not counted as accepted.
- The case-bootstrap intervals cross 0.70. These results are subset estimates,
  not proof of full-corpus performance.

Promotion remains conditional:

- Strict accepted-only micro precision and recall did not both reach 0.70 on
  both locked subsets. The promotion gate therefore failed.
- High-recall output must keep unresolved findings in review rather than count
  candidate sets as correct answers.
- Median runtime must remain no more than twice the compatibility pipeline,
  and provider-token totals must stay within the predeclared balanced and
  high-recall limits.
- `model` remains the default and staged modes stay clearly experimental. No
  individual independent-set failures will be used for retuning.

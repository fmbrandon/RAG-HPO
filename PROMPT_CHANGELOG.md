# Prompt changelog

Prompt text is versioned in `src/rag_hpo/data/system_prompts.json`. Prompt
changes require a version change, a new hash, and benchmark evidence before
default behavior changes.

## 2026-07-28 — prompt bundle schema 1.1

Bundle SHA-256:
`2ca340ed1a7e43a8e852fda4e152cf257df2a1a7acd5fee318ae1c79deea9bb0`

- `phenotype_extraction` 1.0: unchanged.
- `hpo_mapping` 1.0: unchanged.
- `phenotype_validation` 1.0: unchanged.
- `cascade_verification` 1.0: added for resolving tied lexical candidates
  using sentence context and supplied IDs only. It requires explicit
  distinguishing evidence for a more-specific concept and permits abstention.
- `prediction_verification` 1.0: added for batched review of model-selected
  mappings. It cannot invent or replace an ID. Production policy removes only
  high-confidence unsupported predictions and retains ambiguous findings.

These additions are used only by benchmark Phase 3 tooling. Existing annotation
prompts and default CLI behavior did not change.

## 2026-07-28 — prompt bundle schema 1.2

Bundle SHA-256:
`3d3fc7fe0f49e7e54dbff7df01ca48822568dc9161079db2f7117f95160bd180`

- `coverage_extraction` 2.0
  (`11cc6e49e7f5c1dfc22bfa16812d8a634cdf2d59ccd9db76161b856e1d29c2df`):
  requests exhaustive, exact-span extraction, compound splitting, abnormal
  measurements, and explicit non-inference. The model returns verbatim
  phrases; RAG-HPO calculates offsets locally. This design replaced a numeric
  offset output schema after Groq strict mode rejected it on longer notes.
- `coverage_audit` 1.0
  (`fc9476bb4dc617c1c85e185e94c0058c01b8380afbdb5b630416f7562d6464c3`):
  audits only sentences selected by deterministic coverage cues and returns
  only missed exact phrases.
- `batch_mapping_verification` 1.0
  (`8e2157812e07b6a31f7165e6b71ce8e3ab8b741e02fc690b7b36dd854fdde724`):
  maps bounded sentence-local mention batches to supplied candidates, requires
  explicit evidence for descendants, and permits ambiguity or rejection
  without inventing an ID.

The first complete live Case 1 smoke produced 5 TP, 8 FP, and 3 FN under
strict accepted-only scoring (precision 0.385, recall 0.625, F1 0.476).
This is an end-to-end staged-pipeline observation, not an isolated causal
estimate for any one prompt. The 30-case discovery and independent evaluations
are recorded separately once complete. No case ID, reference answer, or
case-specific phrase appears in these prompts.

## 2026-07-29 — prompt bundle schema 1.3

Bundle SHA-256:
`81cecb3a41195fc233800509095fc8d871e702d71a687c8a3c06d851db85d212`

- `context_mapping_zero_shot` 1.0
  (`1199e30407d501da82731f8499fef9d11d7d33236ee8237937364b03417961cc`):
  adds a deterministic patient/relative subject, assertion, section, and
  selectively bounded adjacent-sentence context to the existing batched
  mapping decision. It adds no model call.
- `context_mapping_one_shot` 1.0
  (`1cf80e24c7e0ab3eaad8f54b61fb088cdb4d41382917fbc50a642c928cbb5a80`):
  uses the same context with one fixed synthetic reduced-height example. It
  contains no benchmark case or manual answer.

Zero-shot remains the staged-mode default. One-shot is an explicit discovery
ablation selected with `--mapping-prompt one-shot`; it must be frozen before
confirmation use. No live inference had been performed when the prompt was
introduced; the later screening evidence is recorded below.

### 2026-07-29 screening evidence

The controlled 10-case mapping-only screen held extraction and candidates
fixed. Zero-shot reached precision 0.648, recall 0.603, and F0.5 0.639.
One-shot reached precision 0.656, recall 0.615, and F0.5 0.648. Both improved
the reused staged baseline, but neither met the predeclared 0.70 precision
eligibility floor. Therefore one-shot was not promoted and no prompt was
advanced to repeated confirmation.

## 2026-07-29 — prompt bundle schema 1.4

Bundle SHA-256:
`d3d1f332552e139f6624283468875bdf29bac90e44ef98f4db1152a52b5978bf`

- `phenotype_extraction` 1.0 text was constrained to the three public
  categories (`4b02c01a4f11604465ccb3ffaef004692f6e5cdd81f468b84f84d07f872a0895`).
  Suspected, historical, resolved, and hypothetical patient findings are
  abnormal with separate assertion metadata; irrelevant text is omitted.
- `coverage_extraction` 3.0
  (`2e53ca90d4ca0ade6ada5c13480e8ba170cbba51c2677bdde4e0421dfe4a0025`)
  and `coverage_audit` 2.0
  (`f564a0a372d55d352b01d51ddf99d1c0128d70ff09c9a1062e83001b0d6f8cf5`)
  now return spans only. Category is assigned once after recognizer and model
  spans merge. Telegraphic headings, lists, shared modifiers, measurements,
  and coordinated structures are explicit coverage cues.
- `context_mapping_zero_shot` 2.0
  (`8e3008a689ac6c1b1d98ebacb80af8ff76f6739a99b74de2d0299d7d2bc69b0b`)
  and `context_mapping_one_shot` 2.0
  (`d2698147b8098b5d86cde6bf7c2ae3c133aeb88fd4fd08c2f630b92314780826`)
  return zero to three supplied IDs. Ambiguous sets must be clinically
  plausible and unsupported decisions retain no ID.
- `final_categorization` 1.0
  (`24fb3fc63c145d4a3fd58eb99e377c50338e258e8ae0f42ddcccb7a7095fbe71`)
  makes one authoritative whole-note decision per routed span: Abnormal,
  Normal, or Family History. It adds one consolidated request per note in
  staged model modes.

No benchmark case, answer, or case-specific rule was added. The compatibility
pipeline remains the default until this revised schema is evaluated on a
locked subset. A short synthetic live schema smoke was run once, then repeated
only after it exposed overlapping-span and assertion-provenance defects; the
corrected run required four requests and 8,766 total provider tokens.

The subsequent fixed 30-case CSC discovery evaluation produced strict
candidate-set precision/recall/F1 of 0.6913/0.5198/0.5934 under the qualitative
accepted policy and 0.6800/0.6071/0.6415 when every mapped candidate was
included. A point-floor calibration reached 0.7011/0.6052/0.6496. Because the
candidate ceiling did not reach 0.70 recall, schema 1.4 was not promoted and
no GSC provider run was performed.

## 2026-07-29 — pipeline 3.2 (no prompt-text change)

The stage-loss remediation changed deterministic routing, span alignment,
clinical-modifier handling, and short-phrase retrieval fusion. Prompt bundle
schema 1.4 and all prompt texts/hashes remain unchanged. This distinction is
intentional: the paired five-case smoke measures a machine-pipeline change,
not a prompt rewrite.

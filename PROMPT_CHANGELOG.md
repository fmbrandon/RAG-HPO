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
confirmation use. No new live inference was performed for this prompt change.

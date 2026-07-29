# RAG-HPO

RAG-HPO extracts phenotype mentions from clinical text and maps abnormal
findings to Human Phenotype Ontology (HPO) terms. Version 0.2 provides an
installable package and four commands: `doctor`, `demo`, `vectorize`, and
`annotate`.

RAG-HPO sends note text to the configured language-model endpoint. Never submit
identifiable or restricted data unless your institution has approved both the
endpoint and workflow. See [SECURITY.md](SECURITY.md).

## Five-minute macOS quick start

The validated development baseline is an Apple Silicon Mac with Python 3.12.
The prebuilt SapBERT vector download is approximately 150 MB; allow at least
500 MB of free disk space for the bundle, environment, and outputs. Initial
setup usually takes several minutes depending on network speed.

```bash
git clone https://github.com/PoseyPod/RAG-HPO.git
cd RAG-HPO
python3.12 setup_environment.py
source .venv/bin/activate
```

FastHPOCR is not required for the current RAG-HPO workflow. Researchers who
want the optional local recognizer and benchmark adapter can install:

```bash
pip install -e '.[fasthpocr]'
```

Its current integration findings and limitations are documented in
[PHASE1_FASTHPOCR_FINDINGS.md](PHASE1_FASTHPOCR_FINDINGS.md) and
[PHASE2_REGISTRY_FINDINGS.md](PHASE2_REGISTRY_FINDINGS.md). Phase 3 confirmed
that the deterministic recognizer is a high-confidence aid, not a complete
annotator, because its recall is too low. The recall-preserving verified mode
improved precision and F0.5 but remains experimental until integration and
beginner-facing work are complete. See
[PHASE3_CASCADE_FINDINGS.md](PHASE3_CASCADE_FINDINGS.md).

Load the Groq key into this terminal without putting it in shell history or a
repository file:

```zsh
read -s "RAG_HPO_API_KEY?Paste Groq key: " && export RAG_HPO_API_KEY && echo
```

Then diagnose the setup and run the synthetic demonstration:

```bash
rag-hpo doctor --skip-provider
rag-hpo demo --vector-dir /path/to/prebuilt-vectors
```

Expected final messages resemble:

```text
Demo complete using a built-in synthetic note.
Mapped HPO IDs: HP:...
CSV results: rag_hpo_demo_output/rag_hpo_results.csv
JSON results: rag_hpo_demo_output/rag_hpo_results.json
```

The demo never uses the repository’s clinical examples. It uses a built-in
synthetic note, does not retain raw provider responses, and does not store the
API key.

After the lab owner publishes the approved v0.2.0 vector asset, `rag-hpo demo`
can download and cache it when these two release values are configured:

```bash
export RAG_HPO_VECTOR_BUNDLE_URL='https://release.example/vectors.zip'
export RAG_HPO_VECTOR_BUNDLE_SHA256='64-character-release-sha256'
rag-hpo demo
```

The archive SHA-256 and internal artifact manifest are both validated before
use. The cache is versioned under `~/.cache/rag-hpo/` on macOS/Linux and the
user’s local application-data directory on Windows. Until the lab release is
approved, pass `--vector-dir` or build vectors locally.

## Supported platforms

| Platform | Status |
| --- | --- |
| macOS ARM, Python 3.12 | Validated |
| macOS ARM, Python 3.11 and 3.13 | Dependency locks validated; 3.12 remains the release gate |
| Linux, Python 3.11–3.13 | Experimental/best-effort; statically reviewed, not natively validated |
| Windows, Python 3.11–3.13 | Experimental/best-effort; statically reviewed, not natively validated |

Linux and Windows reports and native CI contributions are welcome. Static path
tests and dependency profiles do not constitute native validation.

On Windows, the expected activation command is:

```powershell
.\.venv\Scripts\activate
```

## Diagnose problems

```bash
rag-hpo doctor --vector-dir artifacts/hpo --output-dir rag_hpo_output
```

Each failed or warning check prints an exact corrective command. Add
`--skip-provider` when offline; add `--json` for machine-readable diagnostics.
The full provider check makes a minimal schema-valid request and verifies model
access.

The convenience provider defaults are:

```text
RAG_HPO_BASE_URL=https://api.groq.com/openai/v1/chat/completions
RAG_HPO_MODEL=openai/gpt-oss-120b
RAG_HPO_RESPONSE_MODE=strict
```

Override these variables or the matching command options for another approved
OpenAI-compatible provider. Response modes are `strict`, `json-object`, and
`prompt-only`; RAG-HPO never silently downgrades a mode.

## Build complete HPO vectors

The bootstrap installs vectorization support. A complete build downloads the
ontology and pinned SapBERT model, then creates roughly 48,000 phrase vectors.
It can require more than 1 GB of temporary/cache space and may take from minutes
to much longer depending on hardware and network access.

```bash
rag-hpo vectorize \
  --hpo-addons HPO_addons.csv \
  --output-dir artifacts/hpo
```

For a quick functional smoke only:

```bash
rag-hpo vectorize \
  --hpo-addons HPO_addons.csv \
  --output-dir artifacts/hpo-smoke \
  --limit 50
```

Do not use a `--limit` artifact for scientific evaluation. The complete command
creates:

- `hpo_meta.json`
- `hpo_embedded.npz`
- `hpo_manifest.json`
- `hpo_registry.json`
- `hpo_registry_manifest.json`
- `hpo_lexical.json`
- `hpo_lexical_manifest.json`
- a private cached `hp.obo` when the ontology is downloaded

The registry keeps official HPO concepts separate from RAG-HPO add-on phrases
and records alternate IDs, synonym scopes, hierarchy, status, and lexical
ambiguities. The manifests link the registry, lexical index, source hashes,
parser version, embedding model and immutable revision, normalization, dtype,
dimensions, counts, and payload hashes. Annotation rejects modified,
incomplete, or model-mismatched artifacts. Advanced options include
`--obo-file`, `--obo-url`, `--backend`, `--refresh`, and `--offline`. SapBERT
is the scientific default; FastEmbed is optional.

## Annotate data

Choose the annotation workflow explicitly:

- `--mode model` preserves the v0.2 compatibility pipeline and remains the
  default until the staged accuracy gates pass.
- `--mode balanced` runs native recognition, optional FastHPOCR recognition,
  two-pass model extraction, hybrid sparse/dense retrieval, and batched
  mapping/verification. It is experimental while validation is in progress.
- `--mode high-recall` uses 32 distinct candidates and retains plausible
  unresolved findings in the review queue.
- `--mode native --no-model` is deterministic and fully local.
- `--mode fasthpocr --no-model` adds the optional attributed FastHPOCR
  recognizer. Install `.[fasthpocr]` and supply `--fasthpocr-index` when the
  index is not inside the vector directory.

`--recognizer fasthpocr` is repeatable and lets balanced/high-recall users
choose whether FastHPOCR participates. Native recognition is always available.
Recognizer-only findings are candidates, not automatically accepted answers.

Staged model modes use `--mapping-prompt zero-shot` by default. This supplies a
compact context packet for each mention: the exact sentence, section,
patient-versus-relative subject, and assertion. Preceding/following sentences
are included only for deterministic ambiguity cues, which limits token use.
`--mapping-prompt one-shot` adds one fixed synthetic example for controlled
research comparisons; it never selects examples from benchmark data.

For a network-prohibited run, use `--offline --no-model`. An offline model run
must use an already-running loopback OpenAI-compatible endpoint, cached model
and ontology artifacts, and an explicitly configured response mode:

```bash
RAG_HPO_API_KEY=local \
rag-hpo annotate \
  --mode balanced \
  --offline \
  --base-url http://127.0.0.1:8000/v1/chat/completions \
  --model local-model \
  --text 'Synthetic example: fever.' \
  --vector-dir artifacts/hpo \
  --output-dir rag_hpo_output
```

`--offline` rejects remote model URLs and prevents embedding-model downloads.
It does not start or install a local language model.

### Manual text

```bash
rag-hpo annotate \
  --text 'Synthetic example: fever and short stature.' \
  --patient-id synthetic-1 \
  --vector-dir artifacts/hpo \
  --output-dir rag_hpo_output
```

### CSV

The required column is `clinical_note`. `patient_id` is optional and remains a
string. The legacy `Case` column is also accepted. A safe example is provided
at [samples/synthetic_input.csv](samples/synthetic_input.csv).

```bash
rag-hpo annotate \
  --input samples/synthetic_input.csv \
  --vector-dir artifacts/hpo \
  --output-dir rag_hpo_output
```

### Standard input

```bash
printf '%s' 'Synthetic example: fever.' | \
  rag-hpo annotate \
    --input - \
    --patient-id synthetic-stdin \
    --vector-dir artifacts/hpo \
    --output-dir rag_hpo_output
```

Every run writes `rag_hpo_results.csv` and `rag_hpo_results.json`. The stable
fields are `patient_id`, `phrase`, `category`, `hpo_id`, `hpo_term`,
`vector_score`, `mapping_status`, `error_code`, and `error_message`. Staged
modes append `evidence_start`, `evidence_end`, `assertion_status`,
`confidence`, `review_status`, `source_methods`, and `candidate_hpo_ids`.
Accepted findings are the primary automated output; `review` findings are
preserved for human resolution and `rejected` findings remain auditable.

Evidence text is omitted by default. `--include-evidence-text` displays a
privacy warning and includes the source sentence in result files. Staged runs
also write `rag_hpo_run_manifest.json` with hashes, configuration, elapsed
time, and provider token totals, but never the API key or complete note. The
machine-readable result contract is in
[samples/expected_result_schema.json](samples/expected_result_schema.json).

## Resume and privacy

RAG-HPO uses a private SQLite state database in the output directory. It stores
input hashes and required derived state, not the API key or original note.
Successful runs delete it automatically. Failed or partial runs retain it:

```bash
rag-hpo annotate ... --resume
```

Resume requires the same input hash, package version, and vector-manifest hash.
Use `--keep-state` only when needed.

Raw provider responses are disabled by default. `--raw-responses` displays a
privacy warning and stores owner-only files that may contain sensitive derived
text.

## Scientific benchmark

CSC and GSC are separate corpora. `Test_Cases.csv` corresponds to the CSC input
and must be scored only against
`benchmarks/references/csc_manual_annotations.csv`. Never join CSC and GSC only
by patient number.

Score a selected run with:

```bash
python benchmarks/run_benchmark.py \
  --predictions rag_hpo_output/rag_hpo_results.json \
  --input Test_Cases.csv \
  --references benchmarks/references/csc_manual_annotations.csv \
  --ontology artifacts/hpo/hp.obo \
  --prompt-file src/rag_hpo/data/system_prompts.json \
  --vector-manifest artifacts/hpo/hpo_manifest.json \
  --case-id 1 \
  --model openai/gpt-oss-120b \
  --prompt-version 1.0 \
  --accepted-only \
  --output-dir benchmark-results/case-1
```

The runner normalizes alternate HPO IDs, removes duplicate predicted IDs within
each patient, preserves Cases 67 and 68 as distinct records, and emits explicit
TP/FP/FN identifiers plus per-case, micro, and macro metrics. Routine live
evaluation uses fixed, performance-blind 30-case subsets of CSC and GSC,
stratified by note length and manual-reference count. The CSC subset excludes
the discovery cases. Selection IDs, seed, strata, and hashes are recorded
under `benchmarks/results/`; note text remains outside Git. A complete corpus
run requires an explicit reason and authorization because it costs
substantially more time and tokens.

Use `--accepted-only` for the primary strict score of staged output. Omitting
it intentionally scores every mapped row, including unresolved review
candidates, as a high-recall sensitivity analysis.

Historical Premium, CSC, and GSC workbook calculations remain separately
labeled. See [benchmarks/README.md](benchmarks/README.md).

The locked accuracy investigation explains the rebuild's higher aggregate
precision but lower recall and F1, including ontology-version, candidate
retrieval, extraction, mapping, and scoring effects. See
[RAG-HPO_ACCURACY_INVESTIGATION.md](RAG-HPO_ACCURACY_INVESTIGATION.md).
Phase 3's prompt additions and unchanged base prompts are recorded in
[PROMPT_CHANGELOG.md](PROMPT_CHANGELOG.md).
The frozen staged subset evaluation, including confidence intervals and the
decision not to promote `balanced`, is recorded in
[PHASE4_70_70_FINDINGS.md](PHASE4_70_70_FINDINGS.md).

## Notebooks and development

`RAG-HPO.ipynb` and `HPO_Vectorization.ipynb` are thin, network-free examples.
They import the package and contain no application implementation, keys,
hard-coded local paths, or unconditional downloads.

```bash
pip install -e '.[vectorize,notebook,test,dev,benchmark]'
ruff check .
ruff format --check .
mypy src/rag_hpo
pytest --cov=rag_hpo --cov-branch
bandit -c pyproject.toml -r src/rag_hpo
python -m build
```

The native macOS dependency locks and experimental-platform lock guidance are
documented in [requirements/README.md](requirements/README.md).

## Citation

If you use RAG-HPO, cite:

> Improving Automated Deep Phenotyping Through Large Language Models Using
> Retrieval-Augmented Generation. *Genome Medicine*.
> https://doi.org/10.1186/s13073-025-01521-w

Machine-readable metadata is in `CITATION.cff`. Contribution and security
guidance are in [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md).

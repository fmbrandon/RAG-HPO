# RAG-HPO

RAG-HPO extracts phenotype mentions from clinical text and maps abnormal
findings to Human Phenotype Ontology terms. Version 0.2 separates the runtime
from the notebooks and provides reproducible `doctor`, `vectorize`, and
`annotate` commands.

RAG-HPO sends note text to the configured language-model endpoint. Do not submit
identifiable or otherwise restricted data unless the endpoint and workflow have
been approved by your institution. See [SECURITY.md](SECURITY.md).

## Requirements

- CPython 3.11, 3.12, or 3.13; Python 3.12 is recommended.
- macOS, Linux, or Windows.
- A Groq API key or another approved OpenAI-compatible endpoint.
- Internet access for the initial HPO and embedding-model downloads, unless
  using prebuilt cached artifacts.

## Installation

```bash
git clone https://github.com/PoseyPod/RAG-HPO.git
cd RAG-HPO
python3.12 setup_environment.py
source .venv/bin/activate
```

On Windows, activate with:

```powershell
.\.venv\Scripts\activate
```

`pyproject.toml` is the dependency source of truth. The setup script creates
only `.venv`; it never installs or switches the system Python interpreter.

For development:

```bash
pip install -e '.[vectorize,notebook,test,dev,benchmark]'
```

## Provider configuration

The API key is always required and is never written by RAG-HPO:

```bash
export RAG_HPO_API_KEY='your-key' # pragma: allowlist secret
```

The convenience defaults are:

```text
RAG_HPO_BASE_URL=https://api.groq.com/openai/v1/chat/completions
RAG_HPO_MODEL=openai/gpt-oss-120b
RAG_HPO_RESPONSE_MODE=strict
```

Override these variables or the corresponding command options for another
approved OpenAI-compatible provider. Response modes are `strict`,
`json-object`, and `prompt-only`; RAG-HPO never silently downgrades modes.

## 1. Diagnose the environment

```bash
rag-hpo doctor --json
```

Once vector artifacts exist:

```bash
rag-hpo doctor \
  --vector-dir artifacts/hpo \
  --output-dir rag_hpo_output
```

Use `--skip-provider` to diagnose an offline environment without making a
minimal provider health request.

## 2. Build HPO vector artifacts

Install the vectorization extra if it was not installed by the bootstrap:

```bash
pip install -e '.[vectorize]'
```

Run a bounded smoke build:

```bash
rag-hpo vectorize \
  --hpo-addons HPO_addons.csv \
  --output-dir artifacts/hpo \
  --limit 50
```

Run the complete build by omitting `--limit`:

```bash
rag-hpo vectorize \
  --hpo-addons HPO_addons.csv \
  --output-dir artifacts/hpo
```

The command creates:

- `hpo_meta.json`
- `hpo_embedded.npz`
- `hpo_manifest.json`
- a private cached `hp.obo` when the ontology is downloaded

The manifest records the HPO and add-on hashes, parser version, embedding model
and immutable revision, normalization, dtype, dimensions, counts, and output
hashes. Annotation rejects mismatched or modified artifacts.

Options include `--obo-file`, `--obo-url`, `--backend`, `--refresh`,
`--offline`, and `--limit`. The default backend is the biomedical SapBERT model
used by the earlier notebook, pinned to an immutable model revision. FastEmbed
is available after installing `rag-hpo[fastembed]`.

## 3. Annotate notes

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
string; the repository’s legacy `Case` column is also accepted.

```bash
rag-hpo annotate \
  --input Test_Cases.csv \
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

Outputs are deterministic:

- `rag_hpo_results.csv`
- `rag_hpo_results.json`

Result fields are `patient_id`, `phrase`, `category`, `hpo_id`, `hpo_term`,
`vector_score`, `mapping_status`, `error_code`, and `error_message`.

## Resume and privacy behavior

RAG-HPO creates a private SQLite state database in the output directory. It
stores input hashes and required derived state, not the API key or original
note. Successful runs delete it automatically. Failed or partial runs retain it
for:

```bash
rag-hpo annotate ... --resume
```

Resume requires identical input, package version, and vector-manifest hash.
Use `--keep-state` to preserve state after success.

Raw provider responses are disabled. `--raw-responses` enables them with a
warning and stores them with owner-only permissions; these files can contain
sensitive derived text.

## Notebooks

`RAG-HPO.ipynb` and `HPO_Vectorization.ipynb` are thin, network-free examples.
They import the package and contain no application implementation, keys,
hard-coded local paths, or automatic downloads.

## Testing and quality

```bash
ruff check .
ruff format --check .
mypy src/rag_hpo
pytest --cov=rag_hpo --cov-branch
bandit -c pyproject.toml -r src/rag_hpo
python -m build
```

Network and live-provider tests are opt-in. Normal tests use synthetic text and
mock endpoints.

## Benchmark data

The repository retains the published example and comparison data. Cases 67 and
68 have identical note text but remain distinct IDs; benchmark consumers must
not silently deduplicate them.

The workbook contains stored TP, FP, and FN counts. Recompute precision, recall,
and F1 into a tidy table with `benchmarks/recompute_metrics.py`. See
[benchmarks/README.md](benchmarks/README.md) for the reproducibility boundary.

## Citation

If you use RAG-HPO, cite:

> Improving Automated Deep Phenotyping Through Large Language Models Using
> Retrieval-Augmented Generation. *Genome Medicine*.
> https://doi.org/10.1186/s13073-025-01521-w

Machine-readable citation metadata is provided in `CITATION.cff`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please report security concerns
privately as described in [SECURITY.md](SECURITY.md). Dependency-license review
and the approved Pronto parser exception are documented in
[LICENSE_POLICY.md](LICENSE_POLICY.md).

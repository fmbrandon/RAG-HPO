# RAG-HPO: Next-Generation Clinical Phenotype Extraction & Ontology Mapping Engine

RAG-HPO extracts phenotype mentions from clinical text and maps abnormal findings to Human Phenotype Ontology (HPO) terms using a high-precision **Hybrid LLM + Algorithmic Ensemble Extractor** and **Retrieval-Augmented Generation (RAG)** vector engine. Version 0.2 provides an installable package and four commands: `doctor`, `demo`, `vectorize`, and `annotate`.

> [!IMPORTANT]
> **Data Security Warning:** RAG-HPO sends note text to the configured language-model endpoint (e.g. Groq, local vLLM, or OpenAI-compatible endpoints). Never submit identifiable or restricted patient data unless your institution has approved both the endpoint and workflow. See [SECURITY.md](SECURITY.md).

---

## 🚀 Remediation Progress & Benchmark Milestone Summary

The codebase has undergone a **6-Stage Progressive Remediation Plan** to eliminate false positives, eliminate incidental evidence-span mappings, and restore scientific recall:

| Stage | Focus & Architecture | Status | Primary Benchmark Achievement |
| :--- | :--- | :---: | :--- |
| **Stage 1** | **Hybrid LLM + Algorithmic Ensemble Extractor ("Best of Both")** | **COMPLETED** | Few-Shot ICL ($P_{\text{LLM}} \cup P_{\text{FastHPO}}$); Micro Recall increased from 0.3333 to **0.5556 (+66.7% gain)**; F1 = **0.5882**. |
| **Stage 2** | **Quick-Win Precision Engine & Head Phrase Recovery** | **COMPLETED** | Defined `SINGLE_TOKEN_MODIFIER_BLOCKLIST = frozenset(...)` and guarded head-phrase variant recovery (`"severe pain"` $\rightarrow$ `"pain"` variant with full span preservation). **Precision = 0.7143; Recall = 0.5556; F1 = 0.6250**. |
| **Stage 3** | **Acronym Normalization & Obsolete Term Resolution** | **COMPLETED** | Implemented `HPOTermRegistry` in `src/rag_hpo/registry.py` with dynamic HPO graph acronym parsing, recursive cycle-protected obsolete resolution (`replaced_by` $\rightarrow$ `consider`), and external UMLS/ADAM JSON ingestion (**185/185 unit tests passed**). |
| **Stage 4** | **SapBERT Embeddings & Token-Overlap Hybrid Retrieval** | **COMPLETED** | Integrated token-overlap hybrid retrieval in `src/rag_hpo/retrieval.py` alongside `SapBERT` embeddings. **Micro Precision = 0.7500 (75.0%), False Positives = 5 (-44.4% reduction), Case 15 Precision = 1.0000 (100%), Micro F1 = 0.6383 (+5.01% pts absolute gain)**. |
| **Stage 5** | **Lexical Rescue Architecture for HPO Mapping** | *In Progress* | Morphological de-adjectivalization retry, adjectival-noun alignment fallback, and `provisional` human review queue. |
| **Stage 6** | **Cross-Encoder Reranking & PhEval Hierarchy Evaluation** | *Planned* | `bge-reranker-base` top-64 reranking and PhEval Hierarchical Graph $F_1$ evaluation. |

---

## 🏛️ Repository Architecture & File Delineation

This workspace is structured with strict separation between **production application files**, **original/legacy artifacts**, and **development & testing tools**:

| Category | Directories & Files | Description |
| :--- | :--- | :--- |
| **🟢 Production App** *(Updated Application)* | • [`src/rag_hpo/`](src/rag_hpo)<br>• [`system_prompts.json`](src/rag_hpo/data/system_prompts.json)<br>• [`HPO_addons.csv`](HPO_addons.csv)<br>• [`RAG-HPO.ipynb`](RAG-HPO.ipynb)<br>• [`pyproject.toml`](pyproject.toml) | Core production code, installed `rag-hpo` package (`doctor`, `demo`, `vectorize`, `annotate`), runtime system prompts, ontology expansion data, user notebook, and package dependencies. |
| **🟡 Original App** *(Legacy Content)* | • [`legacy_original_app/`](legacy_original_app/) | Isolated folder containing historical artifacts: original Excel analysis workbook, presentation poster, original vectorization notebook, and historical baseline retrieval scripts. |
| **🔵 Dev & Build Tools** *(Engineering & Testing)* | • [`tests/`](tests/)<br>• [`benchmarks/`](benchmarks/)<br>• [`scripts/`](scripts/)<br>• [`sample_inputs/`](sample_inputs/) | Pytest automated test suite (185 tests), benchmark evaluation framework & reference datasets, release build & compilation scripts, synthetic test inputs, research remediation logs, and CI pin files. |

---

## ⚡ Five-minute macOS quick start

The validated development baseline is an Apple Silicon Mac with Python 3.12.
Running `setup_environment.py` automatically creates the virtual environment, installs dependencies, extracts/preloads the Hugging Face SapBERT embedding model, and executes an initial vectorization run to generate the HPO vector database into `artifacts/hpo` and user cache root.

```bash
git clone https://github.com/fmbrandon/RAG-HPO-rebuild.git
cd RAG-HPO
python3.12 setup_environment.py
source .venv/bin/activate
```

FastHPOCR is integrated as a optional local recognizer and benchmark adapter:

```bash
pip install -e '.[fasthpocr]'
```

Load your Groq API key into this terminal without putting it in shell history or a repository file:

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

---

## 💻 Supported platforms

| Platform | Status |
| --- | --- |
| macOS ARM, Python 3.12 | **Validated & Primary Release Gate** |
| macOS ARM, Python 3.11 and 3.13 | Dependency locks validated |
| Linux, Python 3.11–3.13 | Experimental/best-effort |
| Windows, Python 3.11–3.13 | Experimental/best-effort |

On Windows, the expected activation command is:

```powershell
.\.venv\Scripts\activate
```

---

## 🔍 Diagnose problems

```bash
rag-hpo doctor --vector-dir artifacts/hpo --output-dir rag_hpo_output
```

Each failed or warning check prints an exact corrective command. Add `--skip-provider` when offline; add `--json` for machine-readable diagnostics.

The convenience provider defaults are:

```text
RAG_HPO_BASE_URL=https://api.groq.com/openai/v1/chat/completions
RAG_HPO_MODEL=openai/gpt-oss-120b
RAG_HPO_RESPONSE_MODE=json-object
```

---

## 🧬 Build complete HPO vectors

The bootstrap installs vectorization support. A complete build downloads the ontology and pinned SapBERT model, then creates roughly 48,000 phrase vectors.

```bash
rag-hpo vectorize \
  --hpo-addons HPO_addons.csv \
  --output-dir artifacts/hpo
```

The complete vector directory creates:
- `hpo_meta.json`
- `hpo_embedded.npz`
- `hpo_manifest.json`
- `hpo_registry.json`
- `hpo_registry_manifest.json`
- `hpo_lexical.json`
- `hpo_lexical_manifest.json`

---

## 🏷️ Annotate clinical data

Choose the annotation workflow explicitly:

- `--mode balanced`: Runs native recognition, optional FastHPOCR recognition, two-pass LLM extraction ($P_{\text{LLM}} \cup P_{\text{FastHPO}}$), single-token modifier filtering, head-phrase variant expansion, hybrid sparse/dense SapBERT retrieval, and batched mapping/verification.

```bash
rag-hpo annotate \
  --mode balanced \
  --input sample_inputs/synthetic_input.csv \
  --vector-dir /path/to/full-sapbert-first \
  --output-dir rag_hpo_output
```

### Manual text entry

```bash
rag-hpo annotate \
  --text 'Synthetic example: severe pain and short stature.' \
  --patient-id synthetic-1 \
  --vector-dir /path/to/full-sapbert-first \
  --output-dir rag_hpo_output
```

---

## 🔬 Scientific benchmark & testing

Run the automated PyTest unit test suite (185 tests):

```bash
.venv/bin/pytest tests/
```

Run the 3-case empirical evaluation audit:

```bash
.venv/bin/python scratch/run_stage1_audit.py
```

Run corpus evaluation on CSC / GSC datasets:

```bash
python benchmarks/run_corpus_evaluation.py \
  --corpus csc \
  --vector-dir /path/to/full-sapbert-artifacts \
  --ontology /path/to/hp.obo \
  --output-dir /path/to/full-csc-results \
  --all \
  --confirm-external-transmission
```

---

## 📜 Citation

If you use RAG-HPO, cite:

> Improving Automated Deep Phenotyping Through Large Language Models Using Retrieval-Augmented Generation. *Genome Medicine*. https://doi.org/10.1186/s13073-025-01521-w

Machine-readable metadata is in `CITATION.cff`. Contribution and security guidance are in [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

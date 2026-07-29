# Workspace Architecture & File Delineation Guide

This workspace is explicitly organized into three distinct categories to separate the **updated production application**, the **original legacy application**, and the **development/building tools**.

---

## 🟢 1. Updated Application (Production / End-User)

These are the essential files needed to run, install, and interact with the updated RAG-HPO v0.2.0 application:

* **[`src/rag_hpo/`](src/rag_hpo)**: Production Python package containing all runtime modules:
  * `cli.py`: CLI commands (`doctor`, `demo`, `vectorize`, `annotate`).
  * `staged_pipeline.py` & `pipeline.py`: Production annotation pipeline engines.
  * `retrieval.py` & `registry.py`: Candidate retrieval, vector searching, and terminology mapping.
  * `provider.py`: Language-model provider abstraction (Groq, OpenAI compatible).
  * `cascade.py` & `hybrid.py`: Deterministic confidence lane & hybrid verification logic.
* **[`system_prompts.json`](system_prompts.json)**: Production LLM prompt templates.
* **[`HPO_addons.csv`](HPO_addons.csv)**: Required supplementary ontology expansion terms.
* **[`RAG-HPO.ipynb`](RAG-HPO.ipynb)**: End-user interactive demonstration notebook.
* **[`pyproject.toml`](pyproject.toml)** & **[`requirements.txt`](requirements.txt)**: Package installation configuration and dependency manifests.
* **[`setup_environment.py`](setup_environment.py)**: Automated environment bootstrap script.
* **Core Documentation**: [`README.md`](README.md), [`LICENSE`](LICENSE), [`SECURITY.md`](SECURITY.md), [`RELEASE.md`](RELEASE.md).

---

## 🟡 2. Original App (Legacy Content)

All files from the original baseline application and historical benchmark runs are isolated within the **[`legacy_original_app/`](legacy_original_app/)** directory:

* **[`legacy_original_app/RAG-HPO Tests and Data Analysis copy.xlsx`](legacy_original_app/RAG-HPO%20Tests%20and%20Data%20Analysis%20copy.xlsx)**: Published historical analysis workbook.
* **[`legacy_original_app/RAG-HPO ASHG Poster.pptx`](legacy_original_app/RAG-HPO%20ASHG%20Poster.pptx)**: Original conference presentation poster.
* **[`legacy_original_app/HPO_Vectorization.ipynb`](legacy_original_app/HPO_Vectorization.ipynb)**: Legacy vectorization notebook.
* **[`legacy_original_app/benchmarks/`](legacy_original_app/benchmarks)**: Legacy retrieval reconstruction scripts (`build_legacy_retrieval_artifact.py`, `summarize_historical_metrics.py`) and historical profile JSONs (`legacy-vector-corpus-profile.json`).

---

## 🔵 3. Development, Build & Benchmark Tools (Building & Testing)

Tools, tests, and data used by developers to build, calibrate, and validate the updated application:

* **[`tests/`](tests/)**: Automated unit, integration, and contract test suite executed via `pytest`.
* **[`benchmarks/`](benchmarks/)**: Evaluation framework, reference manual annotations (`csc_manual_annotations.csv`, `gsc_manual_annotations.csv`), diagnostic scripts, and ablation runners.
* **[`scripts/`](scripts/)**: Development utilities and build scripts (`build_release_artifacts.sh`, `build_vector_bundle.py`, `compile_lock.py`, `check_coverage_targets.py`).
* **[`sample_inputs/`](sample_inputs/)**: Synthetic sample input CSVs (`synthetic_input.csv`) and schemas (`expected_result_schema.json`) for testing.
* **[`reports_and_logs/`](reports_and_logs/)**: Research investigation reports, phase-by-phase findings, remediation logs, and validation records.
* **[`requirements/`](requirements/)**: Environment pin files (`dev.txt`, `macos-py312.txt`, `fasthpocr.txt`).

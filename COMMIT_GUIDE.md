# RAG-HPO Developer Guide: Commits, CI Pipeline Gates, and Quality Enforcement

This guide explains the architecture of RAG-HPO's **Automated Quality & Safety Pipeline**, why CI check failures occur during `git push`, and how to run pre-commit checks locally to guarantee first-time CI success.

---

## 🏛️ Why RAG-HPO Has Unique CI Enforcement

RAG-HPO is designed for clinical phenotype extraction and ontology mapping. Because the core pipeline handles structured data models, privacy state, and medical vector artifacts, the repository enforces **clinical safety-critical engineering standards**.

Unlike standard Python projects that only check if unit tests pass, RAG-HPO executes **14 mandatory quality and security gates** on every `git push` to `main` or `codex/**` branches.

---

## 🔒 The 4 Safety-Critical Modules & 100% Branch Coverage Target

The script [`scripts/check_coverage_targets.py`](scripts/check_coverage_targets.py) designates four core modules as **Safety-Critical**:

1. [`src/rag_hpo/artifacts.py`](src/rag_hpo/artifacts.py) (Vector artifact hash verification & manifests)
2. [`src/rag_hpo/config.py`](src/rag_hpo/config.py) (Provider configuration, URL HTTPS checks, and API key redaction)
3. [`src/rag_hpo/models.py`](src/rag_hpo/models.py) (Pydantic data schemas, LLM payload validators, and candidate sets)
4. [`src/rag_hpo/state.py`](src/rag_hpo/state.py) (SQLite private run state & persistence)

> [!CAUTION]
> **The 100% Coverage Gate:** `check_coverage_targets.py` requires **100.00% Branch Coverage** on these four modules.
> If any code edit adds an `if` condition, Pydantic `@model_validator`, or fallback `else` branch without a corresponding unit test in `tests/`, `check_coverage_targets.py` will abort CI with `exit code 2`.

---

## 📋 The 14 Mandatory CI Pipeline Gates

Whenever code is pushed to GitHub, GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) executes 14 checks:

| Step | Quality Gate | Command | Purpose |
| :---: | :--- | :--- | :--- |
| **1** | **Unit Test Suite** | `pytest --cov=rag_hpo` | Verifies all 185+ unit tests pass |
| **2** | **Safety Branch Coverage** | `python scripts/check_coverage_targets.py coverage.json` | Enforces **100% branch coverage** on `artifacts`, `config`, `models`, `state` |
| **3** | **Package Build** | `python -m build` | Validates PyPI wheel compilation |
| **4** | **Dependency Audit** | `python -m pip check` | Detects broken or missing package dependencies |
| **5** | **Code Linter** | `ruff check .` | Enforces code formatting & unused import rules |
| **6** | **Code Formatter** | `ruff format --check .` | Verifies PEP8 formatting compliance |
| **7** | **Notebook Linter** | `nbqa ruff RAG-HPO.ipynb` | Lints code cells inside Jupyter notebooks |
| **8** | **Static Type Checker** | `mypy src/rag_hpo` | Enforces strict static type annotations |
| **9** | **Security AST** | `bandit -c pyproject.toml -r src/rag_hpo` | Audits code for security flaws (`eval`, unsafe tempfiles) |
| **10**| **Secret Scanner** | `detect-secrets scan` | Prevents hardcoded API keys or credentials |
| **11**| **Package Vulnerabilities**| `pip-audit` | Audits dependencies against known CVE databases |
| **12**| **License Compliance** | `python scripts/check_licenses.py` | Verifies open-source license policy compliance |
| **13**| **Notebook Execution** | `jupyter nbconvert --execute RAG-HPO.ipynb` | Verifies main notebook runs top-to-bottom without errors |
| **14**| **Legacy Execution** | `jupyter nbconvert --execute HPO_Vectorization.ipynb` | Verifies legacy notebook compatibility |

---

## ⚡ Pre-Push Checklist: Guaranteed First-Time CI Success

To prevent CI failures on push, run these two steps locally before committing and pushing:

### Step 1: Run the 1-Line Pre-Push Validation Command

Run this command in your terminal inside the virtual environment:

```bash
.venv/bin/pytest --cov=rag_hpo --cov-branch --cov-report=json && \
.venv/bin/python scripts/check_coverage_targets.py coverage.json && \
.venv/bin/mypy src/rag_hpo && \
.venv/bin/ruff check .
```

If the command completes with:
```text
============================= 185 passed in 8.07s =============================
Safety-critical module branch coverage: 100%
All checks passed!
```
Your push is **100% guaranteed to pass CI on GitHub Actions**.

---

## 🔧 Troubleshooting Common CI Failures

### 1. `check_coverage_targets.py: error: coverage targets below 100%: src/rag_hpo/models.py: XX.XX%`
* **Cause:** A change in `models.py`, `config.py`, `artifacts.py`, or `state.py` introduced an un-tested branch.
* **Fix:** Find missing lines/branches in `coverage.json`:
  ```bash
  .venv/bin/python -c "import json; r=json.load(open('coverage.json')); print(r['files']['src/rag_hpo/models.py']['missing_branches'])"
  ```
  Add unit tests in `tests/test_config_models.py` (or matching test file) covering those specific input conditions.

### 2. `ruff check .` or `ruff format --check .` Failure
* **Cause:** Formatting or linting style mismatch.
* **Fix:** Run automatic local auto-fixer:
  ```bash
  .venv/bin/ruff format .
  .venv/bin/ruff check --fix .
  ```

### 3. `mypy src/rag_hpo` Failure
* **Cause:** Type annotation mismatch (e.g. returning `dict[str, Any]` when function signature specifies `dict[str, str]`).
* **Fix:** Update explicit type hints in the affected function signature or docstring.

---

## 📝 Commit Convention

Use clear, structured commit messages following Conventional Commits format:

* `feat(pipeline): ...` (New features or stage enhancements)
* `fix(ci): ...` (Fixing CI checks, coverage targets, or linting)
* `test(registry): ...` (Adding unit tests or benchmark verification)
* `docs(readme): ...` (Updating documentation or walkthroughs)

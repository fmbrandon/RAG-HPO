# Contributing

Use Python 3.12 for development:

```bash
python3.12 setup_environment.py
source .venv/bin/activate
pip install -e '.[vectorize,notebook,test,dev,benchmark]'
```

Before opening a pull request, run:

```bash
ruff check .
ruff format --check .
mypy src/rag_hpo
pytest --cov=rag_hpo --cov-branch
bandit -c pyproject.toml -r src/rag_hpo
detect-secrets scan
python -m build
```

Tests must use synthetic notes. Network and live-provider tests must be marked
and opt-in. Never commit credentials, raw provider responses, generated vectors,
model caches, or clinical outputs.

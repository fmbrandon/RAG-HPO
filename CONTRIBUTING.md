# Contributing

Use Python 3.12 for development:

```bash
python3.12 setup_environment.py
source .venv/bin/activate
pip install -e '.[vectorize,notebook,test,dev,benchmark]'
```

Before opening a pull request, run the canonical developer validation suite:

```bash
python scripts/validate.py
```

Tests must use synthetic notes. Network and live-provider tests must be marked
and opt-in. Never commit credentials, raw provider responses, generated vectors,
model caches, or clinical outputs.

macOS ARM with Python 3.12 is the release gate. Linux and Windows work must be
described as experimental until it has run natively. Generate target locks only
on that target with `scripts/compile_lock.py`; the script intentionally refuses
cross-platform lock labeling. The manual experimental workflow may be used by a
maintainer who has chosen to evaluate those platforms.

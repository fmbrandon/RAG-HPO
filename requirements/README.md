# Dependency locks

`macos-py311.txt`, `macos-py312.txt`, and `macos-py313.txt` were resolved and
installed on macOS ARM. The root `requirements.txt` is the Python 3.12 macOS
compatibility lock.

No Linux or Windows lock is checked in as validated. Generate one only on the
native target with, for example:

```bash
python scripts/compile_lock.py --platform linux --python 3.12
python scripts/compile_lock.py --platform linux --python 3.12 --profile cpu
python scripts/compile_lock.py --platform windows --python 3.12
```

The script refuses cross-platform labeling. Linux’s CPU profile selects the
PyTorch CPU wheel index. The Windows input explicitly accounts for
`win32-setctime` and `pywinpty`; native resolution remains required.

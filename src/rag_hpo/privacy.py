from __future__ import annotations

import os
from pathlib import Path


def ensure_private_directory(path: Path) -> None:
    """Create a directory and restrict it to its owner on POSIX systems."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        path.chmod(0o700)

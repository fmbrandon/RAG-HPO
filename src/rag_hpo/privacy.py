from __future__ import annotations

import os
from pathlib import Path


def ensure_private_directory(path: Path) -> None:
    """Create a directory and restrict it to its owner where chmod is meaningful."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    restrict_owner(path, directory=True)


def restrict_owner(path: Path, *, directory: bool = False) -> None:
    """Apply owner-only POSIX permissions; Windows inherits the user-directory ACL."""
    if os.name == "posix":
        path.chmod(0o700 if directory else 0o600)

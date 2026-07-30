from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import setup_environment


def test_build_parser_defaults() -> None:
    parser = setup_environment.build_parser()
    args = parser.parse_args([])
    assert args.vector_dir == setup_environment.DEFAULT_VECTOR_DIR
    assert args.backend == "sapbert"
    assert args.skip_vectorize is False


def test_setup_environment_version_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 10, 0))
    ret = setup_environment.main(["--skip-vectorize"])
    assert ret == 2


def test_setup_environment_skip_vectorize(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def mock_run(cmd: list[str]) -> None:
        calls.append(cmd)

    monkeypatch.setattr(setup_environment, "run", mock_run)
    monkeypatch.setattr(Path, "exists", lambda self: True)

    ret = setup_environment.main(["--skip-vectorize"])
    assert ret == 0
    # Should have run pip upgrade, pip install, pip check, but no vectorize step
    assert any("pip" in c and "--upgrade" in c for c in calls)
    assert not any("rag_hpo.cli" in c for c in calls)

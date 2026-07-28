from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts/check_licenses.py"
    spec = importlib.util.spec_from_file_location("check_licenses", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_license_policy_allows_reviewed_chardet_only() -> None:
    module = _module()
    assert (
        module.violations(
            [
                {"Name": "chardet", "License": "LGPLv2+"},
                {"Name": "safe", "License": "MIT"},
            ]
        )
        == []
    )
    assert module.violations([{"Name": "unexpected", "License": "GPL-3.0"}]) == [
        {"Name": "unexpected", "License": "GPL-3.0"}
    ]

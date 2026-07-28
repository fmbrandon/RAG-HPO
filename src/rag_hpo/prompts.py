from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any, cast


@lru_cache(maxsize=1)
def load_prompts() -> dict[str, Any]:
    resource = files("rag_hpo").joinpath("data/system_prompts.json")
    with resource.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    required = {"schema_version", "phenotype_extraction", "hpo_mapping"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"prompt data is missing keys: {sorted(missing)}")
    return cast(dict[str, Any], data)

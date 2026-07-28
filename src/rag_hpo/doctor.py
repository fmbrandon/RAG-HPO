from __future__ import annotations

import os
import platform
import sys
import tempfile
from pathlib import Path

from rag_hpo.artifacts import load_artifacts
from rag_hpo.config import ProviderConfig
from rag_hpo.models import DoctorCheck, DoctorReport
from rag_hpo.privacy import ensure_private_directory
from rag_hpo.provider import OpenAICompatibleProvider, ProviderError


def run_doctor(
    *,
    vector_dir: Path | None,
    output_dir: Path,
    check_provider: bool,
    base_url: str | None = None,
    model: str | None = None,
) -> DoctorReport:
    checks: list[DoctorCheck] = []
    version = sys.version_info[:2]
    checks.append(
        DoctorCheck(
            name="python",
            status="pass" if (3, 11) <= version < (3, 14) else "fail",
            detail=f"{platform.python_implementation()} {platform.python_version()}",
            action=(
                None
                if (3, 11) <= version < (3, 14)
                else "Install Python 3.12, then run: python3.12 setup_environment.py"
            ),
        )
    )

    try:
        ensure_private_directory(output_dir)
        with tempfile.NamedTemporaryFile(dir=output_dir):
            pass
        checks.append(
            DoctorCheck(
                name="output-directory",
                status="pass",
                detail=f"{output_dir} is writable and private",
            )
        )
    except OSError as exc:
        checks.append(
            DoctorCheck(
                name="output-directory",
                status="fail",
                detail=str(exc),
                action=(
                    f"Create a writable private directory and retry with --output-dir {output_dir}"
                ),
            )
        )

    if vector_dir is None:
        checks.append(
            DoctorCheck(
                name="vector-artifacts",
                status="warn",
                detail="no vector directory supplied",
                action=(
                    "Pass --vector-dir PATH, or build one with: rag-hpo vectorize "
                    "--output-dir vectors --hpo-addons HPO_addons.csv"
                ),
            )
        )
    else:
        try:
            entries, matrix, manifest = load_artifacts(vector_dir)
            checks.append(
                DoctorCheck(
                    name="vector-artifacts",
                    status="pass",
                    detail=(
                        f"{len(entries)} entries, {matrix.shape[1]} dimensions, "
                        f"{manifest.embedding_backend}:{manifest.embedding_model}"
                    ),
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            checks.append(
                DoctorCheck(
                    name="vector-artifacts",
                    status="fail",
                    detail=str(exc),
                    action=(
                        "Re-download the release bundle or rebuild it with: rag-hpo vectorize "
                        "--output-dir vectors --hpo-addons HPO_addons.csv --refresh"
                    ),
                )
            )

    if "RAG_HPO_API_KEY" not in os.environ:
        checks.append(
            DoctorCheck(
                name="provider-configuration",
                status="warn" if not check_provider else "fail",
                detail="RAG_HPO_API_KEY is not set",
                action=(
                    'Run: read -s "RAG_HPO_API_KEY?Paste Groq key: " && '
                    "export RAG_HPO_API_KEY && echo"
                ),
            )
        )
    else:
        try:
            config = ProviderConfig.from_env(base_url=base_url, model=model)
            checks.append(
                DoctorCheck(
                    name="provider-configuration",
                    status="pass",
                    detail=f"{config.model} at {config.base_url}",
                )
            )
            if check_provider:
                with OpenAICompatibleProvider(config) as provider:
                    provider.health_check()
                checks.append(
                    DoctorCheck(
                        name="provider-connectivity",
                        status="pass",
                        detail=f"model {config.model} returned a schema-valid response",
                    )
                )
        except (ValueError, ProviderError) as exc:
            checks.append(
                DoctorCheck(
                    name="provider-connectivity",
                    status="fail",
                    detail=str(exc),
                    action=(
                        "Verify the key and model with: rag-hpo doctor --skip-provider, "
                        "then retry rag-hpo doctor"
                    ),
                )
            )
    return DoctorReport(
        ok=not any(check.status == "fail" for check in checks),
        checks=checks,
    )

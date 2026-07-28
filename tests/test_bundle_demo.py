from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np

import rag_hpo.demo as demo_module
import rag_hpo.privacy as privacy_module
from rag_hpo import bundle as bundle_module
from rag_hpo.artifacts import ArtifactEntry, write_artifacts
from rag_hpo.bundle import default_cache_root, download_vector_bundle
from rag_hpo.models import AnnotationResult, Category, DoctorCheck, DoctorReport


def _bundle_bytes(tmp_path: Path) -> bytes:
    vectors = tmp_path / "vectors"
    write_artifacts(
        vectors,
        entries=[
            ArtifactEntry(
                hp_id="HP:1",
                phrase="Fever",
                term="Fever",
                source="test",
            )
        ],
        vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        hpo_source="test",
        hpo_sha256="a",
        addons_sha256=None,
        parser="test",
        parser_version="1",
        embedding_backend="fake",
        embedding_model="fake-model",
        embedding_revision="fake-revision",
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path in sorted(vectors.iterdir()):
            archive.write(path, path.name)
    return output.getvalue()


def test_platform_cache_paths_are_static_and_cross_platform() -> None:
    assert default_cache_root(
        platform_name="posix",
        environment={"XDG_CACHE_HOME": "/cache"},
        home=Path("/home/user"),
    ) == Path("/cache/rag-hpo/v0.2.0-sapbert")
    assert (
        default_cache_root(
            platform_name="nt",
            environment={"LOCALAPPDATA": r"C:\Users\Test\AppData\Local"},
            home=Path("/unused"),
        )
        == Path(r"C:\Users\Test\AppData\Local") / "rag-hpo/v0.2.0-sapbert"
    )


def test_windows_permission_branch_relies_on_inherited_acl(monkeypatch: Any) -> None:
    class FakePath:
        def chmod(self, _: int) -> None:
            raise AssertionError("chmod must not be called on the Windows branch")

    monkeypatch.setattr(privacy_module.os, "name", "nt")
    privacy_module.restrict_owner(FakePath())  # type: ignore[arg-type]


def test_bundle_download_verifies_archive_and_artifacts(tmp_path: Path) -> None:
    content = _bundle_bytes(tmp_path)

    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    cache = download_vector_bundle(
        url="https://example.test/bundle.zip",
        expected_sha256=hashlib.sha256(content).hexdigest(),
        cache_dir=tmp_path / "cache",
        client=client,
    )
    assert (cache / "hpo_manifest.json").is_file()


def test_bundle_download_rejects_insufficient_disk_space(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    content = _bundle_bytes(tmp_path)
    usage = SimpleNamespace(total=100, used=90, free=10)
    monkeypatch.setattr(bundle_module.shutil, "disk_usage", lambda _: usage)
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
    )
    try:
        download_vector_bundle(
            url="https://example.test/bundle.zip",
            expected_sha256=hashlib.sha256(content).hexdigest(),
            cache_dir=tmp_path / "cache",
            client=client,
        )
    except RuntimeError as exc:
        assert "500 MB" in str(exc)
    else:
        raise AssertionError("insufficient space must fail")


def test_demo_reports_actionable_doctor_failure(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.setattr(demo_module, "resolve_vector_dir", lambda _: tmp_path)
    monkeypatch.setattr(
        demo_module,
        "run_doctor",
        lambda **_: DoctorReport(
            ok=False,
            checks=[
                DoctorCheck(
                    name="provider-configuration",
                    status="fail",
                    detail="missing key",
                    action="export the key",
                )
            ],
        ),
    )
    assert demo_module.run_demo(vector_dir=None, output_dir=tmp_path, as_json=False) == 1
    output = capsys.readouterr().out
    assert "missing key" in output
    assert "export the key" in output


def test_demo_reports_missing_vectors_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    def unavailable(_: Path | None) -> Path:
        raise RuntimeError("build vectors with: rag-hpo vectorize")

    monkeypatch.setattr(demo_module, "resolve_vector_dir", unavailable)
    assert demo_module.run_demo(vector_dir=None, output_dir=tmp_path, as_json=True) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["error"] == "vectors_unavailable"
    assert "rag-hpo vectorize" in output["detail"]


def test_demo_success_uses_only_synthetic_input(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.setattr(demo_module, "resolve_vector_dir", lambda _: tmp_path)
    monkeypatch.setattr(
        demo_module,
        "run_doctor",
        lambda **_: DoctorReport(ok=True, checks=[]),
    )
    monkeypatch.setattr(demo_module.ProviderConfig, "from_env", lambda: object())

    class Provider:
        def __init__(self, _: object) -> None:
            pass

        def __enter__(self) -> Provider:
            return self

        def __exit__(self, *_: object) -> None:
            pass

    class Pipeline:
        def __init__(self, **_: object) -> None:
            pass

        def run(self, rows: list[Any]) -> list[AnnotationResult]:
            assert rows[0].patient_id == "synthetic-demo-1"
            assert "migraine" in rows[0].clinical_note
            return [
                AnnotationResult(
                    patient_id="synthetic-demo-1",
                    phrase="migraine headaches",
                    category=Category.ABNORMAL,
                    hpo_id="HP:0002076",
                    hpo_term="Migraine",
                    mapping_status="mapped",
                )
            ]

    monkeypatch.setattr(demo_module, "OpenAICompatibleProvider", Provider)
    monkeypatch.setattr(demo_module, "AnnotationPipeline", Pipeline)
    assert demo_module.run_demo(vector_dir=tmp_path, output_dir=tmp_path, as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["synthetic_input"] is True

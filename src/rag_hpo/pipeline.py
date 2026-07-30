from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
from pydantic import ValidationError

from rag_hpo import __version__
from rag_hpo.artifacts import ArtifactEntry, ArtifactManifest, load_artifacts, sha256_file
from rag_hpo.embeddings import EmbeddingBackend, create_backend
from rag_hpo.export import export_results
from rag_hpo.models import (
    AnnotationInput,
    AnnotationResult,
    Candidate,
    Category,
    HPOMapping,
    PhenotypeExtraction,
)
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.prompts import load_prompts
from rag_hpo.provider import OpenAICompatibleProvider, ProviderError
from rag_hpo.state import PipelineState


def load_csv_inputs(path: Path) -> tuple[list[AnnotationInput], list[AnnotationResult]]:
    valid: list[AnnotationInput] = []
    errors: list[AnnotationResult] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "clinical_note" not in reader.fieldnames:
            raise ValueError("input CSV requires a clinical_note column")
        for index, row in enumerate(reader, start=1):
            patient_id = row.get("patient_id") or row.get("Case") or str(index)
            try:
                valid.append(
                    AnnotationInput(
                        patient_id=str(patient_id),
                        clinical_note=row.get("clinical_note") or "",
                    )
                )
            except ValidationError as exc:
                errors.append(
                    AnnotationResult(
                        patient_id=str(patient_id).strip() or str(index),
                        phrase="",
                        category=None,
                        mapping_status="error",
                        error_code="invalid_input",
                        error_message=_short_error(exc),
                    )
                )
    return valid, errors


def hash_inputs(rows: list[AnnotationInput]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.patient_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(row.clinical_note.encode("utf-8")).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def _short_error(error: Exception) -> str:
    return str(error).replace("\n", " ")[:300]


class AnnotationPipeline:
    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        vector_dir: Path,
        output_dir: Path,
        resume: bool,
        keep_state: bool,
        keep_raw_responses: bool,
        top_k: int = 64,
        backend: EmbeddingBackend | None = None,
        offline: bool = False,
    ) -> None:
        self.provider = provider
        self.vector_dir = vector_dir
        self.output_dir = output_dir
        self.resume = resume
        self.keep_state = keep_state
        self.keep_raw_responses = keep_raw_responses
        self.top_k = top_k
        self.entries, matrix, self.manifest = load_artifacts(vector_dir)
        self.backend = backend or create_backend(
            self.manifest.embedding_backend,
            offline=offline,
        )
        self._validate_backend(self.manifest, self.backend)
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(np.asarray(matrix, dtype=np.float32))
        self.prompts = load_prompts()

    def run(
        self,
        rows: list[AnnotationInput],
        *,
        initial_errors: list[AnnotationResult] | None = None,
    ) -> list[AnnotationResult]:
        ensure_private_directory(self.output_dir)
        state_path = self.output_dir / ".rag-hpo-state.sqlite3"
        manifest_hash = sha256_file(self.vector_dir / "hpo_manifest.json")
        all_results = list(initial_errors or [])
        completed_successfully = False
        try:
            with PipelineState(
                state_path,
                input_sha256=hash_inputs(rows),
                artifact_sha256=manifest_hash,
                pipeline_version=__version__,
                resume=self.resume,
            ) as state:
                for row_index, row in enumerate(rows):
                    note_hash = hashlib.sha256(row.clinical_note.encode("utf-8")).hexdigest()
                    cached = state.completed(row_index, note_hash) if self.resume else None
                    if cached is not None:
                        all_results.extend(cached)
                        continue
                    try:
                        results = self._annotate_row(row, row_index)
                        state.save_complete(row_index, row.patient_id, note_hash, results)
                        all_results.extend(results)
                    except (ProviderError, ValueError, RuntimeError) as exc:
                        code = exc.code if isinstance(exc, ProviderError) else "row_failure"
                        message = _short_error(exc)
                        state.save_error(row_index, row.patient_id, note_hash, code, message)
                        all_results.append(
                            AnnotationResult(
                                patient_id=row.patient_id,
                                phrase="",
                                category=None,
                                mapping_status="error",
                                error_code=code,
                                error_message=message,
                            )
                        )
                export_results(all_results, self.output_dir)
                completed_successfully = not any(
                    result.mapping_status == "error" for result in all_results
                )
        finally:
            if completed_successfully and not self.keep_state:
                state_path.unlink(missing_ok=True)
                Path(f"{state_path}-wal").unlink(missing_ok=True)
                Path(f"{state_path}-shm").unlink(missing_ok=True)
        return all_results

    def _annotate_row(
        self,
        row: AnnotationInput,
        row_index: int,
    ) -> list[AnnotationResult]:
        extraction, extraction_raw = self.provider.request(
            system_message=self.prompts["phenotype_extraction"],
            user_message=row.clinical_note,
            response_model=PhenotypeExtraction,
        )
        self._write_raw(row_index, "extract", extraction_raw)
        results: list[AnnotationResult] = []
        for phenotype_index, phenotype in enumerate(extraction.phenotypes):
            if phenotype.category is not Category.ABNORMAL:
                results.append(
                    AnnotationResult(
                        patient_id=row.patient_id,
                        phrase=phenotype.phrase,
                        category=phenotype.category,
                        mapping_status="not_mapped_category",
                    )
                )
                continue

            candidates = self._candidates(phenotype.phrase)
            payload = json.dumps(
                {
                    "phrase": phenotype.phrase,
                    "original_context": row.clinical_note,
                    "candidates": [
                        candidate.model_dump(mode="json", exclude_none=True)
                        for candidate in candidates
                    ],
                },
                ensure_ascii=False,
            )
            mapping, mapping_raw = self.provider.request(
                system_message=self.prompts["hpo_mapping"],
                user_message=payload,
                response_model=HPOMapping,
                temperature=0.0,
            )
            self._write_raw(row_index, f"map-{phenotype_index}", mapping_raw)
            by_id = {candidate.hpo_id: candidate for candidate in candidates}
            selected = by_id.get(mapping.hpo_id or "")
            if mapping.hpo_id is not None and selected is None:
                raise ValueError("provider selected an HPO ID outside the supplied candidates")
            results.append(
                AnnotationResult(
                    patient_id=row.patient_id,
                    phrase=phenotype.phrase,
                    category=phenotype.category,
                    hpo_id=selected.hpo_id if selected else None,
                    hpo_term=selected.term if selected else None,
                    vector_score=selected.score if selected else None,
                    mapping_status="mapped" if selected else "no_candidate_fit",
                )
            )
        return results

    def _candidates(self, phrase: str) -> list[Candidate]:
        query = self.backend.encode([phrase])
        if query.ndim != 2 or query.shape[1] != self.manifest.dimension:
            raise ValueError("query embedding dimension does not match vector artifacts")
        scores, indices = self.index.search(np.asarray(query, dtype=np.float32), self.top_k)
        candidates: list[Candidate] = []
        seen_ids: set[str] = set()
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            entry: ArtifactEntry = self.entries[int(index)]
            if entry.hp_id in seen_ids:
                continue
            seen_ids.add(entry.hp_id)
            candidates.append(
                Candidate(
                    hpo_id=entry.hp_id,
                    term=entry.term,
                    score=round(float(score), 6),
                )
            )
        return candidates

    def _write_raw(self, row_index: int, stage: str, content: str) -> None:
        if not self.keep_raw_responses:
            return
        directory = self.output_dir / "raw_responses"
        ensure_private_directory(directory)
        path = directory / f"row-{row_index:05d}-{stage}.json"
        path.write_text(content, encoding="utf-8")
        restrict_owner(path)

    @staticmethod
    def _validate_backend(
        manifest: ArtifactManifest,
        backend: EmbeddingBackend,
    ) -> None:
        if backend.name != manifest.embedding_backend:
            raise ValueError("embedding backend does not match the vector manifest")
        if backend.model_id != manifest.embedding_model:
            raise ValueError("embedding model does not match the vector manifest")
        if backend.revision != manifest.embedding_revision:
            raise ValueError("embedding revision does not match the vector manifest")

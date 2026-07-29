#!/usr/bin/env python3
"""Replay only the mapping stage to compare fixed context prompts economically."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from rag_hpo.artifacts import sha256_file
from rag_hpo.assertion import analyze_assertion
from rag_hpo.config import ProviderConfig, ResponseMode
from rag_hpo.models import Candidate, Category, MappingSetDecisionBatch
from rag_hpo.privacy import ensure_private_directory, restrict_owner
from rag_hpo.prompts import load_prompts
from rag_hpo.provider import OpenAICompatibleProvider, ProviderError
from rag_hpo.registry import load_registry_bundle
from rag_hpo.staged_pipeline import (
    MAPPING_BATCH_SIZE,
    MappingPromptMode,
    Mention,
    build_context_packet,
)


def _selection_ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    selection = raw.get("selection", raw)
    values = selection.get("selected_case_ids")
    if not isinstance(values, list):
        raise ValueError("selection manifest requires selected_case_ids")
    return [str(value) for value in values]


def _notes(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, str] = {}
    for row in rows:
        patient_id = str(
            row.get("patient_id") or row.get("Case") or row.get("Patient ID") or ""
        ).strip()
        note = str(row.get("clinical_note") or "").strip()
        if patient_id and note:
            output[patient_id] = note
    return output


def _candidate(hpo_id: str, concepts: dict[str, Any]) -> Candidate | None:
    concept = concepts.get(hpo_id)
    if concept is None:
        return None
    return Candidate(
        hpo_id=hpo_id,
        term=concept.label,
        score=0.0,
        definition=concept.definition[:500] or None,
        synonyms=sorted({value.phrase for value in concept.phrases})[:12],
        parents=concept.parents,
        source_methods=["replayed-candidate"],
    )


def _request_batch(
    provider: OpenAICompatibleProvider,
    *,
    prompt: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        result, _ = provider.request(
            system_message=prompt,
            user_message=json.dumps({"items": items}, ensure_ascii=False, sort_keys=True),
            response_model=MappingSetDecisionBatch,
            temperature=0.0,
        )
    except ProviderError as exc:
        if exc.status_code != 400 or len(items) == 1:
            raise
        midpoint = len(items) // 2
        return {
            **_request_batch(provider, prompt=prompt, items=items[:midpoint]),
            **_request_batch(provider, prompt=prompt, items=items[midpoint:]),
        }
    decisions = {value.mention_id: value for value in result.decisions}
    expected = {str(value["mention_id"]) for value in items}
    if len(decisions) != len(result.decisions) or set(decisions) != expected:
        if len(items) > 1:
            midpoint = len(items) // 2
            return {
                **_request_batch(provider, prompt=prompt, items=items[:midpoint]),
                **_request_batch(provider, prompt=prompt, items=items[midpoint:]),
            }
        raise ValueError("mapping replay returned an incomplete decision")
    return decisions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hold extraction and candidates fixed while comparing context prompts."
    )
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument(
        "--mapping-prompt",
        choices=[value.value for value in MappingPromptMode],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    args = parser.parse_args()

    selected = set(_selection_ids(args.selection_manifest))
    notes = _notes(args.input)
    raw_rows = json.loads(args.base_predictions.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError("base predictions must be a JSON list")
    registry, registry_manifest, _ = load_registry_bundle(args.vector_dir)
    concepts = {value.hp_id: value for value in registry.concepts if not value.obsolete}
    mode = MappingPromptMode(args.mapping_prompt)
    prompt_key = (
        "context_mapping_one_shot"
        if mode is MappingPromptMode.ONE_SHOT
        else "context_mapping_zero_shot"
    )
    prompt = load_prompts()[prompt_key]
    config = ProviderConfig.from_env(
        base_url=args.base_url,
        model=args.model,
        response_mode=ResponseMode.STRICT,
    )

    by_patient: dict[str, list[dict[str, Any]]] = {}
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        patient_id = str(raw.get("patient_id") or "")
        if (
            patient_id not in selected
            or raw.get("category") != Category.ABNORMAL.value
            or not raw.get("candidate_hpo_ids")
        ):
            continue
        methods = set(raw.get("source_methods") or [])
        if "model-pass-1" not in methods and not {"native", "fasthpocr"} <= methods:
            # Balanced mode keeps these lower-confidence additions in review
            # without paying for mapping.
            continue
        by_patient.setdefault(patient_id, []).append(raw)

    started = time.perf_counter()
    output: list[dict[str, Any]] = []
    with OpenAICompatibleProvider(config) as provider:
        for patient_id in sorted(selected, key=lambda value: int(value)):
            note = notes.get(patient_id)
            if note is None:
                raise ValueError(f"selected patient {patient_id} is missing input text")
            source_rows = by_patient.get(patient_id, [])
            pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for index, raw in enumerate(source_rows):
                start = int(raw.get("evidence_start"))
                end = int(raw.get("evidence_end"))
                mention = Mention(
                    phrase=str(raw.get("phrase") or ""),
                    start=start,
                    end=end,
                    category=Category.ABNORMAL,
                    methods=set(raw.get("source_methods") or []),
                )
                assertion = analyze_assertion(note, start, end)
                candidates = [
                    candidate
                    for hpo_id in raw.get("candidate_hpo_ids") or []
                    if (candidate := _candidate(str(hpo_id), concepts)) is not None
                ]
                mention_id = f"{patient_id}-m{index:04d}"
                item = {
                    "mention_id": mention_id,
                    "phrase": mention.phrase,
                    "context": {
                        key: value
                        for key, value in build_context_packet(
                            note,
                            mention,
                            assertion,
                            candidates,
                        ).__dict__.items()
                        if value is not None
                    },
                    "source_methods": sorted(mention.methods),
                    "recognizer_hpo_ids": [],
                    "candidates": [
                        {
                            "hpo_id": value.hpo_id,
                            "term": value.term,
                            "definition": ((value.definition or "")[:160] if rank <= 4 else ""),
                            "synonyms": ((value.synonyms or [])[:3] if rank <= 4 else []),
                            "parents": ((value.parents or [])[:2] if rank <= 4 else []),
                        }
                        for rank, value in enumerate(candidates, start=1)
                    ],
                }
                pending.append((raw, item))

            decisions: dict[str, Any] = {}
            items = [value[1] for value in pending]
            for start in range(0, len(items), MAPPING_BATCH_SIZE):
                decisions.update(
                    _request_batch(
                        provider,
                        prompt=prompt,
                        items=items[start : start + MAPPING_BATCH_SIZE],
                    )
                )
            for raw, item in pending:
                decision = decisions[str(item["mention_id"])]
                supplied_ids = {str(value["hpo_id"]) for value in item["candidates"]}
                candidate_ids = [
                    value for value in decision.candidate_hpo_ids if value in supplied_ids
                ]
                selected_id = candidate_ids[0] if len(candidate_ids) == 1 else None
                accepted = (
                    "model-pass-1" in set(raw.get("source_methods") or [])
                    or (
                        decision.verdict == "supported"
                        and decision.confidence in {"high", "medium"}
                    )
                ) and bool(candidate_ids)
                revised = dict(raw)
                revised.update(
                    {
                        "hpo_id": selected_id,
                        "hpo_term": (
                            concepts[selected_id].label if selected_id is not None else None
                        ),
                        "candidate_hpo_ids": candidate_ids,
                        "vector_score": None,
                        "mapping_status": ("mapped" if candidate_ids else "no_candidate_fit"),
                        "mapping_verdict": decision.verdict,
                        "confidence": decision.confidence,
                        "review_status": (
                            "accepted"
                            if accepted
                            else (
                                "rejected"
                                if decision.verdict == "unsupported"
                                and decision.confidence == "high"
                                else "review"
                            )
                        ),
                    }
                )
                output.append(revised)
        usage = dict(provider.usage)

    ensure_private_directory(args.output_dir)
    output_path = args.output_dir / "rag_hpo_results.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    restrict_owner(output_path)
    manifest = {
        "schema_version": "1.0",
        "purpose": "mapping-only context prompt ablation",
        "mapping_prompt": mode.value,
        "selected_case_count": len(selected),
        "selected_case_ids_sha256": hashlib.sha256(
            json.dumps(sorted(selected), separators=(",", ":")).encode()
        ).hexdigest(),
        "base_predictions_sha256": sha256_file(args.base_predictions),
        "input_sha256": sha256_file(args.input),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "registry_sha256": registry_manifest.registry_sha256,
        "prompt_bundle_sha256": sha256_file(
            Path(__file__).parents[1] / "src/rag_hpo/data/system_prompts.json"
        ),
        "provider": config.redacted(),
        "provider_usage": usage,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "output_sha256": sha256_file(output_path),
    }
    manifest_path = args.output_dir / "mapping_ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    restrict_owner(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

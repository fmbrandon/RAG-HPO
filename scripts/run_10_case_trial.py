#!/usr/bin/env python3
"""Run 10 randomly selected CSC benchmark cases using Groq API."""

import csv
import json
import os
import time
from pathlib import Path

from rag_hpo.config import ProviderConfig, ResponseMode
from rag_hpo.models import AnnotationInput
from rag_hpo.provider import OpenAICompatibleProvider
from rag_hpo.staged_pipeline import AnnotationMode, MappingPromptMode, StagedAnnotationPipeline

# 10 Reproducibly Selected CSC Target Cases (Seed=42)
TARGET_CASES = ["2", "23", "28", "70", "75", "82", "101", "110", "111", "114"]


def _load_groq_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY") or os.environ.get("RAG_HPO_API_KEY")
    if key:
        return key.strip()

    env_paths = [
        Path.home() / ".continue" / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GROQ_API_KEY=") or line.startswith("RAG_HPO_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> None:
    print("=== 🚀 RUNNING 10-CASE CSC TRIAL BENCHMARK (openai/gpt-oss-120b) ===")
    print(f"Target Patient Cases: {TARGET_CASES}")

    api_key = _load_groq_key()
    if not api_key:
        print("\n❌ ERROR: GROQ_API_KEY environment variable is not set!")
        return

    # Load Clinical Notes from Test_Cases.csv
    test_cases_csv = Path("Test_Cases.csv")
    notes = {}
    with open(test_cases_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("Case") or row.get("Patient ID") or row.get("patient_id")).strip()
            text = str(row.get("Text") or row.get("clinical_note") or row.get("Text_Case")).strip()
            if cid in TARGET_CASES and text:
                notes[cid] = text

    print(f"✓ Loaded {len(notes)} clinical notes from {test_cases_csv.name}")

    # Provider & Pipeline Setup with openai/gpt-oss-120b
    config = ProviderConfig(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1/chat/completions",
        model="openai/gpt-oss-120b",
        response_mode=ResponseMode.PROMPT_ONLY,
        connect_timeout=30.0,
        read_timeout=90.0,
        max_attempts=5,
    )
    provider = OpenAICompatibleProvider(config)

    vector_dir = Path(
        "/Users/cameegarcia/Documents/RAG-HPO Setup/"
        "RAG-HPO_AUDIT_ARTIFACTS/phase2-registry/full-sapbert-first"
    )
    output_dir = Path("rag_hpo_output/10_case_trial")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = StagedAnnotationPipeline(
        provider=provider,
        vector_dir=vector_dir,
        output_dir=output_dir,
        mode=AnnotationMode.BALANCED,
        recognizers={"native"},
        fasthpocr_index=None,
        mapping_prompt=MappingPromptMode.ZERO_SHOT,
        resume=False,
        keep_state=False,
        keep_raw_responses=True,
        include_evidence_text=True,
        offline=False,
    )

    # Run Pipeline Annotations Case by Case with rate-limit pacing
    results = []
    print("\nStarting pipeline inference with rate-limit pacing...")
    for idx, cid in enumerate(TARGET_CASES, start=1):
        if cid in notes:
            print(f"[{idx:2d}/10] Processing Case {cid:3s}...", end="", flush=True)
            res = []
            for attempt in range(1, 6):
                res = pipeline.run([AnnotationInput(patient_id=cid, clinical_note=notes[cid])])
                # Check if inference succeeded (more than 1 result or not an error phrase)
                valid_res = [r for r in res if r.mapping_status != "error" and r.phrase]
                if valid_res:
                    res = valid_res
                    break
                backoff_sec = 5.0 * attempt
                print(
                    f" (Attempt {attempt} rate-limited, waiting {backoff_sec:.0f}s...)",
                    end="",
                    flush=True,
                )
                time.sleep(backoff_sec)
            results.extend(res)
            print(f" Done ({len(res)} mentions found).")
            time.sleep(4.0)  # Paced delay between cases to respect Groq API limits

    # Save Annotations Output JSON
    output_file = output_dir / "predictions.json"
    predictions_payload = [r.model_dump(mode="json") for r in results]
    output_file.write_text(json.dumps(predictions_payload, indent=2), encoding="utf-8")
    print(f"\n✓ Saved trial predictions to {output_file}")

    # Evaluate Against Gold Reference Annotations
    gold_csv = Path("benchmarks/references/csc_manual_annotations.csv")
    gold_dict = {}
    with open(gold_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("Patient ID") or row.get("patient_id") or row.get("Case")).strip()
            if cid in TARGET_CASES:
                gold_dict.setdefault(cid, set()).add(
                    str(row.get("hpo_term") or row.get("hpo_id")).strip()
                )

    predictions_by_case = {}
    for r in results:
        if (
            r.mapping_status == "mapped"
            and r.review_status in ["accepted", "provisional"]
            and r.hpo_id
        ):
            predictions_by_case.setdefault(r.patient_id, set()).add(r.hpo_id)

    print("\n" + "=" * 75)
    print(" 📊 10-CASE CSC BENCHMARK EVALUATION BREAKDOWN (openai/gpt-oss-120b)")
    print("=" * 75)

    total_tp, total_fp, total_fn = 0, 0, 0
    for cid in TARGET_CASES:
        pred_set = predictions_by_case.get(cid, set())
        gold_set = gold_dict.get(cid, set())
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        msg = (
            f"Case {cid:3s}: Preds={len(pred_set):2d}, Gold={len(gold_set):2d} | "
            f"TP={tp:2d}, FP={fp:2d}, FN={fn:2d} | P={p:.4f}, R={r:.4f}, F1={f1:.4f}"
        )
        print(msg)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    print("-" * 75)
    print(f"OVERALL 10-CASE MICRO-AGGREGATED: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"  • MICRO PRECISION = {micro_p:.4f} ({micro_p * 100:.2f}%)")
    print(f"  • MICRO RECALL    = {micro_r:.4f} ({micro_r * 100:.2f}%)")
    print(f"  • MICRO F1 SCORE  = {micro_f1:.4f} ({micro_f1 * 100:.2f}%)")

    # Save results summary JSON
    output_summary = {
        "model": "openai/gpt-oss-120b",
        "target_cases": TARGET_CASES,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
    }
    Path("rag_hpo_output/10_case_trial/summary.json").write_text(
        json.dumps(output_summary, indent=2)
    )


if __name__ == "__main__":
    main()

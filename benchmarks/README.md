# Benchmark reproduction

## Corpus boundary

CSC and GSC are distinct input/reference pairs:

- `CSC Input` / `Test_Cases.csv` → `CSC Manual Annotations`
- `GSC Input` → `GSC Manual Annotations`

Patient numbers are local to a corpus. Do not compare Test Case 1 with GSC Case
1 and do not join CSC to GSC only by patient number. Immutable CSV exports and
their source hashes are in [references/SOURCE.md](references/SOURCE.md).

Cases 67 and 68 intentionally remain separate CSC records even though their
note text is identical.

## Score current predictions

`run_benchmark.py` evaluates actual normalized HPO sets:

```bash
python benchmarks/run_benchmark.py \
  --predictions rag_hpo_output/rag_hpo_results.json \
  --input Test_Cases.csv \
  --references benchmarks/references/csc_manual_annotations.csv \
  --ontology /path/to/hp.obo \
  --prompt-file src/rag_hpo/data/system_prompts.json \
  --vector-manifest /path/to/hpo_manifest.json \
  --case-id 1 \
  --model openai/gpt-oss-120b \
  --prompt-version 1.0 \
  --output-dir benchmark-results/case-1
```

The runner:

- maps current and alternate HPO IDs through the supplied ontology;
- collapses duplicate predicted IDs only within each patient;
- writes the exact predicted, reference, TP, FP, and FN sets;
- writes per-case CSV plus micro/macro JSON metrics;
- records hashes for predictions, references, ontology, and the supplied vector
  manifest identity;
- produces byte-identical output for identical files and metadata.

Run Case 1 before any authorized subset. A complete live benchmark is opt-in:
repository examples are published but are not automatically approved for
external transmission.

## Historical workbook arithmetic

The published workbook stores TP, FP, and FN counts as static values.
`recompute_metrics.py` checks their arithmetic:

```bash
python benchmarks/recompute_metrics.py \
  "RAG-HPO Tests and Data Analysis copy.xlsx" \
  --output benchmark-results/historical-metrics.csv
```

This does not reconstruct historical predictions. Premium, CSC, and GSC metrics
remain separately labeled and are not interchangeable gold standards.

For CSC Case 1, the historical LLaMa 4-Scout row is TP 3, FP 8, FN 5, F1
0.315789. It is a historical baseline, not a required performance guarantee for
the new provider/model.

# Benchmark reproduction

The published workbook stores TP, FP, and FN counts as static values. Recompute
precision, recall, and F1 into a tidy table with:

```bash
python benchmarks/recompute_metrics.py \
  "RAG-HPO Tests and Data Analysis copy.xlsx" \
  --output benchmark-results/metrics.csv
```

This validates the arithmetic from the stored counts. It does not independently
reconstruct TP, FP, or FN from raw model responses because the repository does
not contain a complete machine-readable prediction set for every comparison.

Cases 67 and 68 in `Test_Cases.csv` intentionally remain distinct benchmark IDs
even though their clinical-note text is identical. Consumers must not silently
deduplicate them.

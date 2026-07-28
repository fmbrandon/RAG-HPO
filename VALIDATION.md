# v0.2.0 macOS validation record

Validated on Apple Silicon macOS with CPython 3.12.13. Linux and Windows have
not been executed and remain experimental.

## Complete SapBERT artifact

Two independent builds used the complete ontology, `HPO_addons.csv`, and the
pinned SapBERT revision
`ad34b857369884acb59a6e67f69008f533c8ca3e`.

| Property | Result |
| --- | --- |
| Ontology SHA-256 | `a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b` |
| Add-on SHA-256 | `f5a62a07ff66b4def2de5a6c86b90bf6c6ee6f846abd6f2e26ab0c01a9fe915c` |
| Metadata count | 48,444 |
| Vector count | 48,444 |
| Dimensions | 768 |
| Metadata SHA-256, both builds | `73f8436d30f5a8b31f0169386930cc871c49c82322b34d15798f18db996f881b` |
| Vector SHA-256, both builds | `3e13050c7e0287714083181a29855e5a50033366627805938a53b727ddc6dbb7` |

Both artifacts passed count, dimension, finite-value, hash, and model-identity
validation.

The two release-bundle builds were also byte-identical after canonicalizing the
non-scientific `created_at` field. The 139,934,193-byte ZIP SHA-256 is
`f6ae879e51a6125b668af472b24595c78aae91f6e5a28f2cc564dee22f970e53`.

## Live functional checks

`rag-hpo doctor` passed Python, private output-directory, complete artifact,
Groq configuration, and schema-valid model connectivity checks.

The built-in synthetic demo completed without row errors and mapped:

- migraine → `HP:0002076`
- polydipsia → `HP:0001959`
- obesity → `HP:0001513`

No raw provider response was retained.

## CSC Case 1

The authorized repository Case 1 was scored only against the eight IDs in
`CSC Manual Annotations`. It produced 17 extracted rows, eight mapped rows,
seven unique predicted IDs, and no row errors in 15.11 seconds.

| Run | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Groq `openai/gpt-oss-120b`, v0.2.0 pipeline | 5 | 2 | 3 | 0.7143 | 0.6250 | 0.6667 |
| Historical LLaMa 4-Scout CSC Case 1 | 3 | 8 | 5 | 0.2727 | 0.3750 | 0.3158 |

The historical row is a labeled comparison, not a performance requirement.
Exact IDs, hashes, and configuration are recorded in
`benchmarks/results/case-1-groq-gpt-oss-120b.json`.

An authorized Cases 1–3 subset then completed 69 extracted rows with no row
errors in 44.01 seconds. Exact-set micro metrics were TP 25, FP 8, FN 10,
precision 0.7576, recall 0.7143, and F1 0.7353; macro F1 was 0.7384. The live
model produced a slightly different Case 1 set inside the multi-case request,
which confirms that live inference can vary even though scoring the same stored
prediction file is byte-for-byte reproducible.

## Case 68 source repair

The lab owner authorized replacing the duplicated repository Case 68 note with
the distinct Case 68 note in the workbook-derived `csc_input.csv`. The workbook
also contains ten Case 68 manual HPO annotations. The synchronization is
reproducible and guarded by a regression test; Cases 67 and 68 are no longer
text duplicates.

## Paired 30-case accuracy comparison

An authorized live comparison used a fixed sample selected before inference:
Case 68 was required to validate the repaired input, and 29 of the remaining
eligible CSC cases were sampled without replacement with seed `20260728`.
Eligibility required a repository input, CSC manual annotations, and a
historical LLaMa 4-Scout score. The sample manifest records all 30 case IDs and
source hashes.

The first live pass produced one HTTP 400 placeholder for Case 53. An isolated
retry of the same unchanged note succeeded, so only that placeholder was
replaced. The complete prediction set contained all 30 patients and no error
rows.

| Measure | Groq `openai/gpt-oss-120b` rebuild | Historical LLaMa 4-Scout |
| --- | ---: | ---: |
| TP / FP / FN | 297 / 126 / 207 | 331 / 160 / 173 |
| Micro precision | 0.7021 | 0.6741 |
| Micro recall | 0.5893 | 0.6567 |
| Micro F1 | 0.6408 | 0.6653 |
| Macro F1 | 0.6391 | 0.6554 |

The mean paired per-case F1 difference (rebuild minus historical) was
`-0.0164`; its deterministic 95% bootstrap interval was `[-0.0797, 0.0488]`.
The one-sided paired sign-flip p-value for the rebuild being better was
`0.6858`. The rebuild won 10 cases, tied one, and lost 19. Under the
predeclared rule requiring a wholly positive interval and p < 0.05, the rebuild
does **not** have better accuracy scores. The observed difference is not
statistically significant, so this result also does not prove that the
historical model is superior.

Case 68 completed with TP 6, FP 8, FN 4, and F1 0.5000. Its historical LLaMa
4-Scout row is TP 9, FP 2, FN 1, and F1 0.8571.

Exact normalized HPO sets and provenance are in
`benchmarks/results/csc-sample-30-groq-gpt-oss-120b.json`. Paired statistics are
in `benchmarks/results/csc-sample-30-current-vs-llama4-scout.json`. Raw
predictions and note text remain outside the repository.

The rebuild is substantially stronger in installation, validation, artifact
integrity, privacy controls, resumability, and beginner-facing operation.
Those engineering improvements should not be described as an accuracy
improvement.

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

## Data discrepancy retained for review

Repository `Test_Cases.csv` contains identical notes for Cases 67 and 68. The
workbook’s `CSC Input` sheet contains a different Case 68 note. Neither source
was changed. A complete publication-comparable CSC benchmark requires the lab
owner to choose the intended Case 68 source.

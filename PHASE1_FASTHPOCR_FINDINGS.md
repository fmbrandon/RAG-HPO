# Phase 1 FastHPOCR findings

Phase 1 is complete. FastHPOCR can be included legally as an optional,
unmodified dependency under its MIT license, but its bundled linguistic
resources will not be copied into RAG-HPO. The current evidence supports using
FastHPOCR as a fast recognizer and agreement signal, not as a replacement for
RAG-HPO or as an unverified source of extra findings.

## Licensing and packaging

- Reviewed version: `FastHPOCR==0.1.4`.
- The source distribution contains an MIT license naming Tudor Groza.
- The PyPI metadata leaves `License`, `Home-page`, and `Requires-Dist` empty.
- FastHPOCR imports Pronto even though it does not declare it. The optional
  `rag-hpo[fasthpocr]` extra therefore declares Pronto explicitly.
- RAG-HPO does not vendor FastHPOCR code, its 6.3 MB morphological vocabulary,
  or its base-synonym resource.
- `pip check` passes and `pip-audit` reported no known vulnerabilities in the
  256-package rebuild environment.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[`fasthpocr-0.1.4.json`](benchmarks/provenance/fasthpocr-0.1.4.json).

## Historical result reproduction

The workbook arithmetic reproduced 1,730 rows byte-for-byte. The new export
and the earlier audit export have the same SHA-256:
`c86be5af92b66104cc5c1146db04d14e836e8e535ef34653a9f38e7254ab7192`.

On paired historical rows:

| Corpus | System | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| CSC, 112 cases | FastHPOCR | 0.529 | 0.454 | 0.489 |
| CSC, 112 cases | LLaMa 4-Scout | 0.681 | 0.701 | 0.691 |
| GSC, 114 cases | FastHPOCR | 0.743 | 0.688 | 0.715 |
| GSC, 114 cases | LLaMa 4-Scout | 0.615 | 0.774 | 0.686 |

CSC contains 116 input notes but only 112 cases with manual reference
findings. Cases 22, 25, 44, and 89 remain valid inputs, but accuracy reports
must identify them as unscored rather than treating every prediction as a
false positive.

## Current FastHPOCR results

All current runs used FastHPOCR 0.1.4, HPO `2026-06-23`, alternate-ID
normalization, and the confirmed rule that comma-delimited IDs are alternatives
for one finding.

| Corpus and configuration | TP | FP | FN | Precision | Recall | F1 | F0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CSC default | 822 | 739 | 967 | 0.527 | 0.459 | 0.491 | 0.512 |
| CSC longest match | 809 | 552 | 980 | 0.594 | 0.452 | 0.514 | 0.559 |
| CSC add-ons + longest | 819 | 555 | 970 | 0.596 | 0.458 | 0.518 | 0.562 |
| GSC default | 694 | 241 | 318 | 0.742 | 0.686 | 0.713 | 0.730 |
| GSC longest match | 681 | 108 | 331 | 0.863 | 0.673 | 0.756 | 0.817 |

Longest-match filtering improved paired per-case F1 on both corpora:

- CSC mean improvement `0.0265`, 95% CI `0.0200–0.0335`.
- GSC mean improvement `0.0499`, 95% CI `0.0393–0.0607`.
- Both two-sided sign-flip p-values were below `0.00005`.

The 3,079 add-on phrases added ten CSC true positives and three false positives
under longest matching. They produced no prediction changes on GSC. This is a
small, corpus-dependent gain, so add-ons should remain a provenance-tagged
extension rather than be merged silently into official HPO data.

## Complementarity with rebuilt RAG-HPO

On the same 112 CSC cases:

| System | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Rebuilt RAG-HPO | 0.705 | 0.608 | 0.653 |
| FastHPOCR add-ons + longest | 0.596 | 0.458 | 0.518 |
| Exact-ID intersection | 0.937 | 0.357 | 0.517 |
| Unverified union | 0.567 | 0.709 | 0.630 |

The two systems shared 638 true positives and only 43 false positives. RAG-HPO
alone supplied another 450 true positives and 413 false positives. FastHPOCR
alone supplied 181 true positives but 512 false positives.

Therefore:

- Exact-ID agreement is a strong high-confidence lane.
- FastHPOCR-only findings are not safe to accept automatically.
- A raw union raises recall but damages precision and lowers F1.
- The cascade should verify recognizer-specific findings and may abstain.

## Reproducibility and operational findings

- A full official-only index was 139 MB and took 267 seconds to build.
- Adding lab phrases increased it to 167 MB and 290 seconds.
- Once indexed, 114–116 cases were recognized in approximately 0.05–0.13
  seconds, excluding file I/O.
- Two identical full builds produced different index hashes.
- Most differences selected a different synonym label for the same HPO ID, but
  one CSC phrase changed from `HP:0002094` to `HP:0005957`. The corresponding
  metrics changed by one TP, one FP, and one FN.
- FastHPOCR/Pronto emitted an encoding-confidence warning. No replacement
  characters or common UTF-8 mojibake sequences were found in the indexes or
  predictions, but the warning remains recorded.

The upstream `hp.index` byte hash cannot be treated as sufficient
reproducibility evidence. RAG-HPO needs deterministic ambiguity resolution and
semantic output tests around the adapter.

## Phase 1 decision

FastHPOCR is approved for optional research integration with attribution. The
remaining work will:

1. create a canonical HPO registry and deterministic derived views;
2. adopt and independently test useful lexical techniques;
3. use agreement as a precision-first fast lane;
4. verify or abstain on disagreements;
5. preserve offline and local-model operation.

The machine-readable aggregate is
[`fasthpocr-phase1-summary.json`](benchmarks/results/fasthpocr-phase1-summary.json).
Note-level predictions, indexes, logs, and comparison details remain private
under `RAG-HPO_AUDIT_ARTIFACTS/phase1-fasthpocr/`.

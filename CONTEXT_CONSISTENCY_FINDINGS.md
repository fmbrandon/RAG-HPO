# Context, hierarchy, and consistency findings

Date: 2026-07-29

## Scope

This work added context-aware mapping and evaluation tools without performing
new provider inference, rebuilding vectors, or running a full corpus. The
hierarchy analysis reused the locked 30-case CSC and GSC staged predictions.
The consistency analysis reused the earlier predeclared 20-case, three-run CSC
repeat experiment. Raw notes and prediction rows remain outside Git.

## Hierarchy-aware sensitivity

Strict alternative-aware exact scoring remains primary.

| Corpus | Distance | Precision | Recall | F1 | Added one-to-one related matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| CSC | exact | 0.727 | 0.711 | 0.719 | 0 |
| CSC | 1 edge | 0.771 | 0.753 | 0.762 | 19 |
| CSC | 2 edges | 0.791 | 0.774 | 0.782 | 28 |
| GSC | exact | 0.858 | 0.698 | 0.770 | 0 |
| GSC | 1 edge | 0.908 | 0.739 | 0.815 | 11 |
| GSC | 2 edges | 0.922 | 0.750 | 0.827 | 14 |

At two edges, CSC added 14 predicted ancestors, seven predicted descendants,
and seven sibling matches. GSC added seven predicted ancestors, five predicted
descendants, and two sibling matches. These are not automatically correct
answers. They show that a measurable part of strict disagreement is nearby in
the ontology and belongs in a small human-adjudication packet.

## Historical repeat-run consistency baseline

The earlier three runs on the same 20 CSC cases produced:

- micro precision range: 0.011;
- micro recall range: 0.022;
- micro F1 range: 0.016;
- mean pairwise per-case HPO-set Jaccard: 0.801;
- IDs present in all three runs divided by the union of IDs: 0.707;
- mean per-case F1 range: 0.071.

Aggregate scores were fairly close, but exact accepted IDs were not stable
enough for the proposed reproducibility gate. This is a historical baseline
from the earlier pipeline, not a validation of the new context prompt.

## Machine-side changes

- Mapping now receives a deterministic context packet in the existing batched
  request. It adds no provider call.
- Adjacent sentences are included only for explicit ambiguity cues and are
  capped at 280 characters each.
- Zero-shot is the staged default. One-shot uses one fixed synthetic example
  and is an explicit ablation.
- Exact, one-edge, and two-edge metrics are produced separately with one-to-one
  matching and alternative-ID reference groups.
- Repeatability can now be measured from existing prediction files before any
  additional inference is authorized.

## Next controlled evaluation

Use a newly locked 10–20 case subset, three runs, and identical artifacts.
Compare zero-shot with one-shot only on discovery data, freeze one prompt, then
run the frozen prompt on the new subset. Do not run the full CSC or GSC corpus.
The consistency gate remains: metric ranges at most 0.02, mean pairwise Jaccard
at least 0.85, and all-run accepted-ID recurrence at least 0.90.

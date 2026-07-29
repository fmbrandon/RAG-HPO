# Phase 2 canonical-registry findings

Phase 2 is complete. RAG-HPO now builds every terminology view from a single,
versioned HPO registry and refuses incomplete or mismatched registry
provenance. The pinned HPO source and RAG-HPO add-ons remain logically
separate.

## Canonical registry

The complete build from HPO `2026-06-23` contains:

- 20,413 HPO concept records;
- 45,397 official label and synonym records;
- 3,047 distinct, active add-on records;
- 47,659 normalized lexical signatures;
- 15 signatures that explicitly map to more than one active HPO ID.

Each concept retains its label, definition, status, direct parents, alternate
IDs, and synonym scopes. Each phrase receives a stable SHA-256 record key.
Add-ons retain their own source and file hash and are never represented as
official HPO synonyms.

Two independent complete builds produced identical registry, lexical index,
manifest, metadata, and SapBERT vector hashes. The canonical registry SHA-256
is `e47253f1aac6a256248b7be9162e054ae84adbf7b2f89e107beee41e831c35b9`.

## Change from the earlier native matrix

The earlier matrix contained 48,444 rows. The canonical matrix contains 47,859
rows:

- 47,774 rows are unchanged;
- 669 obsolete label, synonym, or add-on rows were removed;
- one synonym whose capitalization was previously selected by unordered
  iteration is now selected deterministically;
- 84 add-on rows using valid HPO alternate IDs are now normalized to their
  current IDs instead of being silently skipped.

This is a terminology correction rather than dataset-specific tuning. Obsolete
concepts remain documented in the registry but cannot enter new mapping
artifacts.

Both complete SapBERT builds produced:

- metadata SHA-256
  `1c4ca3794182621199ee4d45044a99147344e08705f08b905cb77add7ec58b03`;
- vector SHA-256
  `83d2490bf951ab6db73a4fad9036f03715a3378534d93300dfd4f81f3fdfefde`.

## Deterministic FastHPOCR adapter

FastHPOCR's generated token clusters can make several HPO phrases
morphologically identical. The adapter now reconstructs this signature set and
uses the canonical registry to:

1. prefer a unique exact official/add-on phrase match;
2. return all IDs tied by an exact or morphological signature;
3. normalize alternate IDs;
4. reject IDs absent from the registry.

This resolves the Phase 1 nondeterminism. The phrase `difficult breathing`
matched three morphological candidates: `HP:0002094`, `HP:0002098`, and
`HP:0005957`. Both previously byte-different indexes now return that same
ordered ambiguity set instead of selecting one ID by container order.

Across all 116 CSC inputs, the two prior indexes now produce identical
normalized output SHA-256
`1f6a7572493eee6023769e120695b46a01069f99232a4f631bbecffd3e0bc40f`.
There were 1,644 recognized mentions, of which 53 remain explicitly ambiguous
and require later context verification or abstention.

The new full add-on index records:

- byte SHA-256
  `ac7e5148155c79344f429aaf1de34e99c6576d9235d5462ddb27836af2fd1806`;
- semantic SHA-256
  `de618ab2322597a3abd98cca5ab0fb26224b20e51dc2695a78ca1017d60b4067`;
- normalized 271-probe output SHA-256
  `14ac1a7c37acd2e2b51d10912e0debbbb83676559d5d44f44d8e2e436a00045e`.

Reuse validation checks all three identities. The full first build took about
305 seconds; validation and reuse took about ten seconds.

## Encoding and integrity

- OBO and add-on inputs must decode as UTF-8.
- Replacement characters, NULs, and known mojibake sequences are rejected.
- The registry parser suppresses its known Pronto confidence warning only
  after RAG-HPO has independently decoded and validated the complete source as
  UTF-8. FastHPOCR's separate parser may still display its own copy of that
  warning.
- Dense artifacts link the registry, lexical index, and both manifests by
  SHA-256.
- Tampering, partial provenance, different ontology/add-on hashes, unknown
  add-on IDs, and unknown FastHPOCR result IDs fail explicitly.

Machine-readable aggregate evidence is stored in
[`phase2-registry-summary.json`](benchmarks/results/phase2-registry-summary.json).
Complete registries, matrices, indexes, and probe artifacts remain private
under `RAG-HPO_AUDIT_ARTIFACTS/phase2-registry/`.

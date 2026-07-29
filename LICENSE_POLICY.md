# Dependency license policy

RAG-HPO core prefers permissive licenses. CI rejects dependencies whose reported
license contains GPL or LGPL unless the package is explicitly reviewed below.

## Approved exception

- `chardet` 5.x — LGPLv2+. It is an unmodified transitive parser dependency of
  Pronto, which is required to read HPO OBO files, and of the release SBOM
  tooling. RAG-HPO imports it dynamically through those libraries and does not
  copy or modify its source.

## Reviewed optional dependency

- `FastHPOCR` 0.1.4 — MIT, copyright 2024 Tudor Groza. It is an optional,
  unmodified concept-recognition dependency. The PyPI package includes its MIT
  license file but does not populate the standard `License`, `Home-page`, or
  `Requires-Dist` metadata fields. It imports Pronto at runtime, so the
  `rag-hpo[fasthpocr]` extra declares Pronto explicitly. RAG-HPO does not vendor
  FastHPOCR's bundled morphological vocabulary or base-synonym files.

This inventory is a technical compliance aid, not legal advice. Re-review the
exception when Pronto, SBOM tooling, or distribution practices change.

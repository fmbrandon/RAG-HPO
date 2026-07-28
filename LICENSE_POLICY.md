# Dependency license policy

RAG-HPO core prefers permissive licenses. CI rejects dependencies whose reported
license contains GPL or LGPL unless the package is explicitly reviewed below.

## Approved exception

- `chardet` 5.x — LGPLv2+. It is an unmodified transitive parser dependency of
  Pronto, which is required to read HPO OBO files, and of the release SBOM
  tooling. RAG-HPO imports it dynamically through those libraries and does not
  copy or modify its source.

This inventory is a technical compliance aid, not legal advice. Re-review the
exception when Pronto, SBOM tooling, or distribution practices change.

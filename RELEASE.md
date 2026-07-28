# Version 0.2.0 release-candidate process

The release gate is macOS ARM with Python 3.12. Linux and Windows are
experimental and must not be described as validated until native runs exist.

1. Run the local test, quality, notebook, package-build, and complete-artifact
   gates.
2. Build every candidate artifact with one command:

   ```bash
   RAG_HPO_VECTOR_DIR=/absolute/path/to/complete-vectors \
     scripts/build_release_artifacts.sh
   ```

   The script creates and removes its own temporary environment from the
   hash-locked macOS Python 3.12 requirements, so audit-only packages in the
   project workbench cannot contaminate the SBOM or license report.

3. Verify `dist/release/SHA256SUMS`, vector-bundle metadata, provenance, SBOM,
   dependency audit, licenses, and regenerated historical metric arithmetic.
4. Upload the candidate files for review only to the private personal
   repository. Do not create or publish an upstream tag without lab-owner
   authorization.
5. After an authorized upstream release owns the poster and workbook assets,
   remove those binaries from the source tree in a normal commit. Do not rewrite
   Git history.

Live provider tests are manual and must use synthetic text or repository cases
that have been explicitly authorized for that endpoint. Never place a provider
key in release files or workflow logs.

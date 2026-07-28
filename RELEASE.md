# Version 0.2.0 release process

1. Run the complete local and CI validation suite.
2. Run `scripts/build_release_artifacts.sh`.
3. Verify `dist/release/SHA256SUMS`, the SBOM, dependency audit, licenses, and
   regenerated benchmark metrics.
4. Create the `v0.2.0` GitHub release and attach everything under
   `dist/release/`.
5. Only after the release assets are available, remove the tracked poster and
   workbook in a normal follow-up commit. Do not rewrite Git history.

Live provider tests are manual and must use synthetic or explicitly authorized
example text. Never place a provider key in release files or workflow logs.

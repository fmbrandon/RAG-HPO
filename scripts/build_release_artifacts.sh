#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release_dir="${repo_root}/dist/release"

mkdir -p "${release_dir}/security" "${release_dir}/sbom"
cd "${repo_root}"

python -m build
python benchmarks/recompute_metrics.py \
  "RAG-HPO Tests and Data Analysis copy.xlsx" \
  --output "${release_dir}/benchmark-metrics.csv"
cyclonedx-py environment --output-file "${release_dir}/sbom/environment.cdx.json"
pip-audit --format json --output "${release_dir}/security/pip-audit.json"
pip-licenses --format=json --output-file="${release_dir}/security/licenses.json"
python scripts/check_licenses.py "${release_dir}/security/licenses.json"

cp "RAG-HPO ASHG Poster.pptx" "${release_dir}/"
cp "RAG-HPO Tests and Data Analysis copy.xlsx" "${release_dir}/"

(
  cd "${release_dir}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 shasum -a 256 > SHA256SUMS
)

printf 'Release artifacts created in %s\n' "${release_dir}"

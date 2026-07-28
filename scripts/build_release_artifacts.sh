#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release_dir="${repo_root}/dist/release"
vector_dir="${RAG_HPO_VECTOR_DIR:-}"
bootstrap_python="${RAG_HPO_RELEASE_PYTHON:-python3.12}"

if [[ -z "${vector_dir}" ]]; then
  printf 'RAG_HPO_VECTOR_DIR must point to a validated complete artifact.\n' >&2
  exit 2
fi

release_env="$(mktemp -d "${TMPDIR:-/tmp}/rag-hpo-release.XXXXXX")"
cleanup() {
  rm -rf -- "${release_env}"
}
trap cleanup EXIT

"${bootstrap_python}" -m venv "${release_env}"
release_python="${release_env}/bin/python"
"${release_python}" -m pip install --upgrade pip
"${release_python}" -m pip install \
  --require-hashes \
  -r "${repo_root}/requirements/macos-py312.txt"
"${release_python}" -m pip install --no-deps -e "${repo_root}"
export PATH="${release_env}/bin:${PATH}"

mkdir -p \
  "${release_dir}/benchmarks" \
  "${release_dir}/packages" \
  "${release_dir}/security" \
  "${release_dir}/sbom"
cd "${repo_root}"

pytest
python -m build
cp dist/rag_hpo-0.2.0-py3-none-any.whl "${release_dir}/packages/"
cp dist/rag_hpo-0.2.0.tar.gz "${release_dir}/packages/"
python scripts/build_vector_bundle.py \
  "${vector_dir}" \
  --output "${release_dir}/rag-hpo-v0.2.0-sapbert-vectors.zip"
python benchmarks/recompute_metrics.py \
  "RAG-HPO Tests and Data Analysis copy.xlsx" \
  --output "${release_dir}/benchmark-metrics.csv"
cp benchmarks/results/*.json "${release_dir}/benchmarks/"
cyclonedx-py environment --output-file "${release_dir}/sbom/environment.cdx.json"
pip-audit --format json --output "${release_dir}/security/pip-audit.json"
pip-licenses --format=json --output-file="${release_dir}/security/licenses.json"
python scripts/check_licenses.py "${release_dir}/security/licenses.json"

cp "RAG-HPO ASHG Poster.pptx" "${release_dir}/"
cp "RAG-HPO Tests and Data Analysis copy.xlsx" "${release_dir}/"

python - <<'PY' > "${release_dir}/provenance.json"
import json
import platform
import subprocess
import sys

print(json.dumps({
    "release": "v0.2.0-candidate",
    "platform": platform.platform(),
    "python": sys.version.split()[0],
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "platform_validation": {
        "macos_arm": "validated",
        "linux": "experimental-not-executed",
        "windows": "experimental-not-executed",
    },
}, indent=2, sort_keys=True))
PY

(
  cd "${release_dir}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 shasum -a 256 > SHA256SUMS
)

printf 'Release artifacts created in %s\n' "${release_dir}"

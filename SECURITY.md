# Security and clinical-data handling

## Reporting vulnerabilities

Please report suspected vulnerabilities privately to the maintainers listed in
`CITATION.cff`. Do not open a public issue containing an API key, clinical note,
provider response, checkpoint, or other sensitive material.

## Credentials

RAG-HPO reads the provider key from `RAG_HPO_API_KEY`. It never needs a key in
source code, a notebook, a flag file, or a committed environment file. Rotate a
key immediately if it is pasted into chat, logs, notebooks, or issue trackers.

## Clinical data

Users are responsible for institutional approval and provider agreements.
RAG-HPO does not make a remote provider HIPAA-eligible. De-identify notes before
use and confirm that the configured endpoint is approved for the data.

Raw provider responses are disabled by default. Resume databases and outputs may
still contain derived phenotype information and must be handled as sensitive.
Successful runs delete resume state unless `--keep-state` is selected.

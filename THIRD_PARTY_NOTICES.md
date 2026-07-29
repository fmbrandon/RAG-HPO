# Third-party notices

## FastHPOCR

RAG-HPO can optionally use FastHPOCR for local phenotype concept recognition.

- Project: <https://github.com/tudorgroza/fast_hpo_cr>
- Paper: <https://doi.org/10.1093/bioinformatics/btae406>
- Reviewed package: `FastHPOCR==0.1.4`
- Copyright: Copyright 2024 Tudor Groza
- License: MIT

The upstream MIT license permits use and modification provided its copyright
and permission notice are retained. RAG-HPO currently uses FastHPOCR only as an
optional installed dependency and does not copy its source or bundled
linguistic resources.

The package includes a generated morphological-cluster vocabulary and a small
base-synonym resource. Those files remain inside the separately installed
FastHPOCR distribution. They must not be vendored into RAG-HPO or included in a
RAG-HPO release bundle until their source and redistribution provenance has
been reviewed separately.

The complete upstream MIT text remains available in the installed
`FastHPOCR/license.txt` file.

## Human Phenotype Ontology

RAG-HPO uses the Human Phenotype Ontology. HPO requires acknowledgement,
version identification for publicly displayed files, and preservation of the
official vocabulary and logical relationships:
<https://human-phenotype-ontology.github.io/license.html>.

Official HPO files remain unmodified. Lab-provided phrases in `HPO_addons.csv`
are identified as RAG-HPO extensions and are not represented as official HPO
content.

This notice is a technical compliance record, not legal advice.

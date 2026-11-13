# X-ERD

Reproducible ransomware-detection experiments with behavioral parsing, stacked
tree ensembles, additive explanations and candidate MITRE ATT&CK mappings.

## Software version

Current software: **0.4.2**. Repository: https://github.com/matrupriya2048/X-ERD
Archived software version 0.4.2: https://doi.org/10.5281/zenodo.24190787

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24190787.svg)](https://doi.org/10.5281/zenodo.24190787)

The archive corresponds to tag `v0.4.2`, commit
`053c811a769ec28b33f0d8dfe861a389de380b38`. Later citation-only changes
on the main branch do not alter that archived software snapshot.

## Layout and reproduction

Keep these two directories side by side:

- `X_ERD_Reviewer_Revision`: companion implementation, pinned dependencies,
  Docker images, source acquisition, frozen benchmark results and tests.
- `X_ERD_Evidence_Followup`: current primary RF+LightGBM configuration,
  CAPE schema adapter, multi-collection adaptation, evaluation and attribution.

Start with the companion `README.md` for data acquisition and Docker setup.
Then follow `X_ERD_Evidence_Followup/README.txt`, especially its reproduction
order, using the companion scientific dependency locks. Raw reports and model
binaries are not distributed here; acquisition scripts, checksums, split
inventories, predictions and training/export commands are supplied.

The component folders preserve historical experiment versions and provenance.
The supplied release/follow-up manifests describe their earlier author working
packages. `PUBLICATION_MANIFEST.json` identifies this public source snapshot.
Manuscript preparation files, reviewer correspondence and author response
documents are excluded. Document-generation instructions in historical component
READMEs refer to the separate author package and are not required for experiments.

## Interpretation

Benchmark classification, raw-report transfer and retrospective adaptation are
distinct protocols. Preserve adverse results and model identities. The CAPE
adaptation holdout is retrospective; publisher benign labels are not independently
adjudicated ground truth. No independent human ATT&CK adjudication, verified
encryption-onset annotation or deployment-readiness claim is supplied.

## Citation and license

Author and software metadata are in `CITATION.cff`. The project code is MIT
licensed; see `LICENSE` and the companion `NOTICE`. External data remain subject
to their original publishers' licenses and citation requirements.

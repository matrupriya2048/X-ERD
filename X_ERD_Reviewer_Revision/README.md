# X-ERD reviewer revision

This isolated codebase corrects the reviewer experiments without changing the
original Random Forest / XGBoost / LightGBM plus L2 logistic stacking architecture.
The official 3,905/975 split and 483-feature benchmark remain the primary reference.
Original submitted files outside this folder are preserved.

## Reproduce

Python 3.11 and Docker are required for the container route. From this folder:

```bash
python scripts/prepare_data.py --strict-temporal
docker build -t xerd-reviewer-revision:0.1.0 .
bash scripts/run_docker.sh test
bash scripts/run_docker.sh all
```

On this Windows machine Docker runs in the existing `ClawbackVerify` WSL distro:

```powershell
wsl -d ClawbackVerify -- bash /mnt/c/Users/hp/Downloads/X_ERD_Scientific_Reports_Complete_Package/X_ERD_Reviewer_Revision/scripts/run_docker.sh test
wsl -d ClawbackVerify -- bash /mnt/c/Users/hp/Downloads/X_ERD_Scientific_Reports_Complete_Package/X_ERD_Reviewer_Revision/scripts/run_docker.sh all
```

The script mounts this new project at `/work`. No other project is modified.
Data downloads are checksum-verified; existing conflicting files are preserved and
cause a failure. No ransomware executable archive members are extracted or run.

For a local environment: `python -m pip install -e '.[test]'`, then
`python -m xerd.cli all`. Individual stages are available with `--help`.
The completed results in this delivery come from Docker, not a mixture of host runs.
The base image runs the core experiments. Build the version 0.3.0 extended image
below for sequence dependencies and the complete test suite; the base image skips
the optional sequence test.

## Scientific corrections

- Labels are explicitly derived from `type_label > 0` and checked against
  `sample_type`. IDs and all label metadata are forbidden inputs.
- Fresh inner 5-fold OOF predictions fit each logistic head. All 50 outer CV
  observations come from the declared original ten seeds. Nadeau-Bengio tests
  use the 0.27 variance factor; four comparisons are Holm-corrected separately
  for FPR and recall. Wilcoxon and paired rank-biserial are descriptive.
- Every base SHAP explanation uses probability output, an identical training
  background and interventional perturbation. Unnormalized logistic coefficients
  explain the trained stack **logit**; sigmoid reconstruction is checked. Uniform
  base averaging is explicitly a separate surrogate.
- All classifiers share identical corruption masks. Both recall and benign FPR
  are reported. Feature masking is a feature-space sensitivity experiment.
- Threshold selection uses training-validation data; test ROC/matched-recall
  summaries are descriptive. A validation recall constraint does not guarantee
  the same population or test recall after refitting.
- Strict temporal selection uses only samples dated before 2020, tests 2021-2024,
  and leaves 2020 as a gap. The declared raw pool contains API/network/system/
  dropped-file/signature features. Training-only MI top2000 and RFE select483.
  This is an additional representation, not a repeat of full 6.4M-feature RFE.
  The original fixed483 temporal diagnostic is separately labeled as subject to
  selection lookahead.
- All available20 raw reports are excluded before fitting the parser-validation
  model. All eight categories, prediction changes and attribution changes are
  reported. These reports contain ransomware only, so their benign FPR is undefined.
  Their earlier inclusion in upstream feature selection is disclosed: this is
  a conditional reconstruction check, not an untouched representation holdout.
- Time-prefix experiments use timestamped API calls only, with a separate API-only
  model. Full-run strings/signatures are excluded from prefixes. Encryption-onset
  labels were not supplied; prevention latency cannot be established from these
  reports. The extension below uses actual external benign/ransomware sequences.
- ATT&CK rules are versioned candidate suggestions. No independent analyst audit
  is invented. A reproducible blank30-item worksheet is generated for real review.

## Outputs

`results/` contains machine-readable metrics, per-record predictions, fold logs,
data checksums, execution timings, figures and tables. `docs/` contains reviewer
tracking and the remaining external requirements. `manuscript/` contains the
revised manuscript source and the clearly identified original reference copy.
Generated manuscript numbers must come from this results directory.

The supplied reference-only LaTeX and bibliography are generation inputs, not the
corrected paper. Submit only the explicitly named revised manuscript after the
remaining evidence/archive requirements are met. Old copied figures are isolated
under the ignored `manuscript/reference_snapshot/` directory.

Build the manuscript with Python `scripts/build_manuscript.py`, then
`scripts/build_bibliography.py`; run `pdflatex`, `bibtex`, and two further
`pdflatex` passes on `main_scirep_revised.tex` from inside `manuscript/`.
Repeat the same LaTeX/BibTeX sequence for `supplementary_information.tex`.
Run `python scripts/build_revision_report.py` and `python scripts/verify_release.py`
after experiments. The dependency lock targets the tested Linux Docker image.
After both PDFs and all artifact checks are complete, install the optional
`authoring` dependency and run `python scripts/build_response_letter.py`.
Compile `docs/response_letter.tex` with pdfLaTeX. Its 69 entries include manuscript
locations checked against the final PDF, plus the corresponding evidence paths.

The feed-forward neural control uses unordered binary indicators and is explicitly
not described as an API sequence model. Published scores from unrelated datasets
must not be inserted into the same-split comparison table.

## Extended experiments (version 0.3.0)

The original results remain intact. Additions include published MLRan grids,
EldeRan and RANSEC adaptations, hash-disjoint collection transfer, 30 genuine
LSTM/CNN fits on 80 real traces, 200 preselected external benign reports,
MITRE/Cuckoo reference concordance, and training-only configuration optimization.

Acquire the original data locally. The Nunes archive is solid-compressed: even
extracting only JSON reports can require several GB of cached compressed ranges.
The cache is resumable and excluded from Git/source releases. No executable
members are extracted or run.

```bash
python scripts/fetch_reference_sources.py
python scripts/prepare_external_data.py
python -m pip install -r requirements-acquisition.lock
python scripts/prepare_benign_reports.py
docker build -f Dockerfile.extended -t xerd-reviewer-revision:0.3.0 .
bash scripts/run_extended.sh published
bash scripts/run_extended.sh external_dataset
bash scripts/run_extended.sh external_mapping
bash scripts/run_extended.sh sequences
bash scripts/run_extended.sh optimization
bash scripts/run_extended.sh group_cv
bash scripts/run_extended.sh optimized_validation
bash scripts/run_extended.sh residual
bash scripts/run_extended.sh residual_validation
bash scripts/run_extended.sh external_comparators
bash scripts/run_extended.sh external_raw
docker run --rm -v "$PWD:/work" -w /work xerd-reviewer-revision:0.3.0 python scripts/build_extended_figures.py
docker run --rm -v "$PWD:/work" -w /work -e PYTHONPATH=/work/src xerd-reviewer-revision:0.3.0 python scripts/extended_statistics.py
docker run --rm -v "$PWD:/work" -w /work -e PYTHONPATH=/work/src xerd-reviewer-revision:0.3.0 python scripts/verify_extended.py
```

Run these after the core stages: external raw evaluation requires the saved
original, optimized and passthrough models. The optional acquisition lock is separate from model dependencies.
Source manifests record publisher labels, SHA-256 and exact selection. RANSEC
adaptations document ambiguous source descriptions and completion choices.
Neural sequence logs expose inner/outer partitions, epoch selection and vocabulary.
Optimization compares 8 fixed base settings and 3 logistic C values using training
CV, with fresh inner OOF fits. The selected configuration is evaluated once on the
previously reported official test; this is not a fresh prospective blind trial.

Interpret external results by protocol. Refitting EldeRan is different from
source-to-target transfer. API logs are not complete raw Cuckoo reports. Benign
reports from another collection must not be pooled with the original ransomware
subset to manufacture overall accuracy. Partial automated ATT&CK references and
AI-assisted source checks do not constitute independent human expert validation.

After experiments, build the manuscript/reviewer report and run BOTH verification
scripts in the tested container. Rebuild the source ZIP after PDF visual review.
For packaged-source testing (no source bind), use:

```bash
docker run --rm xerd-reviewer-revision:0.3.0 python -m pytest -q
docker run --rm --mount "type=bind,source=$PWD/data,target=/work/data,readonly" xerd-reviewer-revision:0.3.0 python -m pytest -q
```

The first command checks the packaged code without data (25 passed, one benchmark
integration test skipped). The second supplies only the benchmark data and runs
all 26 tests against the packaged code. Neither command mounts host source.

After completing all stages and acquisition manifests, `python
scripts/write_source_manifest.py` followed by `bash scripts/verify_final_image.sh`
rebuilds the final image, runs packaged tests, checks installed source hashes and
reconciles persisted experiment artifacts. Only data/results are mounted in the
artifact checks; the executable code and configuration come from the image.

## Disclosed raw-report remediation (version 0.3.0)

The original 182-report adverse evaluation is retained. Those reports became
development data for the new adaptation; they are never relabeled as a fresh test.
`config/raw_remediation_protocol.json` fixes 18 grouped-CV configurations, a source
recall constraint, a 0.50 threshold, and 400 newly sampled report paths. The source
483 features, parser and three base families are unchanged. The selected model's
hash is frozen before new target scoring. `results/remediation/holdout.json`
reports specificity on new executables from the same collection. This is domain
adaptation, not zero-shot transport. `ransomware.json` reports a separate check on
all 138 listed publisher-labeled Zirari JSON reports, with provider-restricted
files explicitly recorded as unavailable in the acquisition audit; the paper
describes 140 selected samples, a different unit from report files. Accessible
metadata do not fully substantiate the stated curation: 134 hashed reports contain
77 distinct target hashes, and 63/134 numeric scores are below the stated cutoff.
These discrepancies are reported without relabeling or score-based filtering.
The sources are not pooled for accuracy or precision.

The new same-collection benign holdout contains 362 eligible executions: the
previous passthrough model has 172 false alerts (47.51%), and the adapted full
stack has six (1.66%; exact 95% interval 0.61%-3.57%). The separate ransomware
endpoint has 55/58 adapted detections, versus 56/58 previously: recall
non-inferiority is not established. The adapted RF+XGB pair has four benign
false alerts and 55 ransomware detections; it is a descriptive ablation, not a
test-selected replacement for the frozen primary model.

Acquire inert reports on the host (optional acquisition dependencies are pinned):

```bash
python -m pip install -r requirements-acquisition.lock
python scripts/prefetch_benign_archive.py --remediation --workers 4
python scripts/prepare_raw_remediation.py --acquire
python scripts/prepare_zirari_raw.py
```

No executable samples are acquired or run. Member CRCs, report SHA-256 hashes,
publisher labels, overlap exclusions and every accepted/rejected path are recorded.
The original run obtained public Drive files through the read-only connector;
`prepare_zirari_raw.py` provides a public-download CLI and checks the same bytes.

Frozen reference outputs are distributed without model binaries. After running
the existing core and passthrough fit stages, reproduce the remediation into a
NEW output directory, preserving the reference evidence:

```bash
docker run --rm -v "$PWD:/work" -w /work -e PYTHONPATH=/work/src xerd-reviewer-revision:0.3.0 python scripts/reproduce_remediation.py --output results/my_remediation_replay
```

This reruns all nested fitting and reconciles every development, official-test
and new-benign-holdout prediction against the reference (tolerance 1e-12). It is
a deterministic replay, not a new prospective study. `python -m xerd.remediation
train` intentionally refuses to overwrite the original frozen evidence. Initial
development construction is in `prepare_remediation_development.py`; it checks
all old predictions against reconstructed raw vectors before development fitting.
The companion `remediation_validation.py` recomputes refitted probability-scale
SHAP and separate raw-ransomware sensitivity. `verify_remediation.py` independently
reconciles folds, selection, frozen hashes, predictions, intervals and source bytes.

Use `--output results/my_remediation_replay` with the companion validation
module to inspect a replay fit. The `components` stage recomputes deletion,
targeted feature-space masking and attribution-weighted mapping coverage.
`remaining_false_alerts.json` explains the six remaining benign errors without
changing the frozen model. `representation_diagnostics.json` checks exact-vector
overlap as a post-evaluation sensitivity analysis; no primary cases are removed.

Document generation uses the existing result-driven Python scripts, a TeX
distribution and Poppler. Optional local inspection dependencies are pinned in
`requirements-documents.lock`. Compile the main manuscript and supplement with
pdflatex, bibtex, pdflatex, pdflatex; generate the response only after their final
pagination. `verify_manuscripts.py` checks references, display inventory and word
counts. Rendered-page inspection remains a separate final check.

## Publication and archive

Read the external raw-report result before drawing deployment conclusions:
among 182 usable, source-disjoint publisher-clean reports, the original full
stack produces 95 false alarms (52.20%) and the passthrough full stack produces
92 (50.55%). The official benchmark and shared-API transfer improvements do
not establish reliable end-to-end benign specificity across arbitrary collections.
The separately documented version 0.3 adaptation and new holdout evidence address
this observed failure within their declared collection and sampling scope.

Do not write that this release is already publicly archived. A GitHub release and
an actual published Zenodo version DOI must be added after author review. The
archive should contain code, configuration, dependency lock, tests, lightweight
results and manuscript source. Keep raw data, cached model binaries and full
archives out of ordinary Git commits; use the official data source plus checksums.
The new source code is provided under the MIT license. `NOTICE` identifies the
third-party and manuscript materials that retain their separate rights.

Upstream MLRan: https://github.com/faithfulco/mlran

Official dataset record: https://zenodo.org/records/23403681

MLRan citation: Onwuegbuche et al. (2026), *Journal of Network and Computer
Applications*,104475. https://doi.org/10.1016/j.jnca.2026.104475

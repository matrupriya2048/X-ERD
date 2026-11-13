X-ERD REVIEWER REVISION 0.4.2 — CURRENT WORKING PACKAGE

LATEST FILES
Use this folder's manuscript/main_scirep_revised.pdf,
manuscript/supplementary_information.pdf and docs/response_letter.pdf.
The companion X_ERD_Reviewer_Revision folder preserves the verified v0.3 code,
historical model/result identities and older documents. Its PDFs are superseded
by the current follow-up PDFs. Base code and the supplied original files remain
unchanged.

WHAT IS COMPLETED
- Reviewer-directed RF+LightGBM primary analysis and two-base-only inference.
- Explicit CAPE event/static-string adapter with operation-provenance tests.
- EN2025: pinned published benign archive, 918 non-pilot inputs, one prior-target
  overlap excluded, 917 unique PE targets audited. Initial transport failure
  is retained, including the schema-translated 100% original-model alert rate.
- Fixed-parameter multi-collection pair repair: 3905 source training records,
  182 prior Nunes development controls and 634 CAPE development controls.
  The 283-case vector-group holdout is RETROSPECTIVE, not a fresh collection.
- On that holdout: 1/283 alerts (0.35%, exact95% interval0.009–1.953%).
  Earlier Nunes holdout: 4/362 alerts (1.10%). Separate raw-ransomware targets:
  55/58 detections. Reused official test: 98.46% accuracy,98.71% recall,1.76% FPR.
- Old source-only primary pair detects56/58 raw-ransomware targets, so the new
  pair has one extra miss. No recall non-inferiority is asserted.
- Provider benign labels conflict with50 CAPE Malicious and270 Suspicious
  statuses. All targets remain; no favorable relabeling/filtering is performed.
  Holdout Clean-status sensitivity is1/172 alerts, not independent human gold.
- Own-model probability SHAP and centered raw terms reconstruct1258 outputs
  within1.01e-08. Current calibration metrics, figures and parameters exported.
- Crypto-indicator reliance and all24 sequence-window/seed control groups.
- 19 focused tests; model/prediction replay; identical model SHA under fresh
  retraining; no development/holdout vector overlap; final raw-report Docker CLI.
- Every one of69 tracked comments has a response and evidence links.

CURRENT SCOPE
FINAL_COMMENT_LEDGER.html records every item. Independent human adjudication
is excluded from this round by the author. Verified first destructive encrypted-
write references are unavailable; timestamps/crypto API hints do not establish
pre-encryption efficacy. GitHub/Zenodo publication and author sign-off remain
the final author-controlled actions. Nothing has been uploaded or submitted.

REPRODUCTION ORDER
Keep the two source folders side by side. First reproduce the v0.3 companion
using its instructions, including pinned source data and frozen benchmark models.
Use its scientific dependency locks/Docker image; acquisition also needs the
companion requirements-acquisition.lock. Then in this folder:
  python acquire_mtm.py
  python evaluate_mtm.py
  python evidence_packets.py
  python close_available_work.py
  python primary_pair.py export --root ../X_ERD_Reviewer_Revision --output closure_results/primary_pair.joblib
  python acquire_en2025.py
  python evaluate_en2025.py
  python evaluate_cape_adapter.py
  python train_multicollection_pair.py
  python explain_multicollection.py
  python calibrate_final_pair.py
  python -m unittest test_evidence test_primary_pair test_cape_adapter -v
  python verify_closure.py
  python verify_en2025.py
  python verify_translated.py
  python verify_multicollection.py
Training refuses to overwrite an evaluated fitted model. Use a clean checkout
or a separate reproduction output directory; retain the original artifact hashes.
The acquisition protocol and split are supplied; do not reselect target paths.

DOCKER INFERENCE
Build the companion xerd-reviewer-revision:0.3.0 image, then:
  docker build -f Dockerfile.primary -t xerd-primary-pair:0.4.2 .
The final image supports predict and predict-report commands. Use
multicollection/model.joblib for the final adapted pair. The source-only
closure_results/primary_pair.joblib is a preserved benchmark comparator.
For raw reports, supply --feature-names pointing to the original 483-ID dictionary.
Mount input/output files and run with --network none. The CLI translates declared
CAPE inputs and leaves original Cuckoo inputs unchanged.

DOCUMENT REPRODUCTION (bundled pypdf/Pillow; MiKTeX; Poppler)
  python build_documents.py papers
  python apply_closure_documents.py papers
  python apply_en2025_documents.py papers
  python apply_multicollection_documents.py papers
  python compile_documents.py main
  python compile_documents.py supplement
  python build_documents.py response
  python apply_closure_documents.py response
  python apply_en2025_documents.py response
  python apply_multicollection_documents.py response
  python compile_documents.py response
  python finish_ledger.py
  python document_qa.py
Inspect the rendered pages before marking visual QA complete.

RELEASE CONTENTS
The archive contains code, protocols, results, figures, PDFs and checksums.
Raw reports, local downloaded archives, model binaries and reproduction copies
are excluded. MODEL_CHECKSUMS.json identifies the tested local model artifacts.
Data-provider licenses differ; consult their source records before redistribution.
No malware executable was acquired or run for this work; datasets contain reports.

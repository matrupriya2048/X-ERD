#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
python scripts/remediation_diagnostics.py
python scripts/build_extended_figures.py
python scripts/build_remediation_report.py
python scripts/build_manuscript.py
python scripts/build_bibliography.py
python scripts/build_revision_report.py

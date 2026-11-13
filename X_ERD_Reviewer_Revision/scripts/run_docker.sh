#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
mkdir -p results
case "${1:-all}" in
  test)
    docker run --rm --name xerd-reviewer-tests -e PYTHONPATH=/work/src \
      --mount "type=bind,source=$project_root,target=/work" --workdir /work \
      xerd-reviewer-revision:0.1.0 python -m pytest tests -q 2>&1 | tee results/docker_tests.log
    ;;
  *)
    docker run --rm --name "xerd-reviewer-${1:-all}" -e PYTHONPATH=/work/src \
      --mount "type=bind,source=$project_root,target=/work" --workdir /work \
      xerd-reviewer-revision:0.1.0 python -m xerd.cli "${1:-all}" 2>&1 | tee "results/docker_${1:-all}.log"
    ;;
esac

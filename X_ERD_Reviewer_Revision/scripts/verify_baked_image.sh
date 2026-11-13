#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
docker run --rm --name xerd-final-image-check \
  --mount "type=bind,source=$project_root/data,target=/work/data,readonly" \
  --mount "type=bind,source=$project_root/results,target=/work/results" \
  xerd-reviewer-revision:0.1.0 sh -c \
  'python -m pytest tests -q && python -m xerd.cli core --out results/image_smoke' \
  2>&1 | tee results/baked_image_test.log
docker image inspect xerd-reviewer-revision:0.1.0 > results/docker_image.json

#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
image=xerd-reviewer-revision:0.3.0
docker build --progress=plain -f Dockerfile.extended -t "$image" . 2>&1 | tee results/docker_build_v03.log
docker image inspect "$image" > results/docker_image_v03.json
docker run --rm "$image" python -m pytest -q 2>&1 | tee results/docker_packaged_tests_v03.log
docker run --rm --mount "type=bind,source=$project_root/data,target=/work/data,readonly" "$image" python -m pytest -q 2>&1 | tee results/docker_packaged_data_tests_v03.log
# Only data/results are bound: scripts, installed modules, tests and configuration
# must come from the image that a fresh reproducer will actually execute.
for script in verify_packaged_source verify_release verify_extended verify_remediation; do
  docker run --rm \
    --mount "type=bind,source=$project_root/data,target=/work/data,readonly" \
    --mount "type=bind,source=$project_root/results,target=/work/results" \
    "$image" python "scripts/$script.py" 2>&1 | tee "results/docker_$script.log"
done

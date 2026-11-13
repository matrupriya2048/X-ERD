#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
mkdir -p results/extended
stage=${1:?Specify experiment stage}
image=${XERD_IMAGE:-xerd-reviewer-revision:0.3.0}
docker run --rm --name "xerd-extended-$stage" -e PYTHONPATH=/work/src \
  --mount "type=bind,source=$project_root,target=/work" --workdir /work \
  "$image" python -m xerd.extended "$stage" 2>&1 | tee "results/extended/docker_$stage.log"

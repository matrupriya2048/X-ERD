#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
docker image inspect xerd-reviewer-revision:0.1.0 > results/docker_image.json
docker run --rm --name xerd-dependency-lock xerd-reviewer-revision:0.1.0 \
  python -m pip freeze --exclude xerd-reviewer-revision > requirements.lock
docker version > results/docker_version.txt
uname -a > results/wsl_platform.txt

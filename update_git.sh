#!/usr/bin/env bash
set -e

checkout() {
    local path="$1"
    local branch="$2"
    echo "  -> $path ($branch)"
    cd "$path"
    git fetch origin
    git checkout "$branch"
    git pull origin "$branch"
    cd - > /dev/null
}

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[1/2] Pulling main repository..."
git pull

echo "[2/2] Updating submodules..."
git submodule update --init --recursive

checkout services/diploma-course-service docker-integration
checkout services/diploma-frontend       docker-integration
checkout services/diploma-gateway        docker-integraion
checkout services/diploma-user-service   master

echo "Done."

#!/usr/bin/env bash
set -e

MSG="${1:-update}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "================================================"
echo " Git commit + push"
echo " Message: $MSG"
echo "================================================"

push_sub() {
    local path="$1"
    local branch="$2"
    echo ""
    echo "-> $path  ($branch)"
    cd "$ROOT/$path"
    git checkout "$branch" 2>/dev/null
    git add -A
    if ! git diff --cached --exit-code --quiet 2>/dev/null; then
        git commit -m "$MSG"
        git push origin "$branch"
    else
        echo "   nothing to commit"
    fi
    cd "$ROOT"
}

push_sub services/diploma-course-service docker-integration
push_sub services/diploma-frontend       docker-integration
push_sub services/diploma-gateway        docker-integraion
push_sub services/diploma-user-service   master

# master repo (includes diploma-terrain-service, terrain-gen, etc.)
echo ""
echo "-> master repo  (master)"
cd "$ROOT"
git add -A
if ! git diff --cached --exit-code --quiet 2>/dev/null; then
    git commit -m "$MSG"
    git push origin master
else
    echo "   nothing to commit"
fi

echo ""
echo "================================================"
echo " Done."
echo "================================================"

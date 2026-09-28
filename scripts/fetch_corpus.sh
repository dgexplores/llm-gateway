#!/usr/bin/env bash
# Fetch the evaluation corpus.
#
# FastAPI documentation, pinned to a commit. MIT licensed, which keeps the vendored
# copy redistributable. The pin matters: an unpinned corpus makes every retrieval
# number in docs/RESULTS.md incomparable to the run before it.
set -euo pipefail

REPO="https://github.com/fastapi/fastapi.git"
TARGET="${1:-corpus}"

if [ -d "$TARGET/.git" ]; then
  echo "corpus already present at $TARGET"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "cloning $REPO ..."
git clone --depth 1 --filter=blob:none --sparse "$REPO" "$tmp/fastapi" 2>&1 | tail -2
git -C "$tmp/fastapi" sparse-checkout set docs/en/docs
RESOLVED="$(git -C "$tmp/fastapi" rev-parse HEAD)"

mkdir -p "$TARGET"
cp -R "$tmp/fastapi/docs/en/docs/." "$TARGET/"

cat > "$TARGET/CORPUS.json" <<EOF
{
  "source": "fastapi/fastapi",
  "commit": "$RESOLVED",
  "license": "MIT",
  "path": "docs/en/docs"
}
EOF

count="$(find "$TARGET" -name '*.md' | wc -l | tr -d ' ')"
echo "corpus at $TARGET ($count markdown files)"
echo "commit $RESOLVED"
echo "pin that SHA in this script before publishing results so the run is reproducible"

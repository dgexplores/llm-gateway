#!/usr/bin/env bash
# Fetch the evaluation corpus.
#
# FastAPI documentation, MIT licensed, which keeps the vendored copy redistributable.
#
# The pin is the point. An unpinned corpus makes every retrieval number in
# docs/RESULTS.md incomparable to the run before it, and the numbers are the evidence
# this project rests on. So the commit is hard-coded and checked out directly; a clone
# of HEAD followed by a note asking you to pin it later is a pin that does not exist.
#
# To move the pin deliberately:
#   1. edit COMMIT below
#   2. rm -rf corpus && ./scripts/fetch_corpus.sh
#   3. uv run python scripts/run_eval.py && uv run python scripts/ablation.py
#   4. commit the changed numbers in docs/RESULTS.md with the new SHA
set -euo pipefail

REPO="https://github.com/fastapi/fastapi.git"
# The commit the published numbers in docs/RESULTS.md were measured against.
COMMIT="a3d205bf19640528718cb4f05ab77f4dfca6ad9a"
SPARSE_PATH="docs/en/docs"
TARGET="${1:-corpus}"

if [ -d "$TARGET/.git" ]; then
  echo "corpus already present at $TARGET"
  if [ -f "$TARGET/CORPUS.json" ]; then
    have="$(sed -n 's/.*"commit": "\([0-9a-f]*\)".*/\1/p' "$TARGET/CORPUS.json")"
    if [ "$have" != "$COMMIT" ]; then
      echo "WARNING: corpus is at $have but this script pins $COMMIT."
      echo "         Numbers in docs/RESULTS.md will not match. Run:"
      echo "         rm -rf $TARGET && $0"
    fi
  fi
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "fetching fastapi/fastapi at $COMMIT ..."
# A shallow clone of HEAD cannot reach an older commit, so fetch that one ref directly
# instead of cloning and hoping the default branch still points at it.
git init -q "$tmp/fastapi"
git -C "$tmp/fastapi" remote add origin "$REPO"
git -C "$tmp/fastapi" config core.sparseCheckout true
printf '%s\n' "$SPARSE_PATH" > "$tmp/fastapi/.git/info/sparse-checkout"
git -C "$tmp/fastapi" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git -C "$tmp/fastapi" checkout -q FETCH_HEAD

resolved="$(git -C "$tmp/fastapi" rev-parse HEAD)"
if [ "$resolved" != "$COMMIT" ]; then
  echo "FATAL: asked for $COMMIT, got $resolved" >&2
  exit 1
fi

mkdir -p "$TARGET"
cp -R "$tmp/fastapi/$SPARSE_PATH/." "$TARGET/"

cat > "$TARGET/CORPUS.json" <<EOF
{
  "source": "fastapi/fastapi",
  "commit": "$resolved",
  "license": "MIT",
  "path": "$SPARSE_PATH"
}
EOF

count="$(find "$TARGET" -name '*.md' | wc -l | tr -d ' ')"
echo "corpus at $TARGET ($count markdown files)"
echo "commit $resolved (pinned)"

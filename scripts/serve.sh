#!/usr/bin/env bash
# Run the gateway locally.
#
# Works on a clean clone with no credentials and no Docker: the service falls back to
# the extractive demo provider, which answers by quoting retrieved passages. Fetch the
# corpus first for retrieval to be exercised at all.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

if ! uv run python -c "import fastapi" 2>/dev/null; then
  echo "installing the serve extra ..."
  uv sync --extra serve
fi

if [ ! -d corpus ]; then
  echo "no corpus/ found - retrieval will be disabled and answers will be refusals."
  echo "run ./scripts/fetch_corpus.sh to enable it"
  echo
fi

cat <<INFO
  gateway   http://$HOST:$PORT
  health    http://$HOST:$PORT/api/health
  sources   http://$HOST:$PORT/api/sources?q=dependency+injection
  api docs  http://$HOST:$PORT/docs

  press ctrl-c to stop
INFO

exec uv run uvicorn gateway.api:app --host "$HOST" --port "$PORT" "$@"

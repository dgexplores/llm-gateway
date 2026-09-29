# syntax=docker/dockerfile:1
#
# Two stages so the runtime image carries no toolchain. The builder is discarded; what
# ships is a venv and the source it was built from.

FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# The dependency layer changes far less often than the source, so it is copied and
# installed first. Editing a module should not re-resolve the whole tree.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv==0.5.11 \
    && uv sync --locked --no-dev --no-install-project

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md ./
RUN uv sync --locked --no-dev


FROM python:3.12-slim AS runtime

# Unbuffered output, or the uvicorn access log vanishes into a container that exits.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# A non-root user. A container that runs the app as root turns any container escape
# into a host-root shell.
RUN useradd --create-home --uid 10001 gateway

WORKDIR /app

COPY --from=builder --chown=gateway:gateway /build/.venv /app/.venv
COPY --from=builder --chown=gateway:gateway /build/src /app/src
COPY --from=builder --chown=gateway:gateway /build/scripts /app/scripts
# fetch_corpus.sh is the entrypoint's first step, so it has to be in the image. git is
# needed at runtime too, because the script pins a commit and checks it out.
RUN apt-get update \
    && apt-get install --no-install-recommends -y git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

USER gateway
EXPOSE 8000

# The corpus is fetched at start time rather than baked in, so the image is not
# 1,500 documentation files and the pin lives in one place: scripts/fetch_corpus.sh.
# Without it the app starts and answers from an empty index, which looks like retrieval
# failure rather than a missing fixture.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

CMD ["sh -c", "./scripts/fetch_corpus.sh /app/corpus && exec uvicorn gateway.api:app --host 0.0.0.0 --port 8000"]

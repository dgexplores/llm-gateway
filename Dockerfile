# syntax=docker/dockerfile:1
#
# Two stages so the runtime image carries no toolchain. The builder is discarded; what
# ships is a venv and the source it was built from.

FROM python:3.12-slim AS builder

# UV_PROJECT_ENVIRONMENT pins the venv to the path it will occupy in the runtime stage.
# Left at the default, uv builds it at /build/.venv and every console script it writes gets
# a shebang of "#!/build/.venv/bin/python" — a path that does not exist once only the venv
# is copied into the runtime image. The container then fails at start with
# "exec: uvicorn: not found", which reads like a missing dependency and is actually a
# dangling interpreter path. Building in place at /app/.venv keeps the shebangs valid.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /build

# The dependency layer changes far less often than the source, so it is copied and
# installed first. Editing a module should not re-resolve the whole tree.
COPY pyproject.toml uv.lock ./
# --extra serve installs fastapi and uvicorn, which the image genuinely needs. Without it
# the container builds cleanly, passes --no-dev, and then dies at start with
# "exec: uvicorn: not found" — a confusing failure that looks like a broken image rather
# than a missing dependency. The heavy `rag` extra is deliberately not installed: BM25 is a
# pure-Python implementation, so the default configuration needs no numpy, no
# sentence-transformers and no torch, and the runtime image stays around 90MB.
RUN pip install --no-cache-dir uv==0.5.11 \
    && uv sync --locked --no-dev --extra serve --no-install-project

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md ./
# --no-editable bakes the package into site-packages. An editable install writes a .pth
# pointing back at the build stage's source directory, which the runtime image does not
# have, so the app would die with "No module named 'gateway'" — the same dangling-path
# problem as the shebangs above, one layer up. A container should ship a fixed package
# anyway: changing code means rebuilding the image, not mutating a running container.
RUN uv sync --locked --no-dev --extra serve --no-editable


FROM python:3.12-slim AS runtime

# Unbuffered output, or the uvicorn access log vanishes into a container that exits.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# A non-root user. A container that runs the app as root turns any container escape
# into a host-root shell.
RUN useradd --create-home --uid 10001 gateway

WORKDIR /app

COPY --from=builder --chown=gateway:gateway /app/.venv /app/.venv
COPY --from=builder --chown=gateway:gateway /build/src /app/src
COPY --from=builder --chown=gateway:gateway /build/scripts /app/scripts
# fetch_corpus.sh is the entrypoint's first step, so it has to be in the image. git is
# needed at runtime too, because the script pins a commit and checks it out.
RUN apt-get update \
    && apt-get install --no-install-recommends -y git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create the mount point in the image, owned by the app user. Docker initialises a fresh
# named volume from the image's directory at that path *including its ownership*, so this
# line is what lets the non-root user write into the corpus volume. Without it the volume
# is created root-owned and fetch_corpus.sh fails with "Permission denied" on every file —
# which looks like a network fault rather than a permissions one.
RUN mkdir -p /app/corpus && chown gateway:gateway /app/corpus

USER gateway
EXPOSE 8000

# The corpus is fetched at start time rather than baked in, so the image is not
# 1,500 documentation files and the pin lives in one place: scripts/fetch_corpus.sh.
# Without it the app starts and answers from an empty index, which looks like retrieval
# failure rather than a missing fixture.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

# Exec form, so each element is one argv entry: "sh", "-c", then the script. Written as
# ["sh -c", ...] the container fails at init with `exec: "sh -c": executable file not
# found in $PATH` — exec form does not split on spaces, so it looks for a binary whose
# name is literally "sh -c".
CMD ["sh", "-c", "./scripts/fetch_corpus.sh /app/corpus && exec uvicorn gateway.api:app --host 0.0.0.0 --port 8000"]

# credence-proxy: Bayesian AI gateway
# Routes LLM and search requests via EU maximisation (Credence DSL)
#
# Build:  docker build -t credence-proxy .
# Run:    docker run -p 8377:8377 -e ANTHROPIC_API_KEY=... credence-proxy

# ── Stage 1: Python deps + Julia resolution ──────────────────────
FROM python:3.12-slim-bookworm AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /credence

# Copy workspace definition and all Python packages
COPY pyproject.toml uv.lock* ./
COPY apps/python/credence_bindings/ apps/python/credence_bindings/
COPY apps/python/credence_agents/ apps/python/credence_agents/
COPY apps/python/credence_router/ apps/python/credence_router/
# Stub for bayesian_if (workspace member, not needed but must exist for uv)
RUN mkdir -p apps/python/bayesian_if && \
    echo '[project]\nname = "bayesian-if"\nversion = "0.0.0"\nrequires-python = ">=3.11"' \
    > apps/python/bayesian_if/pyproject.toml

# Install Python packages with server + search extras
RUN uv sync --extra server --extra search --no-dev 2>&1 | tail -5

# Copy Julia DSL source (needed for precompilation)
COPY src/ src/
COPY examples/ examples/
COPY brain/ brain/
COPY Project.toml ./

# Resolve Julia via pyjuliapkg + precompile Credence + brain/server.jl deps.
# Two activation paths because the runtime uses both:
#   - juliacall (in-process) for the precompile smoke test itself
#   - `julia --project=/credence` (subprocess) for the brain server
# Instantiating Credence's Project.toml ensures JSON3/HTTP etc. are
# precompiled in the depot; they would otherwise trigger a first-start
# compile inside the brain subprocess.
ENV PYTHON_JULIACALL_HANDLE_SIGNALS=yes
RUN uv run python -c "\
import os; \
os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'; \
from juliacall import Main as jl; \
jl.seval('push!(LOAD_PATH, \"/credence/src\")'); \
jl.seval('using Credence'); \
jl.seval('load_dsl(read(\"/credence/examples/router.bdsl\", String))'); \
print('Credence DSL precompiled successfully') \
" && \
    "$(uv run python -c 'import juliapkg; print(juliapkg.executable())')" \
      --project=/credence \
      -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile(); using Credence, JSON3; println("brain deps precompiled")'

# ── Stage 2: Slim runtime ────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libatomic1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /credence

# Copy the entire venv (includes Python packages + pyjuliapkg Julia install)
COPY --from=build /credence/.venv /credence/.venv

# Copy source code
COPY --from=build /credence/src/ src/
COPY --from=build /credence/examples/ examples/
COPY --from=build /credence/brain/ brain/
COPY --from=build /credence/Project.toml /credence/Manifest.toml ./
COPY --from=build /credence/apps/python/ apps/python/

# Copy Julia depot with precompiled packages
COPY --from=build /root/.julia /root/.julia

# Environment
ENV PYTHON_JULIACALL_HANDLE_SIGNALS=yes
ENV CREDENCE_STATE_PATH=/data/credence-state.json
ENV CREDENCE_LLM_STATE_PATH=/data/credence-llm-state.bin
ENV PATH="/credence/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/credence/.venv

# State persistence directory
RUN mkdir -p /data
VOLUME /data

EXPOSE 8377

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8377/ready || exit 1

ENTRYPOINT ["credence-router", "serve", "--host", "0.0.0.0", "--port", "8377"]

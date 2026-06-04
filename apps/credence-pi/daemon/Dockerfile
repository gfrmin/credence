# syntax=docker/dockerfile:1
# credence-pi daemon — the Bayesian governance brain (HTTP, port 8787).
#
# The brain the credence-pi OpenClaw plugin (@gfrmin/credence-pi-openclaw)
# talks to. Pure Julia: loads the Credence DSL + the feature-conditioned
# structure-BMA brain, exposes POST /sensor and GET /signals (+ GET /ready).
#
# Build (from repo root):
#   docker build -f apps/credence-pi/daemon/Dockerfile -t credence-pi-daemon .
# Run:
#   docker run -p 8787:8787 -v ~/.credence-pi:/root/.credence-pi credence-pi-daemon

ARG JULIA_VERSION=1.11

# ── Stage 1: instantiate + precompile the Credence project ───────────────
FROM julia:${JULIA_VERSION}-bookworm AS build
ENV JULIA_DEPOT_PATH=/depot
WORKDIR /credence

# Only what the daemon needs: the Credence project + DSL source, and the
# brain's bdsl/brain/daemon dirs (not the bodies, tests, or node_modules).
COPY Project.toml ./
COPY src/ src/
COPY apps/credence-pi/bdsl/   apps/credence-pi/bdsl/
COPY apps/credence-pi/brain/  apps/credence-pi/brain/
COPY apps/credence-pi/daemon/ apps/credence-pi/daemon/

# Instantiate Credence's declared deps (HTTP ~1.11, JSON3, …) and precompile
# into the depot, then a load smoke so the first container start is warm.
RUN julia --project=/credence -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' \
 && julia --project=/credence -e 'push!(LOAD_PATH,"/credence/src"); using Credence, HTTP, JSON3; println("Credence + daemon deps precompiled")'

# ── Stage 2: runtime ─────────────────────────────────────────────────────
FROM julia:${JULIA_VERSION}-bookworm AS runtime
ENV JULIA_DEPOT_PATH=/depot \
    CREDENCE_PI_HOST=0.0.0.0 \
    CREDENCE_PI_PORT=8787
# curl for the HEALTHCHECK / smoke probe.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /depot /depot
COPY --from=build /credence /credence
WORKDIR /credence
EXPOSE 8787

# The observation log persists under /root/.credence-pi (mount a volume there).
VOLUME ["/root/.credence-pi"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=45s --retries=3 \
  CMD curl -fsS http://localhost:8787/ready || exit 1

ENTRYPOINT ["julia", "--project=/credence", "/credence/apps/credence-pi/daemon/main.jl"]

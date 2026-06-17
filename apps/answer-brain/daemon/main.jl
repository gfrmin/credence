# Role: brain
"""
    main.jl — standalone entrypoint for the answer-brain daemon (Stage-1 skeleton).

Loads Credence + the brain + the observation log, and reports ready. The HTTP/SSE wire
surface (`server.jl`, the sensor→effector loop) is deferred to Stage 2, where the pi-mono
body defines the sensor-event schema (move-1-design §0/§"Q2"). Until then this entrypoint
exists to confirm the app loads under the repo project and to be the seam Stage 2 grows the
server onto.

Run:
    julia --project=<repo-root> apps/answer-brain/daemon/main.jl
"""

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence

include(joinpath(@__DIR__, "..", "brain", "answer_brain.jl"))
using .AnswerBrain
include(joinpath(@__DIR__, "observation_log.jl"))
using .ObservationLog

const BDSL_DIR = get(ENV, "ANSWER_BRAIN_BDSL_DIR",
                     normpath(joinpath(@__DIR__, "..", "bdsl")))
const LOG_PATH = get(ENV, "ANSWER_BRAIN_LOG",
                     joinpath(homedir(), ".answer-brain", "observations.jsonl"))

@info "answer-brain loaded (Stage 1: no wire surface — see move-1-design §Q2)" #=
=#    channel = AnswerBrain.CANONICAL_CHANNEL bdsl_dir = BDSL_DIR log = LOG_PATH
println(Dict("status" => "ready", "stage" => 1, "wire" => "deferred-to-stage-2"))

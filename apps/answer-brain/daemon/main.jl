# Role: brain
"""
    main.jl — answer-brain daemon entrypoint (Stage 2a: the stateless decision wire).

Loads Credence + the brain + the HTTP server (`server.jl`) and serves `POST /decide` + `GET /ready`
until interrupted. Stateless, so there is nothing to replay on boot (move-2-design §5 Resolution); the
observation log / live-session wiring lands with the pi-mono body in Stage 2b (Move 3).

Run:
    julia --project=<repo-root> apps/answer-brain/daemon/main.jl
Env:
    ANSWER_BRAIN_HOST (default 127.0.0.1), ANSWER_BRAIN_PORT (default 8799)
"""

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence

include(joinpath(@__DIR__, "..", "brain", "answer_brain.jl"))
using .AnswerBrain
include(joinpath(@__DIR__, "server.jl"))
using .Server

const HOST = get(ENV, "ANSWER_BRAIN_HOST", "127.0.0.1")
const PORT = parse(Int, get(ENV, "ANSWER_BRAIN_PORT", "8799"))

function main()
    server = start_daemon(; port = PORT, host = HOST)
    @info "answer-brain daemon listening (Stage 2a: stateless /decide)" host = HOST port = PORT
    # SIGINT → catchable InterruptException (not a hard exit), so the finally below runs
    # stop_daemon and releases the port cleanly on Ctrl-C (credence-pi daemon parity).
    Base.exit_on_sigint(false)
    try
        wait(server)
    catch e
        e isa InterruptException || rethrow()
    finally
        stop_daemon(server)
        @info "answer-brain daemon stopped"
    end
end

main()

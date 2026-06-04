# Role: brain
"""
    main.jl — standalone entrypoint for the credence-pi daemon.

`server.jl` is a module that expects `Credence` already loaded in `Main`
(the test/demo loaders do `using Credence` then `include` it). This script
is the canonical way to run the daemon as its own process: it loads
Credence, starts the HTTP server, and blocks until the process is signalled.
It is also the Docker image's ENTRYPOINT.

Run:
    julia --project=<repo-root> apps/credence-pi/daemon/main.jl

Configuration via environment (all optional):
    CREDENCE_PI_HOST       bind host          (default 127.0.0.1)
    CREDENCE_PI_PORT       bind port          (default 8787)
    CREDENCE_PI_BDSL_DIR   BDSL program dir   (default ../bdsl next to this file)
    CREDENCE_PI_LOG        observation log    (default ~/.credence-pi/observations.jsonl)
"""

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence

include(joinpath(@__DIR__, "server.jl"))
using .Server: start_daemon, stop_daemon

const HOST     = get(ENV, "CREDENCE_PI_HOST", "127.0.0.1")
const PORT     = parse(Int, get(ENV, "CREDENCE_PI_PORT", "8787"))
const BDSL_DIR = get(ENV, "CREDENCE_PI_BDSL_DIR", normpath(joinpath(@__DIR__, "..", "bdsl")))
const LOG_PATH = get(ENV, "CREDENCE_PI_LOG", joinpath(homedir(), ".credence-pi", "observations.jsonl"))

mkpath(dirname(LOG_PATH))

server, _state = start_daemon(; port = PORT, host = HOST, log_path = LOG_PATH, bdsl_dir = BDSL_DIR)
@info "credence-pi daemon listening" host = HOST port = PORT bdsl_dir = BDSL_DIR log = LOG_PATH

# Block until the listener closes. SIGINT (Ctrl-C) surfaces as an
# InterruptException; SIGTERM (docker stop) lets Julia exit. The
# observation log is fsync'd per event, so an abrupt stop never corrupts
# state — replay reconstructs the posterior on the next start.
Base.exit_on_sigint(false)
try
    wait(server)
catch e
    e isa InterruptException || rethrow()
finally
    stop_daemon(server)
    @info "credence-pi daemon stopped"
end

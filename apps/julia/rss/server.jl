#!/usr/bin/env julia
"""
    server.jl — HTTP service for RSS article ranking.

    Reads from the shared Miniflux + sidecar PostgreSQL database.
    Exposes ranking and conditioning endpoints for the Python sidecar.

    Usage:
        DATABASE_URL=postgres://... julia apps/julia/rss/server.jl

    Startup: connects to DB, initializes agent from feeds/tags,
    bootstraps from historical read events, then serves HTTP on port 8081.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
include(joinpath(@__DIR__, "host.jl"))
include(joinpath(@__DIR__, "db.jl"))

using HTTP
using JSON3
using Dates

# ── Global state ──

mutable struct ServerState
    agent::Union{Nothing, AgentState}
    registry::Union{Nothing, FeatureRegistry}
    conn::Union{Nothing, LibPQ.Connection}
    priorities::Dict{Int, Int}
    last_trained::DateTime
    lock::ReentrantLock
end

const STATE = ServerState(nothing, nothing, nothing, Dict{Int,Int}(), DateTime(2000,1,1), ReentrantLock())

# ── Helpers ──

function json_response(data; status=200)
    HTTP.Response(status, ["Content-Type" => "application/json"], JSON3.write(data))
end

function with_lock(f)
    lock(STATE.lock) do
        f()
    end
end

# ── Endpoints ──

function handle_init(req::HTTP.Request)
    db_url = get(ENV, "DATABASE_URL", "")
    isempty(db_url) && return json_response(Dict(:error => "DATABASE_URL not set"), status=500)

    with_lock() do
        STATE.conn = connect_db(db_url)
        feeds = fetch_feeds(STATE.conn)
        categories = fetch_categories(STATE.conn)
        known_tags = fetch_known_tags(STATE.conn)
        STATE.priorities = fetch_feed_priorities(STATE.conn)

        state, reg = init_rss_agent(
            feeds = feeds, categories = categories, known_tags = known_tags,
            program_max_depth = 3, verbose = true,
        )
        STATE.agent = state
        STATE.registry = reg
        STATE.last_trained = DateTime(2000, 1, 1)

        json_response(Dict(
            :status => "initialized",
            :n_programs => length(state.belief.components),
            :n_features => length(reg.feature_set),
            :n_feeds => length(feeds),
            :n_tags => length(known_tags),
        ))
    end
end

function handle_train(req::HTTP.Request)
    STATE.agent === nothing && return json_response(Dict(:error => "not initialized"), status=400)

    # Parse optional 'since' from body
    since = STATE.last_trained
    if !isempty(HTTP.payload(req))
        body = JSON3.read(HTTP.payload(req))
        if haskey(body, :since)
            since = DateTime(body[:since])
        end
    end

    with_lock() do
        conn = STATE.conn
        reg = STATE.registry
        now = Dates.now(Dates.UTC)

        # Fetch events
        reads = fetch_read_events_since(conn, since)
        dismissals = fetch_dismiss_events_since(conn, since)

        if isempty(reads) && isempty(dismissals)
            return json_response(Dict(
                :n_reads => 0, :n_dismissals => 0,
                :n_components => length(STATE.agent.belief.components),
            ))
        end

        # Load features for all articles involved
        all_entry_ids = unique(vcat(
            [e.entry_id for e in reads],
            [e.entry_id for e in dismissals],
        ))

        # Also need all unread articles for the Plackett-Luce denominator
        unread_articles = fetch_unread_articles(conn, STATE.priorities)
        unread_features = Dict{Int, Dict{Symbol, Float64}}()
        for a in unread_articles
            unread_features[a.id] = extract_features(a, reg; now=now)
        end

        # Add features for read/dismissed articles (they may no longer be unread)
        event_features = fetch_article_features_for_entries(
            conn, all_entry_ids, reg, STATE.priorities; now=now,
        )
        merge!(unread_features, event_features)

        # Group reads into sessions (30-min gaps)
        sessions = _group_into_sessions(reads, Dates.Minute(30))

        for session_reads in sessions
            process_session!(STATE.agent, session_reads, DismissEvent[],
                              unread_features; verbose=false)
        end

        # Process dismissals
        if !isempty(dismissals)
            process_session!(STATE.agent, ReadEvent[], dismissals,
                              unread_features; verbose=false)
        end

        STATE.last_trained = now

        json_response(Dict(
            :n_reads => length(reads),
            :n_dismissals => length(dismissals),
            :n_sessions => length(sessions),
            :n_components => length(STATE.agent.belief.components),
        ))
    end
end

function handle_rank(req::HTTP.Request)
    STATE.agent === nothing && return json_response(Dict(:error => "not initialized"), status=400)

    # Parse ?n=50 from query string
    params = HTTP.queryparams(HTTP.URI(req.target))
    n = parse(Int, get(params, "n", "50"))

    with_lock() do
        conn = STATE.conn
        reg = STATE.registry
        now = Dates.now(Dates.UTC)

        unread = fetch_unread_articles(conn, STATE.priorities)
        features = Dict{Int, Dict{Symbol, Float64}}()
        for a in unread
            features[a.id] = extract_features(a, reg; now=now)
        end

        ranking = rank_articles(STATE.agent, features)
        top_n = first(ranking, n)

        json_response(Dict(
            :ranking => [Dict(:entry_id => id, :score => round(s, digits=6)) for (id, s) in top_n],
            :total_unread => length(unread),
        ))
    end
end

function handle_health(req::HTTP.Request)
    if STATE.agent === nothing
        return json_response(Dict(:status => "not_initialized"))
    end

    with_lock() do
        n_comp = length(STATE.agent.belief.components)
        w = weights(STATE.agent.belief)
        gw = aggregate_grammar_weights(w, STATE.agent.metadata)
        top_gid = isempty(gw) ? 0 : first(sort(collect(gw), by=x->-x[2]))[1]

        json_response(Dict(
            :status => "ok",
            :n_components => n_comp,
            :top_grammar_id => top_gid,
            :top_grammar_weight => isempty(gw) ? 0.0 : round(gw[top_gid], digits=4),
            :last_trained => string(STATE.last_trained),
        ))
    end
end

# ── Session grouping ──

function _group_into_sessions(reads::Vector{ReadEvent}, gap::Dates.Period)::Vector{Vector{ReadEvent}}
    isempty(reads) && return Vector{ReadEvent}[]
    sorted = sort(reads, by=e -> e.read_at)
    sessions = Vector{ReadEvent}[]
    current = [sorted[1]]
    for i in 2:length(sorted)
        if sorted[i].read_at - sorted[i-1].read_at > gap
            push!(sessions, current)
            current = [sorted[i]]
        else
            push!(current, sorted[i])
        end
    end
    push!(sessions, current)
    sessions
end

# ── Router ──

const ROUTER = HTTP.Router()
HTTP.register!(ROUTER, "POST", "/init", handle_init)
HTTP.register!(ROUTER, "POST", "/train", handle_train)
HTTP.register!(ROUTER, "GET", "/rank", handle_rank)
HTTP.register!(ROUTER, "GET", "/health", handle_health)

# ── Main ──

function main()
    port = parse(Int, get(ENV, "CREDENCE_PORT", "8081"))
    println("Credence RSS service starting on port $port")

    # Auto-init if DATABASE_URL is set
    if haskey(ENV, "DATABASE_URL")
        println("Auto-initializing from DATABASE_URL...")
        handle_init(HTTP.Request("POST", "/init"))
        println("Bootstrapping from historical read events...")
        handle_train(HTTP.Request("POST", "/train", [], UInt8[]))
        println("Ready.")
    end

    HTTP.serve(ROUTER, "0.0.0.0", port)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

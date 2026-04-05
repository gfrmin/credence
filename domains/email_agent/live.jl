#!/usr/bin/env julia
"""
    live.jl — Live host driver for the email agent on real Fastmail email

Fetches inbox emails via JMAP, runs the same Bayesian decision loop as
host.jl, but with interactive CLI review instead of a simulated user.
All actions are timed and feed the cost model. State persists across sessions.

Usage:
    julia domains/email_agent/live.jl
    OLLAMA_URL=http://localhost:11434 julia domains/email_agent/live.jl
"""

# host.jl brings all domain files + Credence imports + shared functions
include("host.jl")
include("jmap_client.jl")
include("state_persistence.jl")

# ═══════════════════════════════════════
# Interactive review
# ═══════════════════════════════════════

const ACTION_NAMES = Dict(
    :archive => "ARCHIVE", :flag_urgent => "FLAG_URGENT",
    :schedule_later => "SCHEDULE_LATER", :draft_response => "DRAFT_RESPONSE",
    :delegate => "DELEGATE", :summarise => "SUMMARISE",
    :triage_urgent => "TRIAGE_URGENT", :silent_archive => "SILENT_ARCHIVE",
    :escalate => "ESCALATE",
)

function display_email(idx::Int, total::Int, email::Email, preview::String,
                       recommendation::Symbol, confidence::Float64,
                       cost_model::CostModel)
    println("━" ^ 50)
    println("Email $idx/$total: $(email.subject)")
    println("  From: $(email.sender)  |  hour=$(email.hour_received)")
    !isempty(preview) && println("  Preview: $(first(preview, 120))$(length(preview) > 120 ? "..." : "")")
    rec_name = get(ACTION_NAMES, recommendation, string(recommendation))
    println("  Agent recommends: $rec_name (confidence: $(round(confidence, digits=2)))")
    llm_sec = round(expected_seconds(cost_model, :ask_llm), digits=1)
    jmap_sec = round(expected_seconds(cost_model, :jmap_mutate), digits=1)
    println("  Cost beliefs: ask_llm=$(llm_sec)s  jmap=$(jmap_sec)s")
    println("  [a]pprove  [o]verride  [s]kip  [q]uit")
    print("> ")
end

"""
    interactive_review(email, recommendation, ...) → (:approve|:override|:skip|:quit, action)

Get user feedback on the agent's recommendation.
"""
function interactive_review(idx, total, email, preview, recommendation, confidence, cost_model)
    display_email(idx, total, email, preview, recommendation, confidence, cost_model)
    input = strip(readline())
    isempty(input) && (input = "a")  # default: approve

    if startswith(input, "a")
        return (:approve, recommendation)
    elseif startswith(input, "o")
        println("  Override — choose action:")
        for (i, a) in enumerate(DOMAIN_ACTIONS)
            name = get(ACTION_NAMES, a, string(a))
            println("    $i. $name")
        end
        print("  > ")
        choice = tryparse(Int, strip(readline()))
        if choice !== nothing && 1 <= choice <= length(DOMAIN_ACTIONS)
            return (:override, DOMAIN_ACTIONS[choice])
        end
        println("  Invalid choice, skipping.")
        return (:skip, :do_nothing)
    elseif startswith(input, "s")
        return (:skip, :do_nothing)
    elseif startswith(input, "q")
        return (:quit, :do_nothing)
    else
        println("  Unknown input, skipping.")
        return (:skip, :do_nothing)
    end
end

# ═══════════════════════════════════════
# JMAP action execution
# ═══════════════════════════════════════

function execute_jmap_action!(session::JMAPSession, jmap_id::String, action::Symbol,
                              cost_model::CostModel; verbose::Bool=true)
    if action in (:archive, :silent_archive)
        elapsed = @elapsed begin
            archive_id = get_mailbox_id(session, "archive")
            move_emails(session, [jmap_id], archive_id)
        end
        observe_cost!(cost_model, :jmap_mutate, elapsed)
        verbose && println("  → Archived ($(round(elapsed, digits=2))s)")
    elseif action in (:flag_urgent, :escalate, :triage_urgent)
        elapsed = @elapsed set_keyword(session, [jmap_id], "\$flagged")
        observe_cost!(cost_model, :jmap_mutate, elapsed)
        verbose && println("  → Flagged ($(round(elapsed, digits=2))s)")
    elseif action == :schedule_later
        elapsed = @elapsed set_keyword(session, [jmap_id], "\$later")
        observe_cost!(cost_model, :jmap_mutate, elapsed)
        verbose && println("  → Marked later ($(round(elapsed, digits=2))s)")
    elseif action == :delegate
        elapsed = @elapsed set_keyword(session, [jmap_id], "\$delegated")
        observe_cost!(cost_model, :jmap_mutate, elapsed)
        verbose && println("  → Delegated ($(round(elapsed, digits=2))s)")
    elseif action == :draft_response
        verbose && println("  → TODO: draft response (not yet implemented)")
    else
        verbose && println("  → $(action) (no JMAP action)")
    end
end

# ═══════════════════════════════════════
# Main live loop
# ═══════════════════════════════════════

function run_live(;
    program_max_depth::Int = 2,
    min_log_prior::Float64 = -15.0,
    max_meta_per_step::Int = 3,
    ask_cost::Float64 = 0.1,
    email_limit::Int = 100,
    state_path::String = DEFAULT_STATE_PATH,
    dry_run::Bool = false,
    verbose::Bool = true
)
    # 1. Connect to Fastmail
    println("Connecting to Fastmail...")
    session = discover_session()
    println("Connected: account=$(session.account_id)")

    # 2. Configure LLM
    ollama_url = get(ENV, "OLLAMA_URL", "")
    if isempty(ollama_url)
        # Fall back to OLLAMA_HOST, adding scheme/port if bare
        raw = get(ENV, "OLLAMA_HOST", "")
        if !isempty(raw)
            ollama_url = startswith(raw, "http") ? raw : "http://$raw:11434"
        end
    end
    llm_config = if !isempty(ollama_url)
        LLMConfig(ollama_url, "llama3.2", 200, true, 10.0)
    else
        default_llm_config()  # disabled
    end
    verbose && !llm_config.enabled && println("LLM enrichment: disabled (set OLLAMA_URL to enable)")
    verbose && llm_config.enabled && println("LLM enrichment: enabled ($(llm_config.host))")
    dry_run && println("DRY RUN: no JMAP mutations will be executed")

    # 3. Load or initialise state
    loaded = load_email_state(state_path)
    if loaded !== nothing
        state = loaded.state
        cost_model = loaded.cost_model
        sender_history = loaded.sender_history
        thread_counts = loaded.thread_counts
        temporal_state = loaded.temporal_state
        verbose && println("Loaded state: $(length(state.belief.components)) components, " *
                          "$(length(state.grammars)) grammars")
    else
        cost_model = default_cost_model()
        sender_history = Dict{String, Int}()
        thread_counts = Dict{String, Int}()
        temporal_state = Dict{Symbol, Any}(:recent => Dict{Symbol, Float64}[])

        grammar_pool = generate_email_seed_grammars()
        components = Measure[]
        log_prior_weights = Float64[]
        metadata = Tuple{Int, Int}[]
        compiled_kernels = CompiledKernel[]
        all_programs = Program[]

        idx = 0
        for g in grammar_pool
            programs = enumerate_programs(g, program_max_depth;
                                           action_space=DOMAIN_ACTIONS,
                                           min_log_prior=min_log_prior)
            for (pi, p) in enumerate(programs)
                idx += 1
                push!(components, TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaMeasure(1.0, 1.0)))
                push!(log_prior_weights, -g.complexity * log(2) - p.complexity * log(2))
                push!(metadata, (g.id, pi))
                push!(compiled_kernels, compile_kernel(p, g, pi))
                push!(all_programs, p)
            end
        end

        belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior_weights)
        grammar_dict = Dict{Int, Grammar}(g.id => g for g in grammar_pool)
        state = AgentState(belief, metadata, compiled_kernels, all_programs,
                           grammar_dict, program_max_depth)
        verbose && println("Fresh state: $(length(components)) components, " *
                          "$(length(grammar_pool)) grammars")
    end

    # 4. Fetch inbox (exclude already-processed and flagged emails)
    println("\nFetching inbox (unflagged, unprocessed)...")
    elapsed_fetch = @elapsed (raw_emails = query_inbox_emails(session;
        limit=email_limit,
        exclude_keywords=["\$flagged", "\$credence_processed"]))
    observe_cost!(cost_model, :jmap_fetch, elapsed_fetch)
    println("Fetched $(length(raw_emails)) emails ($(round(elapsed_fetch, digits=2))s)")

    isempty(raw_emails) && (println("No emails to process."); return)

    # Convert to Email structs
    emails_with_meta = map(enumerate(raw_emails)) do (i, raw)
        email, jmap_id, preview = jmap_to_email(raw, i;
            sender_history=sender_history, thread_counts=thread_counts)
        # Update session tracking
        sender_history[email.sender] = get(sender_history, email.sender, 0) + 1
        thread_id = String(get(raw, "threadId", ""))
        !isempty(thread_id) && (thread_counts[thread_id] = get(thread_counts, thread_id, 0) + 1)
        (email=email, jmap_id=jmap_id, preview=preview)
    end

    # 5. Process emails
    println("\n" * "=" ^ 50)
    println("Processing $(length(emails_with_meta)) emails")
    println("=" ^ 50)

    n_approved = 0
    n_overridden = 0
    n_skipped = 0

    for (step, em) in enumerate(emails_with_meta)
        email, jmap_id, preview = em.email, em.jmap_id, em.preview
        features = extract_features(email)

        # Inner meta/sensor loop (same as host.jl single-decision path)
        meta_cost_this_turn = 0.0
        meta_actions_taken = 0
        already_enriched = false
        chosen_action = :do_nothing

        while true
            rec_cache = Dict{Int, Symbol}()
            evaluate_programs!(rec_cache, state.compiled_kernels,
                               features, temporal_state)
            w = weights(state.belief)

            action_eus = Dict{Symbol, Float64}()
            for a in ALL_ACTIONS
                action_eus[a] = compute_eu(state, a, rec_cache, w;
                    cost_model=cost_model, ask_cost=ask_cost,
                    meta_cost_this_turn=meta_cost_this_turn,
                    already_enriched=already_enriched)
            end
            chosen_action = argmax(action_eus)

            if chosen_action in META_ACTIONS && chosen_action != :do_nothing &&
               meta_actions_taken < max_meta_per_step
                elapsed = @elapsed execute_meta_action!(state, chosen_action;
                    action_space=DOMAIN_ACTIONS, min_log_prior=min_log_prior,
                    verbose=verbose)
                observe_cost!(cost_model, chosen_action, elapsed)
                meta_actions_taken += 1
                meta_cost_this_turn += expected_cost(cost_model, chosen_action)
                sync_prune!(state; threshold=-30.0)
                sync_truncate!(state; max_components=2000)
                continue
            end

            if chosen_action == :ask_llm && !already_enriched
                elapsed = @elapsed (features = llm_enrich_features(
                    llm_config, email, features; preview=preview))
                observe_cost!(cost_model, :ask_llm, elapsed)
                already_enriched = true
                verbose && println("  [Sensor: ask_llm, $(round(elapsed, digits=2))s]")
                continue
            end

            break
        end

        # Fall back to domain action if still on meta/sensor
        if chosen_action in META_ACTIONS || chosen_action in SENSOR_ACTIONS
            rec_cache = Dict{Int, Symbol}()
            evaluate_programs!(rec_cache, state.compiled_kernels,
                               features, temporal_state)
            w = weights(state.belief)
            action_eus = Dict{Symbol, Float64}()
            for a in EMAIL_ACTIONS
                action_eus[a] = compute_eu(state, a, rec_cache, w;
                    cost_model=cost_model, ask_cost=ask_cost)
            end
            chosen_action = argmax(action_eus)
        end

        # Compute confidence
        rec_cache = Dict{Int, Symbol}()
        evaluate_programs!(rec_cache, state.compiled_kernels, features, temporal_state)
        w = weights(state.belief)
        confidence = compute_eu(state, chosen_action, rec_cache, w;
            cost_model=cost_model, ask_cost=ask_cost)

        # Interactive review (timed → observe :ask_user cost)
        elapsed_review = @elapsed begin
            feedback, correct_action = interactive_review(
                step, length(emails_with_meta), email, preview,
                chosen_action, confidence, cost_model)
        end
        observe_cost!(cost_model, :ask_user, elapsed_review)

        if feedback == :quit
            println("\nSaving state and exiting...")
            break
        elseif feedback == :skip
            n_skipped += 1
            continue
        elseif feedback == :approve
            correct_action = chosen_action
            n_approved += 1
        else  # :override
            n_overridden += 1
        end

        # Condition on user feedback
        k = build_email_observation_kernel(
            state.compiled_kernels, features, temporal_state, correct_action)
        state.belief = condition(state.belief, k, 1.0)
        sync_prune!(state; threshold=-30.0)
        sync_truncate!(state; max_components=2000)

        # Execute JMAP action + mark as processed
        if dry_run
            verbose && println("  [DRY RUN] Would execute: $(correct_action)")
        else
            try
                execute_jmap_action!(session, jmap_id, correct_action, cost_model; verbose=verbose)
                set_keyword(session, [jmap_id], "\$credence_processed")
            catch e
                @warn "JMAP action failed" action=correct_action exception=e
            end
        end
    end

    # 6. Save state
    println("\nSaving state to $state_path...")
    save_email_state(state_path, state, cost_model;
        sender_history=sender_history, thread_counts=thread_counts,
        temporal_state=temporal_state)

    # Summary
    println("\n" * "=" ^ 50)
    println("Session summary")
    println("  Approved: $n_approved  Overridden: $n_overridden  Skipped: $n_skipped")
    println("  Components: $(length(state.belief.components))")
    println("  Cost beliefs:")
    for (k, v) in cost_model.beliefs
        println("    $k: $(round(expected_seconds(cost_model, k), digits=2))s")
    end
    println("=" ^ 50)
end

# ═══════════════════════════════════════
# Entry point
# ═══════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    dry_run = "--dry-run" in ARGS
    run_live(; dry_run=dry_run)
end

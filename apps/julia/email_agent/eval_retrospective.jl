#!/usr/bin/env julia
# Role: brain-side application
"""
    eval_retrospective.jl — Evaluate the agent on real email using flag/archive labels

Uses the user's existing email as a labeled dataset:
- Flagged emails (any mailbox) → user chose :flag_urgent at triage time
- Unflagged emails (archive) → user chose :archive at triage time

Fetches both sets, shuffles, splits 70/30 train/test, runs the agent
through training with conditioning, then evaluates on held-out test set.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
using Credence
using Credence: weights, mean, condition
using Credence: TaggedBetaMeasure, MixtureMeasure, BetaMeasure
using Credence: Interval, Finite, Kernel, Measure
using Credence: Grammar, Program, CompiledKernel
using Credence: enumerate_programs, compile_kernel
using Credence: AgentState, sync_prune!, sync_truncate!
using Credence: aggregate_grammar_weights, show_expr

include(joinpath(@__DIR__, "host.jl"))
include(joinpath(@__DIR__, "jmap_client.jl"))

using Random

# ═══════════════════════════════════════
# Fetch labeled data from JMAP
# ═══════════════════════════════════════

function fetch_labeled_emails(; max_flagged::Int=200, max_unflagged::Int=500,
                                after::String="2026-02-06T00:00:00Z")
    println("Connecting to Fastmail...")
    session = discover_session()
    println("Connected: $(session.account_id)")

    props = ["id", "subject", "from", "preview", "receivedAt",
             "threadId", "keywords", "size", "hasAttachment"]

    # 1. Fetch all flagged emails (any mailbox)
    println("Fetching flagged emails...")
    resp_flag = jmap_call(session, [
        ["Email/query", Dict(
            "accountId" => session.account_id,
            "filter" => Dict("hasKeyword" => "\$flagged", "after" => after),
            "sort" => [Dict("property" => "receivedAt", "isAscending" => false)],
            "limit" => max_flagged
        ), "q0"]
    ])
    flag_ids = [String(id) for id in resp_flag[1][2]["ids"]]

    # 2. Fetch recent unflagged from archive
    println("Fetching archived (unflagged) emails...")
    archive_id = get_mailbox_id(session, "archive")
    resp_arch = jmap_call(session, [
        ["Email/query", Dict(
            "accountId" => session.account_id,
            "filter" => Dict(
                "operator" => "AND",
                "conditions" => [
                    Dict("inMailbox" => archive_id),
                    Dict("after" => after),
                    Dict("notKeyword" => "\$flagged"),
                ]
            ),
            "sort" => [Dict("property" => "receivedAt", "isAscending" => false)],
            "limit" => max_unflagged
        ), "q0"]
    ])
    unflag_ids = [String(id) for id in resp_arch[1][2]["ids"]]

    println("  Flagged: $(length(flag_ids)), Unflagged: $(length(unflag_ids))")

    # 3. Fetch full details for both sets
    all_ids = vcat(flag_ids, unflag_ids)
    flag_id_set = Set(flag_ids)
    raw_emails = Any[]

    for batch_start in 1:100:length(all_ids)
        batch = all_ids[batch_start:min(batch_start+99, length(all_ids))]
        resp = jmap_call(session, [
            ["Email/get", Dict(
                "accountId" => session.account_id,
                "ids" => batch,
                "properties" => props
            ), "g0"]
        ])
        for item in resp[1][2]["list"]
            push!(raw_emails, item)
        end
    end

    # 4. Convert to Email structs, building sender_history as we go
    sender_history = Dict{String, Int}()
    thread_counts = Dict{String, Int}()

    # First pass: count senders/threads across all emails
    for raw in raw_emails
        from_list = get(raw, "from", nothing)
        sender = from_list !== nothing && length(from_list) > 0 ?
            String(get(from_list[1], "email", "unknown")) : "unknown"
        sender_history[sender] = get(sender_history, sender, 0) + 1
        tid = String(get(raw, "threadId", ""))
        !isempty(tid) && (thread_counts[tid] = get(thread_counts, tid, 0) + 1)
    end

    # Normalize so jmap_to_email sees realistic frequencies
    # (jmap_to_email divides by 20, so scale counts to give ~0.5 for median sender)
    max_count = maximum(values(sender_history); init=1)
    for (k, v) in sender_history
        sender_history[k] = round(Int, v / max_count * 20)
    end

    # Second pass: convert to Email structs with labels
    labeled = Tuple{Email, Bool, String}[]  # (email, is_flagged, preview)
    for (i, raw) in enumerate(raw_emails)
        email, jmap_id, preview = jmap_to_email(raw, i;
            sender_history=sender_history, thread_counts=thread_counts)
        is_flagged = jmap_id in flag_id_set
        push!(labeled, (email, is_flagged, preview))
    end

    println("  Total labeled: $(length(labeled))")
    labeled
end

# ═══════════════════════════════════════
# Initialize agent
# ═══════════════════════════════════════

function init_agent(; action_space::Vector{Symbol}, program_max_depth::Int=2,
                      min_log_prior::Float64=-15.0)
    grammar_pool = generate_email_seed_grammars()

    components = Measure[]
    log_prior_weights = Float64[]
    metadata = Tuple{Int, Int}[]
    compiled_kernels = CompiledKernel[]
    all_programs = Program[]

    idx = 0
    for g in grammar_pool
        programs = enumerate_programs(g, program_max_depth;
                                       action_space=action_space,
                                       min_log_prior=min_log_prior)
        for (pi, p) in enumerate(programs)
            idx += 1
            push!(components, TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaMeasure(1.0, 1.0)))
            lw = -g.complexity * log(2) - p.complexity * log(2)
            push!(log_prior_weights, lw)
            push!(metadata, (g.id, pi))
            push!(compiled_kernels, compile_kernel(p, g, pi))
            push!(all_programs, p)
        end
    end

    belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior_weights)
    grammar_dict = Dict{Int, Grammar}(g.id => g for g in grammar_pool)
    AgentState(belief, metadata, compiled_kernels, all_programs,
               grammar_dict, program_max_depth)
end

# ═══════════════════════════════════════
# Utility function — pref(true_action, chosen_action) → Float64
# ═══════════════════════════════════════

function eval_utility(true_action::Symbol, chosen_action::Symbol)::Float64
    true_action == :flag_urgent && chosen_action == :flag_urgent && return 1.0
    true_action == :flag_urgent && chosen_action == :archive    && return -1.0
    true_action == :archive     && chosen_action == :flag_urgent && return -0.1
    true_action == :archive     && chosen_action == :archive    && return 0.5
    0.0
end

const EVAL_SKIP_UTILITY = -0.05

# ═══════════════════════════════════════
# Run evaluation
# ═══════════════════════════════════════

function run_eval(; rng_seed::Int=42, train_frac::Float64=0.7)
    action_space = [:flag_urgent, :archive]
    labeled = fetch_labeled_emails()

    # Shuffle deterministically
    rng = MersenneTwister(rng_seed)
    shuffled = labeled[randperm(rng, length(labeled))]

    n_train = round(Int, length(shuffled) * train_frac)
    train_set = shuffled[1:n_train]
    test_set = shuffled[n_train+1:end]

    n_flag_train = count(x -> x[2], train_set)
    n_flag_test = count(x -> x[2], test_set)
    println("\n=== SPLIT ===")
    println("Train: $(length(train_set)) ($(n_flag_train) flag, $(length(train_set) - n_flag_train) archive)")
    println("Test:  $(length(test_set)) ($(n_flag_test) flag, $(length(test_set) - n_flag_test) archive)")
    baseline = max(n_flag_test, length(test_set) - n_flag_test) / length(test_set)
    println("Majority baseline: $(round(baseline * 100, digits=1))%")

    # Initialize
    state = init_agent(; action_space=action_space, program_max_depth=3, min_log_prior=-20.0)
    temporal_state = Dict{Symbol, Any}(:recent => Dict{Symbol, Float64}[])
    println("\nInitialized: $(length(state.belief.components)) components, $(length(state.grammars)) grammars")

    # ── Training ──
    println("\n=== TRAINING ($(length(train_set)) emails) ===")
    train_correct = 0
    train_total = 0
    window_correct = 0
    window_size = 20

    for (i, (email, is_flagged, _)) in enumerate(train_set)
        true_label = is_flagged ? :flag_urgent : :archive
        features = extract_features(email)

        result = select_action_eu(state, features, temporal_state;
            action_space=action_space, utility=eval_utility, skip_utility=EVAL_SKIP_UTILITY)
        predicted = result.action
        correct = predicted == true_label
        train_correct += correct
        train_total += 1

        # Condition on true label
        k = build_email_observation_kernel(
            state.compiled_kernels, features, temporal_state, true_label)
        state.belief = condition(state.belief, k, 1.0)
        sync_prune!(state; threshold=-50.0)
        sync_truncate!(state; max_components=5000)

        # Rolling window accuracy
        if i % window_size == 0
            recent_start = max(1, i - window_size + 1)
            acc = train_correct / train_total
            println("  Step $i/$n_train: cumulative=$(round(acc*100, digits=1))%, " *
                    "components=$(length(state.belief.components))")
        end
    end
    println("Train accuracy: $(round(train_correct/train_total*100, digits=1))%")

    # ── Test ──
    println("\n=== TEST ($(length(test_set)) emails) ===")
    test_correct = 0
    tp = 0; fp = 0; fn = 0; tn = 0
    skipped_flag = 0; skipped_arch = 0

    for (email, is_flagged, _) in test_set
        true_label = is_flagged ? :flag_urgent : :archive
        features = extract_features(email)
        result = select_action_eu(state, features, temporal_state;
            action_space=action_space, utility=eval_utility, skip_utility=EVAL_SKIP_UTILITY)
        predicted = result.action

        if predicted == :ask_user
            is_flagged ? (skipped_flag += 1) : (skipped_arch += 1)
        elseif predicted == true_label
            test_correct += 1
            is_flagged ? (tp += 1) : (tn += 1)
        else
            is_flagged ? (fn += 1) : (fp += 1)
        end
    end

    n_decided = tp + fp + fn + tn
    n_skipped = skipped_flag + skipped_arch
    decided_acc = n_decided > 0 ? (tp + tn) / n_decided : 0.0
    println("Decided: $n_decided, Skipped: $n_skipped")
    println("Accuracy (when decided): $(round(decided_acc*100, digits=1))%")
    println("Majority baseline: $(round(baseline*100, digits=1))%")

    println("\n=== CONFUSION MATRIX ===")
    println("                 Predicted")
    println("              flag    archive  skip")
    println("Actual flag   $(lpad(tp, 4))    $(lpad(fn, 4))    $(lpad(skipped_flag, 4))")
    println("Actual arch   $(lpad(fp, 4))    $(lpad(tn, 4))    $(lpad(skipped_arch, 4))")
    if tp + fp > 0
        println("Precision (flag): $(round(tp/(tp+fp)*100, digits=1))%")
    end
    if tp + fn + skipped_flag > 0
        println("Recall (flag):    $(round(tp/(tp+fn+skipped_flag)*100, digits=1))%")
    end
    if n_skipped > 0
        println("Skip rate: $(round(n_skipped/length(test_set)*100, digits=1))%")
    end

    # ── Top programs ──
    println("\n=== TOP 10 PROGRAMS ===")
    w = weights(state.belief)
    sorted_idx = sortperm(w, rev=true)
    for rank in 1:min(10, length(sorted_idx))
        i = sorted_idx[rank]
        p = state.all_programs[i]
        tbm = state.belief.components[i]::TaggedBetaMeasure
        theta = mean(tbm.beta)
        g_id, _ = state.metadata[i]
        println("  $(rank). w=$(round(w[i]*100, digits=2))% θ=$(round(theta, digits=3)) " *
                "g$(g_id) $(show_expr(p.expr))")
    end

    # Grammar distribution in surviving programs
    println("\n=== GRAMMAR DISTRIBUTION ===")
    grammar_w = Dict{Int, Float64}()
    grammar_n = Dict{Int, Int}()
    for (i, (g_id, _)) in enumerate(state.metadata)
        grammar_w[g_id] = get(grammar_w, g_id, 0.0) + w[i]
        grammar_n[g_id] = get(grammar_n, g_id, 0) + 1
    end
    for g_id in sort(collect(keys(grammar_w)))
        println("  Grammar $g_id: $(grammar_n[g_id]) programs, weight=$(round(grammar_w[g_id]*100, digits=1))%")
    end

    (test_accuracy=decided_acc, baseline=baseline, state=state)
end

# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

run_eval()

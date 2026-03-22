"""
    metrics.jl — Email agent performance tracking

Tracks action accuracy, ASK_USER frequency, grammar weights,
surprise, and meta-action counts across the email sequence.
"""

mutable struct EmailMetricsTracker
    steps::Vector{Int}
    actions_taken::Vector{Symbol}
    correct_actions::Vector{Symbol}
    action_correct::Vector{Bool}
    asked_user::Vector{Bool}
    grammar_weights::Vector{Dict{Int, Float64}}
    n_components::Vector{Int}
    surprise::Vector{Float64}
    cumulative_ask_count::Vector{Int}
    meta_actions_per_step::Vector{Int}

    EmailMetricsTracker() = new(
        Int[], Symbol[], Symbol[], Bool[], Bool[],
        Dict{Int,Float64}[], Int[], Float64[], Int[], Int[])
end

function record_email!(m::EmailMetricsTracker;
                       step::Int,
                       action_taken::Symbol,
                       correct_action::Symbol,
                       is_correct::Bool,
                       asked::Bool,
                       grammar_weights::Dict{Int, Float64},
                       n_components::Int,
                       surprise::Float64,
                       n_meta_actions::Int=0)
    push!(m.steps, step)
    push!(m.actions_taken, action_taken)
    push!(m.correct_actions, correct_action)
    push!(m.action_correct, is_correct)
    push!(m.asked_user, asked)
    push!(m.grammar_weights, grammar_weights)
    push!(m.n_components, n_components)
    push!(m.surprise, surprise)
    prev_asks = isempty(m.cumulative_ask_count) ? 0 : last(m.cumulative_ask_count)
    push!(m.cumulative_ask_count, prev_asks + (asked ? 1 : 0))
    push!(m.meta_actions_per_step, n_meta_actions)
end

function print_email_summary(m::EmailMetricsTracker; last_n::Int=20)
    n = length(m.steps)
    n == 0 && return

    start = max(1, n - last_n + 1)
    println("\n--- Email metrics (steps $(m.steps[start])-$(m.steps[end])) ---")

    recent_correct = sum(m.action_correct[start:end])
    recent_total = n - start + 1
    println("Recent accuracy: $recent_correct / $recent_total ($(round(100 * recent_correct / recent_total, digits=1))%)")
    println("Total asks: $(last(m.cumulative_ask_count))")
    println("Total meta-actions: $(sum(m.meta_actions_per_step))")
    println("Components: $(last(m.n_components))")

    gw = last(m.grammar_weights)
    sorted_g = sort(collect(gw), by=x -> -x[2])
    println("Top grammars:")
    for (gi, w) in sorted_g[1:min(5, length(sorted_g))]
        println("  Grammar $gi: $(round(w, digits=6))")
    end
end

"""
    time_to_convergence(metrics; start_step, end_step, accuracy_threshold, window) → Int

Count steps from start_step until rolling-window accuracy reaches threshold.
"""
function time_to_convergence(m::EmailMetricsTracker;
    start_step::Int, end_step::Int,
    accuracy_threshold::Float64=0.7, window::Int=10)::Int
    for (idx, step) in enumerate(m.steps)
        step >= start_step || continue
        step <= end_step || break
        window_start = max(1, idx - window + 1)
        n = idx - window_start + 1
        correct = sum(m.action_correct[window_start:idx])
        if correct / n >= accuracy_threshold
            return step - start_step
        end
    end
    end_step - start_step
end

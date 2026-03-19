"""
    metrics.jl — Per-step tracking for program-space agent

Tracks grammar weights, program weights, accuracy, energy,
VOI, and surprise across the game.
"""

mutable struct MetricsTracker
    steps::Vector{Int}
    grammar_weights::Vector{Dict{Int, Float64}}
    top_programs::Vector{Vector{Tuple{Int, Int, Float64}}}  # (grammar_id, program_id, weight)
    prediction_correct::Vector{Bool}
    cumulative_energy::Vector{Float64}
    surprise::Vector{Float64}  # negative log-likelihood
    n_components::Vector{Int}
    n_grammars::Vector{Int}

    MetricsTracker() = new(Int[], Dict{Int,Float64}[], Vector{Tuple{Int,Int,Float64}}[],
                           Bool[], Float64[], Float64[], Int[], Int[])
end

function record!(m::MetricsTracker;
                 step::Int,
                 grammar_weights::Dict{Int, Float64},
                 top_programs::Vector{Tuple{Int, Int, Float64}},
                 correct::Bool,
                 energy::Float64,
                 surprise::Float64,
                 n_components::Int,
                 n_grammars::Int)
    push!(m.steps, step)
    push!(m.grammar_weights, grammar_weights)
    push!(m.top_programs, top_programs)
    push!(m.prediction_correct, correct)
    prev_energy = isempty(m.cumulative_energy) ? 0.0 : last(m.cumulative_energy)
    push!(m.cumulative_energy, prev_energy + energy)
    push!(m.surprise, surprise)
    push!(m.n_components, n_components)
    push!(m.n_grammars, n_grammars)
end

function print_summary(m::MetricsTracker; last_n::Int=10)
    n = length(m.steps)
    n == 0 && return

    start = max(1, n - last_n + 1)
    println("\n--- Metrics summary (steps $(m.steps[start])-$(m.steps[end])) ---")
    println("Cumulative energy: $(round(last(m.cumulative_energy), digits=1))")

    recent_correct = sum(m.prediction_correct[start:end])
    recent_total = n - start + 1
    println("Recent accuracy: $recent_correct / $recent_total ($(round(100 * recent_correct / recent_total, digits=1))%)")
    println("Components: $(last(m.n_components)), Grammars: $(last(m.n_grammars))")

    # Top grammars by weight
    gw = last(m.grammar_weights)
    sorted_g = sort(collect(gw), by=x -> -x[2])
    println("Top grammars:")
    for (gi, w) in sorted_g[1:min(5, length(sorted_g))]
        println("  Grammar $gi: $(round(w, digits=6))")
    end

    # Top programs
    tp = last(m.top_programs)
    if !isempty(tp)
        println("Top programs:")
        for (gi, pi, w) in tp[1:min(5, length(tp))]
            println("  G$(gi)P$(pi): $(round(w, digits=6))")
        end
    end
end

"""Aggregate per-component weights into grammar-level weights."""
function aggregate_grammar_weights(component_weights::Vector{Float64},
                                    metadata::Vector{Tuple{Int, Int}})::Dict{Int, Float64}
    gw = Dict{Int, Float64}()
    for (i, (gi, _)) in enumerate(metadata)
        gw[gi] = get(gw, gi, 0.0) + component_weights[i]
    end
    gw
end

"""Get top-k programs by weight."""
function top_k_programs(component_weights::Vector{Float64},
                        metadata::Vector{Tuple{Int, Int}};
                        k::Int=10)::Vector{Tuple{Int, Int, Float64}}
    perm = sortperm(component_weights, rev=true)
    result = Tuple{Int, Int, Float64}[]
    for i in perm[1:min(k, length(perm))]
        gi, pi = metadata[i]
        push!(result, (gi, pi, component_weights[i]))
    end
    result
end

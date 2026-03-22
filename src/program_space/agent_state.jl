"""
    agent_state.jl — AgentState and parallel-array management

Bundles the MixtureMeasure belief with its parallel arrays (metadata,
compiled_kernels, all_programs). sync_prune!/sync_truncate! keep them
in lock-step and reindex TaggedBetaMeasure tags.
"""

using ..Ontology

# ═══════════════════════════════════════
# AgentState — bundles belief with parallel arrays
# ═══════════════════════════════════════

mutable struct AgentState
    belief::Ontology.MixtureMeasure
    metadata::Vector{Tuple{Int, Int}}       # (grammar_id, program_id)
    compiled_kernels::Vector{CompiledKernel}
    all_programs::Vector{Program}
    grammars::Dict{Int, Grammar}            # grammar_id → Grammar
    current_max_depth::Int                  # current enumeration depth
end

"""
    sync_prune!(state; threshold) → state

Prune negligible components AND the parallel arrays together.
Reindex TaggedBetaMeasure tags so that tag == array position.
"""
function sync_prune!(state::AgentState; threshold::Float64=-30.0)
    max_lw = maximum(state.belief.log_weights)
    keep = [i for i in eachindex(state.belief.log_weights)
            if state.belief.log_weights[i] - max_lw > threshold]
    length(keep) == length(state.belief.components) && return state
    new_comps = Ontology.Measure[Ontology.TaggedBetaMeasure(state.belief.components[k].space, j,
                                          state.belief.components[k].beta)
                        for (j, k) in enumerate(keep)]
    state.belief = Ontology.MixtureMeasure(state.belief.space, new_comps,
                                   state.belief.log_weights[keep])
    state.metadata = state.metadata[keep]
    state.compiled_kernels = state.compiled_kernels[keep]
    state.all_programs = state.all_programs[keep]
    state
end

"""
    sync_truncate!(state; max_components) → state

Keep only the top-weighted components. Reindex tags.
"""
function sync_truncate!(state::AgentState; max_components::Int=2000)
    length(state.belief.components) <= max_components && return state
    perm = sortperm(state.belief.log_weights, rev=true)
    keep = perm[1:min(max_components, length(perm))]
    new_comps = Ontology.Measure[Ontology.TaggedBetaMeasure(state.belief.components[k].space, j,
                                          state.belief.components[k].beta)
                        for (j, k) in enumerate(keep)]
    state.belief = Ontology.MixtureMeasure(state.belief.space, new_comps,
                                   state.belief.log_weights[keep])
    state.metadata = state.metadata[keep]
    state.compiled_kernels = state.compiled_kernels[keep]
    state.all_programs = state.all_programs[keep]
    state
end

# ═══════════════════════════════════════
# Grammar weight aggregation
# ═══════════════════════════════════════

"""Aggregate per-component weights into grammar-level weights."""
function aggregate_grammar_weights(component_weights::Vector{Float64},
                                    metadata::Vector{Tuple{Int, Int}})::Dict{Int, Float64}
    gw = Dict{Int, Float64}()
    for (i, (gi, _)) in enumerate(metadata)
        gw[gi] = get(gw, gi, 0.0) + component_weights[i]
    end
    gw
end

"""Return top-k grammar IDs by aggregated posterior weight."""
function top_k_grammar_ids(state::AgentState, k::Int)::Vector{Int}
    w = Ontology.weights(state.belief)
    gw = aggregate_grammar_weights(w, state.metadata)
    sorted = sort(collect(keys(gw)), by=gi -> -get(gw, gi, 0.0))
    sorted[1:min(k, length(sorted))]
end

# ═══════════════════════════════════════
# Add programs to state (with deduplication)
# ═══════════════════════════════════════

"""
    add_programs_to_state!(state, grammar, max_depth; ...) → Int

Enumerate programs from `grammar` at `max_depth`, compile kernels, and
append to all parallel arrays. Deduplicates: skips programs whose
(grammar_id, expr) already exists in state.all_programs — injecting
fresh Beta(1,1) for already-observed hypotheses disrupts the posterior.

Returns count of programs added.
"""
function add_programs_to_state!(
    state::AgentState,
    grammar::Grammar,
    max_depth::Int;
    action_space::Vector{Symbol}=Symbol[:classify],
    min_log_prior::Float64=-20.0,
    include_temporal::Bool=false
)::Int
    programs = enumerate_programs(grammar, max_depth;
                                   action_space=action_space,
                                   min_log_prior=min_log_prior,
                                   include_temporal=include_temporal)

    # Build set of existing expressions for this grammar (for dedup)
    existing_exprs = [state.all_programs[i].expr
                      for i in eachindex(state.all_programs)
                      if state.metadata[i][1] == grammar.id]

    n_added = 0
    base_idx = length(state.compiled_kernels)
    new_components = Ontology.Measure[]
    new_lw = Float64[]
    new_meta = Tuple{Int, Int}[]
    new_ck = CompiledKernel[]
    new_progs = Program[]

    for (pi, p) in enumerate(programs)
        # Skip if this expression already exists for this grammar
        any(e -> expr_equal(e, p.expr), existing_exprs) && continue

        base_idx += 1
        push!(new_components, Ontology.TaggedBetaMeasure(
            Ontology.Interval(0.0, 1.0), base_idx,
            Ontology.BetaMeasure(1.0, 1.0)))
        lw = -grammar.complexity * log(2) - p.complexity * log(2)
        push!(new_lw, lw)
        push!(new_meta, (grammar.id, pi))
        push!(new_ck, compile_kernel(p, grammar, pi))
        push!(new_progs, p)
        n_added += 1
    end

    if !isempty(new_components)
        all_comps = Ontology.Measure[state.belief.components..., new_components...]
        all_lw = Float64[state.belief.log_weights..., new_lw...]
        state.belief = Ontology.MixtureMeasure(
            Ontology.Interval(0.0, 1.0), all_comps, all_lw)
        append!(state.metadata, new_meta)
        append!(state.compiled_kernels, new_ck)
        append!(state.all_programs, new_progs)
    end
    n_added
end

# ═══════════════════════════════════════
# Grammar ID counter
# ═══════════════════════════════════════

let grammar_counter = Ref(0)
    global function next_grammar_id()
        grammar_counter[] += 1
        grammar_counter[]
    end
    global function reset_grammar_counter!()
        grammar_counter[] = 0
    end
end

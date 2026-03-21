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

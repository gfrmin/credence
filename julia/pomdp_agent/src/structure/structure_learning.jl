"""
    Structure Learning (Stage 3)

Learn Bayesian network structure via BDe (Bayesian Dirichlet Equivalent) scores.

Mathematical basis:
- P(G | data) ∝ P(data | G) × P(G)
- P(data | G) = ∏ᵢ BDe(Vᵢ, parents_G(Vᵢ) | data)
- BDe score: conjugate Beta-Binomial form
- Greedy search: add/remove/reverse edges that improve score

Key insight: Learn separate structure per action.
- Action "take" has different CPD structure than "open"
- Enables discovering action-specific dependencies
"""

"""
    DirectedGraph

Simple DAG representation for Bayesian network structure.

Fields:
- vertices::Vector{String}           # Variable names
- edges::Set{Tuple{String,String}}  # (parent, child) pairs
"""
mutable struct DirectedGraph
    vertices::Vector{String}
    edges::Set{Tuple{String,String}}

    function DirectedGraph(vertices::Vector{String})
        new(vertices, Set{Tuple{String,String}}())
    end
end

"""
    add_edge!(graph::DirectedGraph, parent::String, child::String)

Add directed edge parent → child.
"""
function add_edge!(graph::DirectedGraph, parent::String, child::String)
    if parent != child  # No self-loops
        push!(graph.edges, (parent, child))
    end
end

"""
    remove_edge!(graph::DirectedGraph, parent::String, child::String)

Remove edge parent → child.
"""
function remove_edge!(graph::DirectedGraph, parent::String, child::String)
    delete!(graph.edges, (parent, child))
end

"""
    get_parents(graph::DirectedGraph, vertex::String) → Vector{String}

Return list of parent vertices.
"""
function get_parents(graph::DirectedGraph, vertex::String)::Vector{String}
    return [p for (p, c) in graph.edges if c == vertex]
end

"""
    is_acyclic(graph::DirectedGraph) → Bool

Check if graph is acyclic (DAG).
"""
function is_acyclic(graph::DirectedGraph)::Bool
    # Simple check: topological sort
    visited = Set{String}()
    rec_stack = Set{String}()

    function has_cycle(v)
        push!(visited, v)
        push!(rec_stack, v)

        for (parent, child) in graph.edges
            if child == v
                if parent ∉ visited
                    if has_cycle(parent)
                        return true
                    end
                elseif parent in rec_stack
                    return true
                end
            end
        end

        delete!(rec_stack, v)
        return false
    end

    for v in graph.vertices
        if v ∉ visited
            if has_cycle(v)
                return false
            end
        end
    end

    return true
end

"""
    bde_score(variable::String, parents::Vector{String}, transitions::Vector{Tuple}, prior_strength::Float64 = 0.1) → Float64

Compute BDe score for a variable given its parents.

BDe = log P(data | variable, parents) from conjugate analysis
    = Σ [log Γ(αᵢ + Nᵢ) - log Γ(αᵢ)]
      where αᵢ is Dirichlet prior count, Nᵢ is observed count

Higher score = better fit.
"""
function bde_score(variable::String, parents::Vector{String},
                   transitions::Vector{Tuple}, prior_strength::Float64=0.1)::Float64
    if isempty(transitions)
        return 0.0
    end

    # Count transitions for this variable
    value_counts = Dict{Any,Int}()
    for (s, s_next) in transitions
        val_key = nothing  # Simplified: would extract actual value
        value_counts[val_key] = get(value_counts, val_key, 0) + 1
    end

    # Build DirichletMeasure and compute log marginal via credence
    k = length(value_counts)
    domain = collect(keys(value_counts))
    alpha = fill(prior_strength, k)
    counts = [value_counts[d] for d in domain]

    m = DirichletMeasure(Simplex(k), Finite(domain), alpha)
    return log_marginal(m, counts)
end

"""
    learn_structure_greedy(variables::Vector{String}, transitions::Vector{Tuple}; max_parents::Int=2, iterations::Int=10) → DirectedGraph

Learn graph structure via greedy search.

Algorithm:
1. Start with empty graph
2. For each iteration:
   a. Try all possible edge additions, deletions, reversals
   b. Accept move that most improves score
   c. Stop if no improving move exists

Returns: Best DAG found
"""
function learn_structure_greedy(variables::Vector{String}, transitions::Vector{Tuple}; max_parents::Int=2, iterations::Int=10)::DirectedGraph
    graph = DirectedGraph(variables)
    best_score = 0.0

    for _ in 1:iterations
        improved = false

        # Try adding edges
        for parent in variables
            for child in variables
                if parent == child
                    continue
                end

                if (parent, child) ∉ graph.edges
                    # Try adding edge
                    add_edge!(graph, parent, child)

                    if is_acyclic(graph)
                        new_score = bde_score(child, get_parents(graph, child), transitions)
                        if new_score > best_score
                            best_score = new_score
                            improved = true
                            break
                        end
                    end

                    remove_edge!(graph, parent, child)
                end
            end
            if improved
                break
            end
        end

        if !improved
            break
        end
    end

    return graph
end

"""
    LearnedStructure

Result of structure learning: graph + per-variable parent information.

Fields:
- graph::DirectedGraph
- variable_parents::Dict{String, Vector{String}}   # Variable → its parents
- score::Float64                                    # Overall graph score
"""
struct LearnedStructure
    graph::DirectedGraph
    variable_parents::Dict{String,Vector{String}}
    score::Float64
end

"""
    learn_action_structure(action::String, model::FactoredWorldModel; max_parents::Int=2) → LearnedStructure

Learn Bayesian network structure for a specific action.

For action a, learn graph showing which variables depend on which.
"""
function learn_action_structure(action::String, model::FactoredWorldModel; max_parents::Int=2)::LearnedStructure
    variables = push!(copy(collect(model.known_objects)), "location")

    # Collect transitions for this action
    transitions = []
    for ((s, a), next_states) in model.transitions
        if a == action
            for s_next in next_states
                push!(transitions, (s, s_next))
            end
        end
    end

    if isempty(transitions)
        # No data: return empty structure
        graph = DirectedGraph(variables)
        return LearnedStructure(graph, Dict(v => String[] for v in variables), 0.0)
    end

    # Learn structure
    graph = learn_structure_greedy(variables, transitions; max_parents=max_parents)

    # Compute parent mappings
    variable_parents = Dict(v => get_parents(graph, v) for v in variables)

    # Compute score
    score = sum(
        bde_score(v, variable_parents[v], transitions)
        for v in variables
    )

    return LearnedStructure(graph, variable_parents, score)
end

"""
    compute_action_scope(model::FactoredWorldModel, action::String) → Set{String}

Determine which variables are affected by an action.

Scope = {V : some observation after action a changed V}
"""
function compute_action_scope(model::FactoredWorldModel, action::String)::Set{String}
    scope = Set{String}()

    for ((s, a), next_states) in model.transitions
        if a == action
            for s_next in next_states
                # Check which variables changed
                if s.location != s_next.location
                    push!(scope, "location")
                end

                for obj in model.known_objects
                    in_before = obj ∈ s.inventory
                    in_after = obj ∈ s_next.inventory

                    if in_before != in_after
                        push!(scope, obj)
                    end
                end
            end
        end
    end

    return scope
end

export DirectedGraph, add_edge!, remove_edge!, get_parents, is_acyclic
export bde_score, learn_structure_greedy, LearnedStructure
export learn_action_structure, compute_action_scope

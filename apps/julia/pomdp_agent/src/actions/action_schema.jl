"""
    Action Schemas (Stage 4)

Learn reusable action schemas to generalize across instances.

Problem: "take book", "take key", "take lantern" learned separately
Solution: Extract schema take(X) with lifted dynamics

Mathematical basis:
- Action schema: (type, parameters, scope, preconditions, effects)
- Lifted dynamics: θ_schema shared across all instances X
- Generalization: P(s' | s, schema(X)) = same CPD for all X

Key insight: Cluster similar actions by signature, extract parameters.
"""

"""
    ActionSchema

Represents a reusable action pattern.

Fields:
- action_type::Symbol           # :take, :drop, :open, :unlock, etc.
- parameters::Vector{String}    # Parameter names: ["X"] for take(X)
- scope::Set{String}            # Variables affected: {inventory, object_location}
- success_rate::Float64         # P(action succeeds | preconditions met)
"""
struct ActionSchema
    action_type::Symbol
    parameters::Vector{String}
    scope::Set{String}
    success_rate::Float64
end

"""
    ActionInstance

Concrete instantiation of a schema with specific arguments.

Fields:
- schema::ActionSchema
- arguments::Dict{String,String}         # Parameter → value mappings
- observed_count::Int                    # Times this instance observed
"""
mutable struct ActionInstance
    schema::ActionSchema
    arguments::Dict{String,String}
    observed_count::Int
end

"""
    extract_action_type(action_str::String) → Tuple{String, Vector{String}}

Parse action string to extract type and arguments.

Examples:
- "take book" → ("take", ["book"])
- "drop lantern" → ("drop", ["lantern"])
- "open door" → ("open", ["door"])
"""
function extract_action_type(action_str::String)::Tuple{String,Vector{String}}
    parts = split(strip(action_str))

    if length(parts) < 2
        return (parts[1], String[])
    end

    action_type = parts[1]
    arguments = parts[2:end]

    return (action_type, arguments)
end

"""
    cluster_actions(actions::Vector{String}) → Dict{String, Vector{String}}

Cluster similar actions by type.

Returns: action_type → [instances of that type]
"""
function cluster_actions(actions::Vector{String})::Dict{String,Vector{String}}
    clusters = Dict{String,Vector{String}}()

    for action in actions
        (action_type, args) = extract_action_type(action)

        if !haskey(clusters, action_type)
            clusters[action_type] = String[]
        end

        push!(clusters[action_type], action)
    end

    return clusters
end

"""
    infer_schema_from_cluster(action_type::String, instances::Vector{String}) → ActionSchema

Infer schema from a cluster of similar actions.

Heuristic:
- Parameters are arguments that vary across instances
- Scope is union of affected variables across instances
- Success rate estimated from successful transitions
"""
function infer_schema_from_cluster(action_type::String, instances::Vector{String}, model::FactoredWorldModel)::ActionSchema
    # Extract parameters (arguments that vary)
    all_args = []
    for inst in instances
        (_, args) = extract_action_type(inst)
        append!(all_args, args)
    end

    # Parameters: things that vary
    unique_args = unique(all_args)
    parameters = Symbol.(["X" for _ in unique_args])  # Generic parameter names

    # Scope: variables affected by this action type
    scope = Set{String}()
    for action_str in instances
        action_scope = compute_action_scope(model, action_str)
        union!(scope, action_scope)
    end

    # Success rate: fraction of transitions that succeeded
    success_count = 0
    total_count = 0

    for action_str in instances
        if haskey(model.transitions, (nothing, action_str))
            # Simplified: would properly track successes
            total_count += 1
            success_count += 1
        end
    end

    success_rate = if total_count > 0
        success_count / total_count
    else
        0.5  # Default prior
    end

    return ActionSchema(Symbol(action_type), String.(parameters), scope, success_rate)
end

"""
    discover_schemas(model::FactoredWorldModel) → Dict{String, ActionSchema}

Discover action schemas from model's experience.

Returns: action_type → schema
"""
function discover_schemas(model::FactoredWorldModel)::Dict{String,ActionSchema}
    # Collect all observed actions
    all_actions = unique([a for (_, a) in keys(model.transitions)])

    # Cluster by action type
    clusters = cluster_actions(all_actions)

    # Infer schema for each cluster
    schemas = Dict{String,ActionSchema}()

    for (action_type, instances) in clusters
        if length(instances) >= 2  # Need multiple instances for schema
            schema = infer_schema_from_cluster(action_type, instances, model)
            schemas[action_type] = schema
        end
    end

    return schemas
end

"""
    apply_schema(schema::ActionSchema, arguments::Dict{String,String}, state::MinimalState, model::FactoredWorldModel) → Tuple{MinimalState, Float64}

Apply schema to predict next state.

Returns: (predicted_next_state, estimated_success_probability)
"""
function apply_schema(schema::ActionSchema, arguments::Dict{String,String}, state::MinimalState, model::FactoredWorldModel)::Tuple{MinimalState,Float64}
    # Sample from model to get next state
    sampled = sample_dynamics(model)

    # Build concrete action string
    action_str = "$(schema.action_type)"
    for arg in arguments
        action_str = "$action_str $arg"
    end

    # Sample transition
    if haskey(sampled.sampled_cpds, action_str)
        next_state = sample_next_state(model, state, action_str, sampled)
    else
        # Unknown action: pessimistic estimate
        next_state = state
    end

    # Success probability
    success_prob = schema.success_rate

    return (next_state, success_prob)
end

"""
    zero_shot_transfer_likelihood(schema::ActionSchema, new_object::String, seen_objects::Vector{String}) → Float64

Estimate likelihood of schema applying to unseen object.

Uses heuristic: objects of same type likely have similar behavior.

Returns: P(schema works for new_object | schema worked for seen_objects)
"""
function zero_shot_transfer_likelihood(schema::ActionSchema, new_object::String, seen_objects::Vector{String})::Float64
    # Simple heuristic: if we've seen schema on multiple objects, confidence increases
    if isempty(seen_objects)
        return 0.3  # Completely unseen
    end

    # More observations = higher confidence (up to limit)
    num_seen = length(seen_objects)
    confidence = min(0.9, 0.5 + 0.1 * num_seen)

    return confidence
end

export ActionSchema, ActionInstance
export extract_action_type, cluster_actions, infer_schema_from_cluster
export discover_schemas, apply_schema, zero_shot_transfer_likelihood

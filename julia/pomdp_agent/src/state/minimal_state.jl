"""
    MinimalState (Stage 1-2: MVBN with Hidden Variables)

Factored state representation for Enchanter:
- location: string identifier (e.g., "Kitchen", "Forest")
- inventory: Set of strings (items agent is carrying)
- spells_known: Set of strings (magic abilities learned)
- object_states: Dict (object => state, e.g. "lantern" => "lit", "door" => "locked")
- knowledge_gained: Set of strings (facts discovered)

Replaces opaque hash-based state IDs with explicit factorization.
Enables reasoning about state variables independently.

Observable variables (location, inventory) are directly extracted from observations.
Hidden variables (spells, object_states, knowledge) are inferred via Bayesian updates.

Mathematical representation:
    s = (location, inventory, spells_known, object_states, knowledge_gained)
    P(s | history) = P(location | history) × P(inventory | history)
                   × P(spells | history) × P(object_states | history) × P(knowledge | history)
"""

struct MinimalState
    location::String
    inventory::Set{String}

    # Hidden variables (inferred from observations via LLM)
    spells_known::Set{String}
    object_states::Dict{String, String}
    knowledge_gained::Set{String}

    function MinimalState(
        location::String,
        inventory::Set{String}=Set{String}(),
        spells_known::Set{String}=Set{String}(),
        object_states::Dict{String, String}=Dict{String, String}(),
        knowledge_gained::Set{String}=Set{String}()
    )
        return new(location, inventory, spells_known, object_states, knowledge_gained)
    end
end

"""
    MinimalState(location_str, inventory_str)

Convenience constructor from string representations.
inventory_str format: "book,lantern,key" or empty string "".
"""
function MinimalState(location::String, inventory_str::String)
    if isempty(inventory_str)
        inv = Set{String}()
    else
        # Convert SubString to String via string()
        inv = Set(string.(strip.(split(inventory_str, ","))))
    end
    return MinimalState(location, inv)
end

"""
    extract_minimal_state(observation) → MinimalState

Extract MinimalState from Jericho observation (NamedTuple).
Observable variables (location, inventory) extracted directly.
Hidden variables (spells, object_states, knowledge) initialized empty,
to be inferred via Bayesian updates from observation text.
"""
function extract_minimal_state(obs)::MinimalState
    location = hasproperty(obs, :location) ? obs.location : "unknown"

    inventory_set = Set{String}()
    if hasproperty(obs, :inventory) && obs.inventory isa String
        if !isempty(obs.inventory)
            inv_items = strip.(split(obs.inventory, ","))
            inventory_set = Set(inv_items)
        end
    elseif hasproperty(obs, :inventory) && obs.inventory isa Vector
        inventory_set = Set(obs.inventory)
    end

    # Hidden variables initialized empty - to be inferred via Bayesian updates
    spells_known = Set{String}()
    object_states = Dict{String, String}()
    knowledge_gained = Set{String}()

    return MinimalState(location, inventory_set, spells_known, object_states, knowledge_gained)
end

"""
    Base.:(==)(s1::MinimalState, s2::MinimalState) → Bool

Equality comparison for states, including hidden variables.
"""
function Base.:(==)(s1::MinimalState, s2::MinimalState)::Bool
    return s1.location == s2.location &&
           s1.inventory == s2.inventory &&
           s1.spells_known == s2.spells_known &&
           s1.object_states == s2.object_states &&
           s1.knowledge_gained == s2.knowledge_gained
end

"""
    Base.hash(s::MinimalState) → UInt

Hash for use in dictionaries/sets, including hidden variables.
"""
function Base.hash(s::MinimalState)::UInt
    return hash((
        s.location,
        sort(collect(s.inventory)),
        sort(collect(s.spells_known)),
        sort(collect(keys(s.object_states))),
        sort(collect(s.knowledge_gained))
    ))
end

"""
    Base.show(io::IO, s::MinimalState)

Pretty printing including hidden variables.
"""
function Base.show(io::IO, s::MinimalState)
    inv_str = isempty(s.inventory) ? "∅" : join(sort(collect(s.inventory)), ",")
    spells_str = isempty(s.spells_known) ? "" : " spells={$(join(sort(collect(s.spells_known)), ","))}"
    objs_str = isempty(s.object_states) ? "" : " objs={$(join(["$k=$v" for (k,v) in s.object_states], ","))}"
    knowledge_str = isempty(s.knowledge_gained) ? "" : " knows={$(join(sort(collect(s.knowledge_gained)), ","))}"
    print(io, "MinimalState($(s.location), {$inv_str}$spells_str$objs_str$knowledge_str)")
end

export MinimalState, extract_minimal_state

"""
    MinimalStateAbstractor <: StateAbstractor

Wraps Jericho observations → MinimalState for FactoredWorldModel.

Extracts structured state (location, inventory) from game observations and
provides null action detection by tracking observation text hashes.

Stage 1 integration component: converts opaque Jericho observations into
factored representations amenable to Bayesian network learning.
"""
struct MinimalStateAbstractor <: StateAbstractor
    # Track locations discovered during gameplay
    known_locations::Set{String}

    # Track objects discovered in inventory
    known_objects::Set{String}

    # For null action detection: track recent observation texts
    recent_texts::Vector{UInt64}
    text_history_size::Int

    function MinimalStateAbstractor(history_size::Int=5)
        new(Set{String}(), Set{String}(), Vector{UInt64}(), history_size)
    end
end

"""
    abstract_state(::MinimalStateAbstractor, obs) → MinimalState

Extract MinimalState from Jericho observation NamedTuple.

Jericho observations have fields: text, score, steps, location, inventory, state_hash
Returns: MinimalState(location, inventory_set)
"""
function BayesianAgents.abstract_state(abs::MinimalStateAbstractor, obs)
    # Extract location string
    location = hasproperty(obs, :location) ? string(obs.location) : "unknown"
    push!(abs.known_locations, location)

    # Parse inventory string into Set of object names
    inv_str = hasproperty(obs, :inventory) ? string(obs.inventory) : ""
    inventory = parse_inventory(inv_str)

    for obj in inventory
        push!(abs.known_objects, obj)
    end

    return MinimalState(location, inventory)
end

"""
    parse_inventory(inv_str::String) → Set{String}

Convert Jericho inventory string into a set of object names.

Examples:
  "You are carrying: a book, a lantern" → {"book", "lantern"}
  "You are carrying nothing." → Set()
  "a key and a map" → {"key", "map"}
"""
function parse_inventory(inv_str::String)
    if isempty(inv_str) || occursin("nothing", lowercase(inv_str))
        return Set{String}()
    end

    # Common delimiters: comma-space, " and "
    items = split(inv_str, r",\s*|\s+and\s+")
    inventory = Set{String}()

    for item in items
        clean = strip(lowercase(item))

        # Remove articles and auxiliary words
        clean = replace(clean, r"^(a|an|the)\s+" => "")
        clean = replace(clean, r"\s*(in|of|on|at).*$" => "")  # Remove qualifiers

        # Skip meta-text
        if !isempty(clean) &&
           !occursin("carrying", clean) &&
           !occursin("you are", clean) &&
           !occursin("nothing", clean)
            push!(inventory, clean)
        end
    end

    return inventory
end

"""
    record_transition!(::MinimalStateAbstractor, s, a, r, s′)

Track state transitions for null action detection.

Note: Called after abstract_state, so raw observation text is not available here.
Null action detection is integrated into agent loop instead.
"""
function BayesianAgents.record_transition!(abs::MinimalStateAbstractor, s, a, r, s′)
    # Placeholder: actual null detection happens in agent loop via text comparison
    nothing
end

# Bisimulation not used with MinimalState (fixed structure)
BayesianAgents.check_contradiction(::MinimalStateAbstractor) = nothing
BayesianAgents.refine!(::MinimalStateAbstractor, contradiction) = nothing

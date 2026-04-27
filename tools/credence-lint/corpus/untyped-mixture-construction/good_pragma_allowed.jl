# Role: skin
# Deserialisation from JSON necessarily uses Any[] before type recovery.

using Credence

function deserialise_belief(data)
    components = Any[]
    for item in data["components"]
        push!(components, parse_component(item))
    end
    # credence-lint: allow — precedent:untyped-mixture-construction — JSON deserialisation recovers types at parse time
    belief = MixturePrevision(components, data["log_weights"])
    belief
end

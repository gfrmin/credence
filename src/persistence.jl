"""
    persistence.jl — Save/load agent state across sessions.

Uses Julia's Serialization stdlib to roundtrip ontology types
(CategoricalMeasure, BetaMeasure, etc.), score totals, and
configuration.

## Schema versioning (Posture 4 Move 3)

Files written by `save_state` include a `:__schema_version` marker.
`load_state` accepts v3 only. Earlier versions (v1 pre-Move-3; v2 from
Posture 3 Move 3) are retired — no migration path — per
`docs/posture-4/master-plan.md` §Move 3 and `docs/posture-4/decision-log.md`
§Decision 3 (no compatibility with a non-existent user base).

Future schema bumps (v4, v5, …) use the `:__schema_version` marker to
dispatch. Move 9's production-state persistence is the nearest expected
consumer — connection registries, program caches, etc. may introduce
fields that v3 doesn't carry and bump to v4.

When a future bump introduces a struct change that v3 state can't
survive, the migration is designed with the same policy that retired
v1 → v2 in Posture 3: either in-place via a custom deserialize hook
(if the change is local enough), or via a MigrationError with a clear
reinitialisation hint.
"""
module Persistence

using Serialization
using ..Ontology

export save_state, load_state, MigrationError

const SCHEMA_VERSION = 3

"""
    MigrationError(message::String)

Raised by `load_state` when the file's schema version cannot be
migrated in-place. The message names the detected version and the
required action (typically: reinitialise from scratch). Kept live even
in the v3-only tip as load-path vocabulary for "version I don't
understand" — Move 9's production persistence is the expected consumer.
"""
struct MigrationError <: Exception
    message::String
end

Base.showerror(io::IO, e::MigrationError) = print(io, "MigrationError: ", e.message)

"""
    save_state(filepath; rel_beliefs, cov_beliefs, cat_belief, total_score, total_cost)

Save agent state to `filepath` at schema version $(SCHEMA_VERSION). The
written Dict has `:__schema_version => $SCHEMA_VERSION` alongside the
payload.
"""
function save_state(filepath::String;
                    rel_beliefs, cov_beliefs, cat_belief,
                    total_score::Float64, total_cost::Float64)
    state = Dict(
        :__schema_version => SCHEMA_VERSION,
        :rel_beliefs => rel_beliefs,
        :cov_beliefs => cov_beliefs,
        :cat_belief  => cat_belief,
        :total_score => total_score,
        :total_cost  => total_cost,
    )
    open(io -> serialize(io, state), filepath, "w")
end

"""
    load_state(filepath) → Dict

Load agent state from `filepath`. Returns a Dict matching the `save_state`
signature (plus the `:__schema_version` key).

Raises `MigrationError` if the file's `:__schema_version` is not the
current version ($SCHEMA_VERSION), or if the file has no version
marker, or if deserialization fails because of a struct-layout mismatch
(e.g., a file written by an older Credence tip).
"""
function load_state(filepath::String)
    try
        state = open(deserialize, filepath, "r")
        if state isa Dict && get(state, :__schema_version, nothing) == SCHEMA_VERSION
            return state
        elseif state isa Dict
            v = get(state, :__schema_version, nothing)
            throw(MigrationError(
                "state file at $filepath has schema version " *
                "$(v === nothing ? "<missing>" : v); expected $SCHEMA_VERSION. " *
                "Pre-v$SCHEMA_VERSION state cannot be migrated in-place — " *
                "reinitialise state from scratch."
            ))
        else
            throw(MigrationError(
                "state file at $filepath did not deserialize to a Dict " *
                "(got $(typeof(state))); expected a v$SCHEMA_VERSION state Dict."
            ))
        end
    catch e
        if e isa MigrationError
            rethrow(e)
        elseif e isa TypeError || e isa MethodError
            # Struct-layout incompatibility — pre-v3 file, or a file from a
            # post-Move-3 tip that's since had a struct rename/field change.
            # Julia's Serialization is struct-layout-dependent; raise a
            # MigrationError rather than propagating the raw type error.
            throw(MigrationError(
                "state file at $filepath cannot be loaded with the current code. " *
                "Struct-layout mismatch — the file was likely written by an " *
                "incompatible Credence tip. Reinitialise state from scratch. " *
                "(Underlying error: $(sprint(showerror, e)))"
            ))
        else
            rethrow(e)
        end
    end
end

end # module Persistence

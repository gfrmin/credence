"""
    persistence.jl — Save/load agent state across sessions.

Uses Julia's Serialization stdlib to roundtrip ontology types
(CategoricalMeasure, BetaMeasure, etc.), score totals, and
configuration.

## Schema versioning (Move 3)

Files written by `save_state` include a `:__schema_version` marker.
`load_state` dispatches on the marker:

- **v2 (current):** `save_state` writes a Dict with `:__schema_version
  => 2` plus the payload keys. The Measure values inside are serialised
  in their current struct layout (Move-3-wrapped: `prevision::XPrevision
  + space::XSpace`). Julia's `Serialization` handles the round-trip.

- **v1 (pre-Move-3):** no version marker. The Measure values inside are
  the pre-Move-3 struct layout (e.g. `BetaMeasure(space::Interval,
  alpha::Float64, beta::Float64)` with three direct fields). Julia's
  `deserialize` on these bytes raises a `TypeError` because the new
  `BetaMeasure` layout (`prevision::BetaPrevision, space::Interval`)
  doesn't accept three positional Interval/Float64/Float64 arguments.

  `load_state` catches that TypeError and raises `MigrationError` with a
  clear message naming the v1 → v2 migration policy: v1 state cannot be
  migrated in-place (raw Julia Serialization is struct-layout-dependent,
  and a struct layout change is structurally incompatible). Users with
  v1 state reinitialize by rerunning their agent initialisation. This
  is an acceptable cost because `save_state` is only called from
  `examples/host_credence_agent.jl` — an experimental CLI example with
  a small user base.

Future schema bumps (v3, v4, …) use the `:__schema_version` marker to
dispatch. When a future bump introduces a struct change that v2 state
can't survive, that migration is designed with the same policy: either
in-place via a custom deserialize hook (if the change is local enough),
or via a MigrationError with a clear reinitialisation hint.
"""
module Persistence

using Serialization
using ..Ontology

export save_state, load_state, MigrationError

const SCHEMA_VERSION = 2

"""
    MigrationError(message::String)

Raised by `load_state` when the file's schema version cannot be
migrated in-place. The message names the detected version and the
required action (typically: reinitialise from scratch).
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

Raises `MigrationError` if the file is v1 (no schema marker; pre-Move-3
struct layout that the current code cannot deserialize in-place).
"""
function load_state(filepath::String)
    try
        state = open(deserialize, filepath, "r")
        if state isa Dict && get(state, :__schema_version, nothing) == SCHEMA_VERSION
            return state
        elseif state isa Dict && !haskey(state, :__schema_version)
            # Deserialize succeeded but no version marker — this shouldn't
            # happen in practice (v1 deserialize raises TypeError before
            # reaching here; a v2-written file always has the marker).
            # Defensive: treat as v1 and raise MigrationError.
            throw(MigrationError(
                "state file at $filepath has no :__schema_version marker; " *
                "likely a pre-Move-3 (v1) file that happened to deserialize. " *
                "Reinitialise state from scratch — v1 state cannot be migrated in-place."
            ))
        else
            throw(MigrationError(
                "state file at $filepath has unexpected schema version " *
                "$(get(state, :__schema_version, "<missing>")); expected $SCHEMA_VERSION. " *
                "This version is not recognised by load_state."
            ))
        end
    catch e
        if e isa MigrationError
            rethrow(e)
        elseif e isa TypeError || e isa MethodError
            # v1 files trigger TypeError because the pre-Move-3 struct
            # layout (e.g. BetaMeasure(space, alpha, beta)) doesn't match
            # the current layout (BetaMeasure(prevision, space)). Raise a
            # clear MigrationError rather than propagating the raw type
            # error.
            throw(MigrationError(
                "state file at $filepath cannot be loaded with the current code. " *
                "This is expected for v1 (pre-Move-3) state files — Julia's " *
                "Serialization is struct-layout-dependent and v1 used a different " *
                "Measure struct layout. Reinitialise state from scratch. " *
                "(Underlying error: $(sprint(showerror, e)))"
            ))
        else
            rethrow(e)
        end
    end
end

end # module Persistence

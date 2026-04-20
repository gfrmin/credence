# test_persistence.jl — Move 3 persistence schema v2 + v1 fixture migration path.
#
# Loads the commit-pinned v1 fixtures from `test/fixtures/` (captured on
# master at SHA bf74f98, pre-Move-3 struct layout) and asserts the
# migration-error path behaves correctly. Also verifies v2 round-trip
# (save → load → observe via shield) preserves the recorded values.
#
# v1 fixtures cannot be migrated in-place (empirically established:
# Julia's `deserialize` on v1 bytes raises TypeError because the Move-3
# struct layout (prevision::XPrevision + space::XSpace) doesn't accept
# the pre-Move-3 field sequence). `load_state` catches the TypeError and
# raises `MigrationError` with a clear reinitialisation hint. This test
# asserts that behaviour — v1 files produce MigrationError, not silent
# corruption or opaque type errors propagated to the caller.

push!(LOAD_PATH, "src")
using Credence
using Serialization

function check(name, cond, detail="")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("Persistence test assertion failed: $name")
    end
end

println("="^60)
println("Persistence — v2 round-trip")
println("="^60)

let path = tempname()
    # v2 save → load → observe
    rel_beliefs = MixtureMeasure(
        ProductSpace([Interval(0.0, 1.0), Interval(0.0, 1.0)]),
        Measure[ProductMeasure(Measure[BetaMeasure(Interval(0.0, 1.0), 2.0, 3.0),
                                        BetaMeasure(Interval(0.0, 1.0), 4.0, 1.0)])],
        [0.0],
    )
    cov_beliefs = MixtureMeasure(
        ProductSpace([Interval(0.0, 1.0), Interval(0.0, 1.0)]),
        Measure[ProductMeasure(Measure[BetaMeasure(Interval(0.0, 1.0), 2.0, 2.0),
                                        BetaMeasure(Interval(0.0, 1.0), 2.0, 2.0)])],
        [0.0],
    )
    cat_belief = CategoricalMeasure(Finite([:a, :b]), [log(3.0), log(7.0)])

    save_state(path;
               rel_beliefs = rel_beliefs,
               cov_beliefs = cov_beliefs,
               cat_belief  = cat_belief,
               total_score = 1.5,
               total_cost  = 0.25)

    loaded = load_state(path)
    check("v2 round-trip: :__schema_version == 2",
          loaded[:__schema_version] == 2,
          string("got ", get(loaded, :__schema_version, "<missing>")))
    check("v2 round-trip: rel_beliefs BetaMeasure alpha preserved (==)",
          loaded[:rel_beliefs].components[1].factors[1].alpha == 2.0,
          "got $(loaded[:rel_beliefs].components[1].factors[1].alpha)")
    check("v2 round-trip: rel_beliefs BetaMeasure beta preserved (==)",
          loaded[:rel_beliefs].components[1].factors[2].alpha == 4.0,
          "got $(loaded[:rel_beliefs].components[1].factors[2].alpha)")
    check("v2 round-trip: total_score preserved (==)",
          loaded[:total_score] == 1.5,
          "got $(loaded[:total_score])")
    check("v2 round-trip: cat_belief weights preserved (atol=1e-14)",
          isapprox(weights(loaded[:cat_belief]), [0.3, 0.7]; atol=1e-14),
          "got $(weights(loaded[:cat_belief]))")

    rm(path)
end

println()
println("="^60)
println("Persistence — v1 fixture migration path (MigrationError)")
println("="^60)

# Per docs/posture-3/move-3-design.md §6 R3 and the design-doc fixture
# protocol: v1 fixtures captured at master SHA bf74f98 can't be migrated
# in-place. The load path raises MigrationError naming the situation.
# This test exercises that error path directly.

let path = joinpath("test", "fixtures", "agent_state_v1.jls")
    check("agent_state_v1.jls fixture exists",
          isfile(path), "expected $path")
    try
        load_state(path)
        check("agent_state_v1.jls load raises MigrationError", false,
              "load_state did NOT raise; v1 fixture somehow loaded cleanly — investigate")
    catch e
        check("agent_state_v1.jls load raises MigrationError (not TypeError)",
              e isa Credence.Persistence.MigrationError,
              "got $(typeof(e)): $(sprint(showerror, e))")
        msg = sprint(showerror, e)
        check("MigrationError message names the reinitialise path",
              occursin("Reinitialise", msg) || occursin("reinitialise", msg),
              "message: $msg")
    end
end

let path = joinpath("test", "fixtures", "email_agent_state_v1.jls")
    check("email_agent_state_v1.jls fixture exists",
          isfile(path), "expected $path")
    try
        load_state(path)
        check("email_agent_state_v1.jls load raises MigrationError", false,
              "load_state did NOT raise; v1 fixture somehow loaded cleanly — investigate")
    catch e
        check("email_agent_state_v1.jls load raises MigrationError (not TypeError)",
              e isa Credence.Persistence.MigrationError,
              "got $(typeof(e)): $(sprint(showerror, e))")
    end
end

println()
println("="^60)
println("ALL PERSISTENCE TESTS PASSED")
println("="^60)

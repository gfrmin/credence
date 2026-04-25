# test_persistence.jl — Posture 4 Move 3 persistence schema v3.
#
# Verifies v3 round-trip (save → load → observe via shield) preserves
# recorded values, and that load_state raises MigrationError on
# unknown-version files (e.g. a bogus v99 header). v1 and v2 paths are
# retired in Move 3; their fixtures are deleted; their MigrationError
# exercises are subsumed by the unknown-version test below.
#
# The `__schema_version` marker + MigrationError type stay in place as
# load-path vocabulary for "version I don't understand". Move 9's
# production-state persistence is the expected next consumer when it
# bumps to v4.

push!(LOAD_PATH, "src")
using Credence
using Credence: BetaPrevision, CategoricalPrevision  # Posture 4 Move 4
using Credence.Ontology: wrap_in_measure  # Posture 4 Move 4
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
println("Persistence — v3 round-trip")
println("="^60)

let path = tempname()
    # v3 save → load → observe
    rel_beliefs = MixtureMeasure(
        ProductSpace([Interval(0.0, 1.0), Interval(0.0, 1.0)]),
        Measure[ProductMeasure(Measure[wrap_in_measure(BetaPrevision(2.0, 3.0)),
                                        wrap_in_measure(BetaPrevision(4.0, 1.0))])],
        [0.0],
    )
    cov_beliefs = MixtureMeasure(
        ProductSpace([Interval(0.0, 1.0), Interval(0.0, 1.0)]),
        Measure[ProductMeasure(Measure[wrap_in_measure(BetaPrevision(2.0, 2.0)),
                                        wrap_in_measure(BetaPrevision(2.0, 2.0))])],
        [0.0],
    )
    cat_belief = CategoricalMeasure(Finite([:a, :b]), CategoricalPrevision([log(3.0), log(7.0)]))

    save_state(path;
               rel_beliefs = rel_beliefs,
               cov_beliefs = cov_beliefs,
               cat_belief  = cat_belief,
               total_score = 1.5,
               total_cost  = 0.25)

    loaded = load_state(path)
    check("v3 round-trip: :__schema_version == 3",
          loaded[:__schema_version] == 3,
          string("got ", get(loaded, :__schema_version, "<missing>")))
    check("v3 round-trip: rel_beliefs BetaMeasure alpha preserved (==)",
          loaded[:rel_beliefs].components[1].factors[1].alpha == 2.0,
          "got $(loaded[:rel_beliefs].components[1].factors[1].alpha)")
    check("v3 round-trip: rel_beliefs second factor alpha preserved (==)",
          loaded[:rel_beliefs].components[1].factors[2].alpha == 4.0,
          "got $(loaded[:rel_beliefs].components[1].factors[2].alpha)")
    check("v3 round-trip: total_score preserved (==)",
          loaded[:total_score] == 1.5,
          "got $(loaded[:total_score])")
    check("v3 round-trip: cat_belief weights preserved (atol=1e-14)",
          isapprox(weights(loaded[:cat_belief]), [0.3, 0.7]; atol=1e-14),
          "got $(weights(loaded[:cat_belief]))")

    rm(path)
end

println()
println("="^60)
println("Persistence — v3 fixture load")
println("="^60)

let path = joinpath("test", "fixtures", "agent_state_v3.jls")
    check("agent_state_v3.jls fixture exists",
          isfile(path), "expected $path")
    loaded = load_state(path)
    check("agent_state_v3.jls :__schema_version == 3",
          loaded[:__schema_version] == 3,
          "got $(loaded[:__schema_version])")
    check("agent_state_v3.jls has :belief field",
          haskey(loaded, :belief), "expected :belief key")
    check("agent_state_v3.jls :belief is MixtureMeasure",
          loaded[:belief] isa MixtureMeasure,
          "got $(typeof(loaded[:belief]))")
    # Direct field access — v3 fixture wrote :belief + :note, not the
    # save_state payload keys. That's allowed; the round-trip test above
    # covers save_state/load_state symmetry. This block verifies the
    # committed fixture file is well-formed v3.
end

let path = joinpath("test", "fixtures", "email_agent_state_v3.jls")
    check("email_agent_state_v3.jls fixture exists",
          isfile(path), "expected $path")
    loaded = load_state(path)
    check("email_agent_state_v3.jls :__schema_version == 3",
          loaded[:__schema_version] == 3,
          "got $(loaded[:__schema_version])")
    # Captured via save_state, so it has the full payload shape.
    check("email_agent_state_v3.jls :rel_beliefs BetaMeasure alpha preserved",
          loaded[:rel_beliefs].components[1].factors[1].alpha == 3.0,
          "got $(loaded[:rel_beliefs].components[1].factors[1].alpha)")
    check("email_agent_state_v3.jls :total_score preserved",
          loaded[:total_score] == 42.5,
          "got $(loaded[:total_score])")
end

println()
println("="^60)
println("Persistence — unknown-version MigrationError path")
println("="^60)

# The __schema_version + MigrationError machinery stays live in Move 3
# for Move 9 production persistence (v4) to use. Exercise the path here
# so the vocabulary remains covered: write a bogus v99 header and
# assert load_state raises MigrationError naming the expected version.

let path = tempname()
    bogus = Dict(
        :__schema_version => 99,
        :rel_beliefs => nothing,
        :cov_beliefs => nothing,
        :cat_belief  => nothing,
        :total_score => 0.0,
        :total_cost  => 0.0,
    )
    open(io -> serialize(io, bogus), path, "w")

    try
        load_state(path)
        check("unknown-version file raises MigrationError", false,
              "load_state accepted the v99 file; MigrationError machinery is not live")
    catch e
        check("unknown-version file raises MigrationError (not TypeError)",
              e isa Credence.Persistence.MigrationError,
              "got $(typeof(e)): $(sprint(showerror, e))")
        msg = sprint(showerror, e)
        check("MigrationError message names the expected schema version",
              occursin("expected 3", msg),
              "message: $msg")
    end

    rm(path)
end

# Missing-version-marker file also raises MigrationError
let path = tempname()
    bogus_no_version = Dict(:some_key => "no version marker here")
    open(io -> serialize(io, bogus_no_version), path, "w")

    try
        load_state(path)
        check("missing-version file raises MigrationError", false,
              "load_state accepted a file without :__schema_version")
    catch e
        check("missing-version file raises MigrationError",
              e isa Credence.Persistence.MigrationError,
              "got $(typeof(e))")
    end

    rm(path)
end

println()
println("="^60)
println("ALL PERSISTENCE TESTS PASSED")
println("="^60)

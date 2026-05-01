# Role: smoke test for the tool_decider DSL program.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
using Credence
using Test

const BDSL_PATH = joinpath(@__DIR__, "tool_decider.bdsl")
const DSL_ENV = load_dsl(read(BDSL_PATH, String))
const DECIDE_ACTION = DSL_ENV[Symbol("decide-action")]

@testset "decide-action" begin
    # Action layout (host-supplied EUs): [execute, substitute, stop, ask]
    # Pick highest EU if VOI of asking is below threshold.
    @test DECIDE_ACTION([0.9, 0.5, 0.1, 0.0], 0.0, 0.05) == 0  # execute
    @test DECIDE_ACTION([0.2, 0.6, 0.1, 0.0], 0.0, 0.05) == 1  # substitute
    @test DECIDE_ACTION([0.1, 0.05, 0.4, 0.0], 0.0, 0.05) == 2 # stop

    # When VOI(ask) - cost(ask) > best non-ask EU, pick ask (index 3).
    @test DECIDE_ACTION([0.3, 0.2, 0.1, 0.0], 0.5, 0.05) == 3
end

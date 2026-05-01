# Role: brain-side loader exposing the DSL decide-action callable.
"""
    host.jl — load tool_decider.bdsl and expose the decide-action symbol.

Used by the Python juliacall bridge in
`apps/python/credence_router/src/credence_router/tool_decision/decide.py`.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
using Credence

const BDSL_PATH = joinpath(@__DIR__, "tool_decider.bdsl")
const DSL_ENV = load_dsl(read(BDSL_PATH, String))
const DECIDE_ACTION = DSL_ENV[Symbol("decide-action")]

"""
    decide_action(action_eus::Vector{Float64}, voi_ask::Float64, ask_cost::Float64)::Int

Returns the chosen action index (0-3).
"""
function decide_action(action_eus::Vector{Float64}, voi_ask::Float64, ask_cost::Float64)::Int
    return DECIDE_ACTION(action_eus, voi_ask, ask_cost)
end

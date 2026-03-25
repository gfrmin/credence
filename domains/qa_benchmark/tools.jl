"""
    tools.jl — Simulated tools with category-dependent reliability and coverage.

Each tool is a NamedTuple. query_tool is a pure function, deterministic given RNG.
"""

const CATEGORIES = ("factual", "numerical", "recent_events", "misconceptions", "reasoning")

struct SimulatedTool
    name::String
    cost::Float64
    reliability_by_category::Dict{String,Float64}
    coverage_by_category::Dict{String,Float64}
end

"""
    query_tool(tool, question, rng) → candidate_idx::Int or nothing

1. Check coverage — if not covered, return nothing.
2. If covered and rng < reliability → correct answer (question.correct_index).
3. Otherwise → uniform random wrong answer from the 3 incorrect candidates.
"""
function query_tool(tool::SimulatedTool, question, rng)
    category = question.category
    coverage = get(tool.coverage_by_category, category, 0.0)

    if rand(rng) >= coverage
        return nothing
    end

    reliability = get(tool.reliability_by_category, category, 0.0)
    if rand(rng) < reliability
        return question.correct_index
    end

    # Wrong answer: uniform over the 3 incorrect candidates
    wrong = [i for i in 0:3 if i != question.correct_index]
    return wrong[rand(rng, 1:3)]
end

function make_spec_tools()
    all_covered = Dict(c => 1.0 for c in CATEGORIES)

    tool_a = SimulatedTool("quick_search", 1.0,
        Dict("factual" => 0.70, "numerical" => 0.20, "recent_events" => 0.65,
             "misconceptions" => 0.25, "reasoning" => 0.40),
        Dict(all_covered))

    tool_b = SimulatedTool("knowledge_base", 2.0,
        Dict("factual" => 0.92, "numerical" => 0.40, "recent_events" => 0.55,
             "misconceptions" => 0.88, "reasoning" => 0.45),
        Dict("factual" => 0.65, "numerical" => 0.30, "recent_events" => 0.35,
             "misconceptions" => 0.55, "reasoning" => 0.20))

    tool_c = SimulatedTool("calculator", 1.0,
        Dict("factual" => 0.0, "numerical" => 1.0, "recent_events" => 0.0,
             "misconceptions" => 0.0, "reasoning" => 0.0),
        Dict("factual" => 0.0, "numerical" => 1.0, "recent_events" => 0.0,
             "misconceptions" => 0.0, "reasoning" => 0.0))

    tool_d = SimulatedTool("llm_direct", 2.0,
        Dict("factual" => 0.65, "numerical" => 0.50, "recent_events" => 0.45,
             "misconceptions" => 0.40, "reasoning" => 0.72),
        Dict(all_covered))

    [tool_a, tool_b, tool_c, tool_d]
end

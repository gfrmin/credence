#!/usr/bin/env julia

const REPO_ROOT = dirname(dirname(@__DIR__))
push!(LOAD_PATH, REPO_ROOT)
using Credence
using JSON3

const SIDECAR_DIR = joinpath(REPO_ROOT, "apps", "credence-governance-sidecar")
include(joinpath(SIDECAR_DIR, "instruction_patterns.jl"))
include(joinpath(SIDECAR_DIR, "brain.jl"))
include(joinpath(SIDECAR_DIR, "detectors.jl"))
include(joinpath(SIDECAR_DIR, "persistence.jl"))

const CONFIG_PATH = joinpath(SIDECAR_DIR, "config", "budgets.json")
const READ_TOOLS_PATH = joinpath(SIDECAR_DIR, "config", "read_tools.json")
const SCENARIOS_DIR = joinpath(@__DIR__, "scenarios")
const EU_WINDOW_SIZE = DEFAULT_EU_WINDOW_SIZE

# ── Scenario loading ──

struct ScenarioStep
    type::String
    tool_name::String
    params::Dict{String,Any}
    error::Union{Nothing,String}
    duration_ms::Union{Nothing,Int}
    messages::Vector{Dict{String,Any}}
end

struct ExpectedOutcome
    turn::Union{Nothing,Int}
    type::String
    expected_decision::String
    detector::String
    description::String
end

struct Scenario
    name::String
    source::String
    description::String
    steps::Vector{ScenarioStep}
    expected_outcomes::Vector{ExpectedOutcome}
end

function load_scenario(path::String)::Scenario
    data = JSON3.read(read(path, String), Dict{String,Any})

    steps = ScenarioStep[]
    for s in data["steps"]
        step_type = s["type"]
        tool_name = get(s, "toolName", "")
        raw_params = get(s, "params", Dict{String,Any}())
        params = Dict{String,Any}(string(k) => v for (k, v) in raw_params)
        err = get(s, "error", nothing)
        err = err === nothing ? nothing : string(err)
        dur = get(s, "durationMs", nothing)
        dur = dur === nothing ? nothing : Int(dur)
        msgs = Dict{String,Any}[]
        for m in get(s, "messages", Any[])
            push!(msgs, Dict{String,Any}(string(k) => v for (k, v) in m))
        end
        push!(steps, ScenarioStep(step_type, tool_name, params, err, dur, msgs))
    end

    outcomes = ExpectedOutcome[]
    for o in get(data, "expected_outcomes", Any[])
        push!(outcomes, ExpectedOutcome(
            get(o, "turn", nothing),
            get(o, "type", "evaluate"),
            get(o, "expected_decision", ""),
            get(o, "detector", ""),
            get(o, "description", ""),
        ))
    end

    Scenario(data["name"], data["source"], data["description"], steps, outcomes)
end

# ── Step execution ──

struct StepResult
    step_number::Int
    step_type::String
    tool_name::String
    category::String
    decision::String
    action::String
    reason::String
    detector::String
    alpha::Float64
    beta::Float64
    eu_proceed::Float64
    registered_instructions::Vector{String}
end

function execute_step!(brain::BrainState, read_tools::Set{String}, bt::BudgetTable,
                       step::ScenarioStep, step_number::Int)::StepResult
    if step.type == "observe"
        category = infer_category(step.tool_name, step.params)
        success = classify_outcome(bt, step.tool_name, category, step.error, step.duration_ms)
        update_posterior!(brain, step.tool_name, category, success)
        record_outcome!(brain, step.tool_name, step.params, success)
        m = get_posterior(brain, step.tool_name, category)
        return StepResult(step_number, "observe", step.tool_name, category,
                          success ? "success" : "failure", "", "", "",
                          m.alpha, m.beta, 0.0, String[])

    elseif step.type == "evaluate"
        category = infer_category(step.tool_name, step.params)
        sr = check_stationarity(brain, read_tools, step.tool_name, step.params, category)
        if sr !== nothing && sr.fires
            m = get_posterior(brain, step.tool_name, category)
            return StepResult(step_number, "evaluate", step.tool_name, category,
                              "halt", "block",
                              "Posterior stationary on repeated tool call: '$(sr.tool_name)' " *
                              "called $(sr.count) times with identical arguments " *
                              "(KL=$(round(sr.kl, digits=4)) < threshold=$(round(sr.threshold, digits=4)), " *
                              "window=$(sr.window_size))",
                              "#34574", m.alpha, m.beta, 0.0, String[])
        end

        result = compute_eu(brain, step.tool_name, category)
        nc = check_no_confidence(brain, step.tool_name, category, result.eu_proceed, EU_WINDOW_SIZE)
        if nc
            m = get_posterior(brain, step.tool_name, category)
            return StepResult(step_number, "evaluate", step.tool_name, category,
                              "halt", "block",
                              "Posterior over next-action value is flat: EU(proceed) has high " *
                              "coefficient of variation across recent evaluations",
                              "#65550", m.alpha, m.beta, result.eu_proceed, String[])
        end

        m = get_posterior(brain, step.tool_name, category)
        has_instruction = result.decision == "escalate" &&
            any(inst -> instruction_matches_category(string(inst["action_class"]), category),
                brain.registered_instructions)
        detector = has_instruction ? "#1084" : ""
        return StepResult(step_number, "evaluate", step.tool_name, category,
                          result.decision, result.action, result.reason,
                          detector, m.alpha, m.beta, result.eu_proceed, String[])

    elseif step.type == "compaction_preview"
        matches = match_instructions(step.messages)
        registered = String[]
        for (matched_text, action_class) in matches
            for pattern in INSTRUCTION_PATTERNS
                if pattern.action_class == action_class && match(pattern.regex, matched_text) !== nothing
                    if register_instruction!(brain, pattern.id, action_class)
                        push!(registered, "$(pattern.id):$(action_class)")
                    end
                    break
                end
            end
        end
        return StepResult(step_number, "compaction_preview", "", "", "ok", "", "",
                          "", 0.0, 0.0, 0.0, registered)
    else
        error("Unknown step type: $(step.type)")
    end
end

# ── Scenario runner ──

function format_result(r::StepResult)::String
    if r.step_type == "observe"
        "  Step $(r.step_number) [observe]  $(r.tool_name) ($(r.category)) → $(r.decision)  " *
        "posterior=Beta($(round(r.alpha, digits=1)), $(round(r.beta, digits=1)))"
    elseif r.step_type == "evaluate"
        extra = r.detector != "" ? "  detector=$(r.detector)" : ""
        "  Step $(r.step_number) [evaluate] $(r.tool_name) ($(r.category)) → decision=$(r.decision)" *
        "  posterior=Beta($(round(r.alpha, digits=1)), $(round(r.beta, digits=1)))" *
        "  EU(proceed)=$(round(r.eu_proceed, digits=4))$(extra)"
    elseif r.step_type == "compaction_preview"
        if isempty(r.registered_instructions)
            "  Step $(r.step_number) [compaction] no instructions matched"
        else
            "  Step $(r.step_number) [compaction] registered: $(join(r.registered_instructions, ", "))"
        end
    else
        "  Step $(r.step_number) [$(r.step_type)]"
    end
end

function run_scenario(scenario::Scenario, bt::BudgetTable, read_tools::Set{String})::Tuple{Bool, Vector{StepResult}}
    brain = make_brain_state(bt)
    results = StepResult[]

    for (i, step) in enumerate(scenario.steps)
        r = execute_step!(brain, read_tools, bt, step, i)
        push!(results, r)
    end

    passed = true
    evaluate_results = filter(r -> r.step_type == "evaluate", results)

    for expected in scenario.expected_outcomes
        if expected.turn !== nothing
            idx = findfirst(r -> r.step_number == expected.turn, results)
            if idx === nothing
                println("  FAIL: Expected outcome at turn $(expected.turn) but no result found")
                passed = false
                continue
            end
            actual = results[idx]
            if actual.decision != expected.expected_decision
                println("  FAIL: Turn $(expected.turn) expected decision=$(expected.expected_decision), got=$(actual.decision)")
                passed = false
            end
        else
            found = any(r -> r.decision == expected.expected_decision && r.detector == expected.detector, evaluate_results)
            if !found
                println("  FAIL: Expected decision=$(expected.expected_decision) with detector=$(expected.detector), not found in any evaluate step")
                passed = false
            end
        end
    end

    (passed, results)
end

# ── Main ──

function main()
    bt = load_budget_table(CONFIG_PATH)
    read_tools = load_read_tools(READ_TOOLS_PATH)

    scenario_files = if length(ARGS) > 0
        [ARGS[1]]
    else
        sort([joinpath(SCENARIOS_DIR, f) for f in readdir(SCENARIOS_DIR) if endswith(f, ".json")])
    end

    println("=" ^ 70)
    println("MOVE 3 — TARGETED DEMONSTRATION EVALUATION")
    println("=" ^ 70)
    println("Scenarios: $(length(scenario_files))")
    println()

    all_passed = true
    scenario_summaries = Dict{String, Tuple{Bool, Vector{StepResult}}}()

    for path in scenario_files
        scenario = load_scenario(path)
        println("─" ^ 70)
        println("Scenario: $(scenario.name)")
        println("Source:   $(scenario.source)")
        println("─" ^ 70)

        (passed, results) = run_scenario(scenario, bt, read_tools)

        for r in results
            println(format_result(r))
        end
        println()

        if passed
            println("  VERDICT: PASS")
        else
            println("  VERDICT: FAIL")
            all_passed = false
        end
        println()

        scenario_summaries[scenario.name] = (passed, results)
    end

    println("=" ^ 70)
    println("COVERAGE MATRIX")
    println("=" ^ 70)
    println()
    println("| Scenario | #34574 | #1084 | #65550 | Halt | Escalate | Verdict |")
    println("|---|---|---|---|---|---|---|")

    for path in scenario_files
        scenario = load_scenario(path)
        (passed, results) = scenario_summaries[scenario.name]
        eval_results = filter(r -> r.step_type == "evaluate", results)
        has_34574 = any(r -> r.detector == "#34574", eval_results) ? "✓" : ""
        has_1084 = any(r -> r.detector == "#1084", eval_results) ? "✓" : ""
        has_65550 = any(r -> r.detector == "#65550", eval_results) ? "✓" : ""
        has_halt = any(r -> r.decision == "halt", eval_results) ? "✓" : ""
        has_escalate = any(r -> r.decision == "escalate", eval_results) ? "✓" : ""
        verdict = passed ? "PASS" : "FAIL"
        println("| $(scenario.name) | $has_34574 | $has_1084 | $has_65550 | $has_halt | $has_escalate | $verdict |")
    end
    println()

    println("=" ^ 70)
    if all_passed
        println("ALL $(length(scenario_files)) SCENARIOS PASSED")
    else
        println("SOME SCENARIOS FAILED")
    end
    println("=" ^ 70)

    exit(all_passed ? 0 : 1)
end

main()

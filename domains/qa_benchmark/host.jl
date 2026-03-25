"""
    host.jl — QA benchmark host driver.

Runs the Bayesian agent (credence_agent.bdsl) and baselines against 50
simulated multiple-choice questions with 4 tools of varying reliability.

Usage:
    julia domains/qa_benchmark/host.jl                          # 20 seeds, fast agents only
    julia domains/qa_benchmark/host.jl --seeds 3                # quick test
    julia domains/qa_benchmark/host.jl --seeds 20 --include-llm # full run with 8 LLM variants
"""

# --- Setup ---
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Random
using Statistics: mean as smean

include(joinpath(@__DIR__, "tools.jl"))
include(joinpath(@__DIR__, "questions.jl"))
include(joinpath(@__DIR__, "metrics.jl"))

const REWARD_CORRECT = 10.0
const PENALTY_WRONG = -5.0
const REWARD_ABSTAIN = 0.0

# --- Load DSL ---
const BDSL_PATH = joinpath(@__DIR__, "..", "..", "examples", "credence_agent.bdsl")
const DSL_ENV = load_dsl(read(BDSL_PATH, String))
const AGENT_STEP = DSL_ENV[Symbol("agent-step")]
const UPDATE_ON_RESPONSE = DSL_ENV[Symbol("update-on-response")]
const ANSWER_KERNEL = DSL_ENV[Symbol("answer-kernel")]

# --- Bayesian agent ---

function run_bayesian_seed(tools::Vector{SimulatedTool}, seed::Int)
    rng = MersenneTwister(seed)
    questions = get_questions(; seed)
    n_tools = length(tools)
    n_cats = length(CATEGORIES)

    # Per-tool state (reset each seed)
    rel_states = [initial_rel_state(n_cats) for _ in 1:n_tools]
    cov_states = [initial_cov_state(n_cats,
                    Float64[get(t.coverage_by_category, c, 0.0) for c in CATEGORIES])
                  for t in tools]
    cat_belief = CategoricalMeasure(Finite(Float64.(0:n_cats-1)))

    records = QuestionResult[]
    total_reward = 0.0
    total_tool_cost = 0.0
    t_start = time()

    for q in questions
        answer_measure = CategoricalMeasure(Finite(Float64[0, 1, 2, 3]))
        available = collect(1:n_tools)
        tools_queried = Int[]
        q_cost = 0.0

        while true
            cat_w = weights(cat_belief)
            rel_measures = [marginalize_betas(rel_states[t], cat_w) for t in available]
            costs_jl = Float64[tools[t].cost for t in available]

            result = AGENT_STEP(answer_measure, rel_measures, costs_jl,
                                REWARD_CORRECT, REWARD_ABSTAIN, PENALTY_WRONG)
            action_type = Int(result[1])
            action_arg = Int(result[2])

            if action_type == 2  # query
                tool_local_idx = action_arg + 1  # DSL is 0-indexed
                tool_idx = available[tool_local_idx]
                push!(tools_queried, tool_idx)
                q_cost += tools[tool_idx].cost

                response = query_tool(tools[tool_idx], q, rng)

                if response !== nothing
                    # Update answer belief
                    k = ANSWER_KERNEL(marginalize_betas(rel_states[tool_idx], cat_w), 4.0)
                    answer_measure = UPDATE_ON_RESPONSE(answer_measure, k, Float64(response))
                    # Update coverage (responded)
                    cov_states[tool_idx], cat_belief = update_beta_state(
                        cov_states[tool_idx], cat_belief, 1.0)
                else
                    # Update coverage (not responded)
                    cov_states[tool_idx], cat_belief = update_beta_state(
                        cov_states[tool_idx], cat_belief, 0.0)
                end

                filter!(!=(tool_idx), available)

            elseif action_type == 0  # submit
                submitted = action_arg
                was_correct = submitted == q.correct_index
                reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
                total_reward += reward
                total_tool_cost += q_cost

                # Update reliability for each queried tool
                for t in tools_queried
                    tool_response = nothing  # find what this tool said
                    # We need to track responses... let me fix this
                end

                push!(records, QuestionResult(q.id, q.category, tools_queried,
                    submitted, was_correct, reward, q_cost))
                break

            else  # abstain
                total_reward += REWARD_ABSTAIN
                total_tool_cost += q_cost
                push!(records, QuestionResult(q.id, q.category, tools_queried,
                    nothing, nothing, REWARD_ABSTAIN, q_cost))
                break
            end
        end
    end

    wall_time = time() - t_start
    SeedResult(seed, records, total_reward - total_tool_cost,
               total_reward, total_tool_cost, wall_time)
end

# --- Baseline agents ---

function run_single_best_seed(tools::Vector{SimulatedTool}, seed::Int; tool_idx::Int=1)
    rng = MersenneTwister(seed)
    questions = get_questions(; seed)
    records = QuestionResult[]
    total_reward = 0.0
    total_tool_cost = 0.0
    t_start = time()

    for q in questions
        response = query_tool(tools[tool_idx], q, rng)
        cost = tools[tool_idx].cost
        total_tool_cost += cost

        if response !== nothing
            was_correct = response == q.correct_index
            reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
        else
            # Tool didn't respond — submit candidate 0 as fallback
            was_correct = 0 == q.correct_index
            reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
        end
        total_reward += reward
        push!(records, QuestionResult(q.id, q.category, [tool_idx],
            response !== nothing ? response : 0, was_correct, reward, cost))
    end

    wall_time = time() - t_start
    SeedResult(seed, records, total_reward - total_tool_cost,
               total_reward, total_tool_cost, wall_time)
end

function run_random_seed(tools::Vector{SimulatedTool}, seed::Int)
    rng = MersenneTwister(seed)
    questions = get_questions(; seed)
    records = QuestionResult[]
    total_reward = 0.0
    total_tool_cost = 0.0
    t_start = time()

    for q in questions
        t = rand(rng, 1:length(tools))
        response = query_tool(tools[t], q, rng)
        cost = tools[t].cost
        total_tool_cost += cost

        submitted = response !== nothing ? response : rand(rng, 0:3)
        was_correct = submitted == q.correct_index
        reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
        total_reward += reward
        push!(records, QuestionResult(q.id, q.category, [t],
            submitted, was_correct, reward, cost))
    end

    wall_time = time() - t_start
    SeedResult(seed, records, total_reward - total_tool_cost,
               total_reward, total_tool_cost, wall_time)
end

function run_all_tools_seed(tools::Vector{SimulatedTool}, seed::Int)
    rng = MersenneTwister(seed)
    questions = get_questions(; seed)
    records = QuestionResult[]
    total_reward = 0.0
    total_tool_cost = 0.0
    t_start = time()

    for q in questions
        responses = Dict{Int,Union{Int,Nothing}}()
        cost = 0.0
        for t in 1:length(tools)
            responses[t] = query_tool(tools[t], q, rng)
            cost += tools[t].cost
        end
        total_tool_cost += cost

        # Majority vote
        votes = [r for r in values(responses) if r !== nothing]
        submitted = if isempty(votes)
            0
        else
            # Most common
            counts = Dict{Int,Int}()
            for v in votes; counts[v] = get(counts, v, 0) + 1; end
            first(sort(collect(counts); by=last, rev=true))[1]
        end
        was_correct = submitted == q.correct_index
        reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
        total_reward += reward
        push!(records, QuestionResult(q.id, q.category, collect(1:length(tools)),
            submitted, was_correct, reward, cost))
    end

    wall_time = time() - t_start
    SeedResult(seed, records, total_reward - total_tool_cost,
               total_reward, total_tool_cost, wall_time)
end

# --- LLM agents (optional) ---
include("llm_agents.jl")

# 3 variants: bare, reasoning only, maximum prompting
const LLM_VARIANTS = [
    (react=false, strategy=false, history=false),  # bare
    (react=true,  strategy=false, history=false),  # R (reasoning traces)
    (react=true,  strategy=true,  history=true),   # RSH (max prompting)
]

# --- CLI parsing ---

function parse_args()
    n_seeds = 20
    include_llm = false
    models = String[]
    for (i, arg) in enumerate(ARGS)
        if arg == "--include-llm"
            include_llm = true
        elseif arg == "--seeds" && i < length(ARGS)
            n_seeds = parse(Int, ARGS[i + 1])
        elseif startswith(arg, "--seeds=")
            n_seeds = parse(Int, split(arg, "=")[2])
        elseif arg == "--model" && i < length(ARGS)
            push!(models, ARGS[i + 1])
        elseif startswith(arg, "--model=")
            push!(models, split(arg, "=")[2])
        end
    end
    if isempty(models); models = ["llama3.1"]; end
    (; n_seeds, include_llm, models)
end

# --- Main ---

function main()
    args = parse_args()
    tools = make_spec_tools()

    mode = args.include_llm ? "fast + 8 LLM variants" : "fast agents only"
    println(stderr, "QA Benchmark: $(args.n_seeds) seeds, $(length(QUESTION_BANK)) questions, $mode")
    flush(stderr)

    all_results = Dict{String,Vector{SeedResult}}()

    # Fast agents (seconds)
    println(stderr, "\n--- Fast agents ---")
    for seed in 0:args.n_seeds-1
        for (name, runner) in [
            ("bayesian",    s -> run_bayesian_seed(tools, s)),
            ("single_best", s -> run_single_best_seed(tools, s)),
            ("random",      s -> run_random_seed(tools, s)),
            ("all_tools",   s -> run_all_tools_seed(tools, s)),
        ]
            push!(get!(all_results, name, SeedResult[]), runner(seed))
        end
        println(stderr, "  Seed $seed done"); flush(stderr)
    end

    # Intermediate results
    println(stderr, "\n--- Fast agent results ---")
    println(stderr, summary_table(all_results)); flush(stderr)

    # LLM agents — sequential per model × variant × seed
    if args.include_llm
        n_variants = length(LLM_VARIANTS)
        logpath = joinpath(@__DIR__, "llm_debug.log")
        logfile = open(logpath, "w")

        for model in args.models
            println(stderr, "\n--- LLM agents: $model ($n_variants variants × $(args.n_seeds) seeds) ---")
            println(stderr, "  Debug log: $logpath"); flush(stderr)

            for variant in LLM_VARIANTS
                vname = llm_variant_name(; variant...)
                name = length(args.models) > 1 ? "$(model)_$vname" : vname
                println(stderr, "  Starting $name..."); flush(stderr)
                for seed in 0:args.n_seeds-1
                    result = run_llm_seed(tools, seed; variant..., model, logfile)
                    push!(get!(all_results, name, SeedResult[]), result)
                    score = @sprintf("%+.1f", result.total_score)
                    println(stderr, "    $name seed $seed: score=$score"); flush(stderr)
                end
                # Running table after each variant
                println(stderr, "\n--- Results after $name ---")
                println(stderr, summary_table(all_results)); flush(stderr)
            end
        end

        close(logfile)
    end

    # Final table to stdout
    println(summary_table(all_results)); flush(stdout)
end

main()

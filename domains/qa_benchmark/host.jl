"""
    host.jl — QA benchmark host driver.

Runs the Bayesian agent (agent.bdsl) and baselines against 50
simulated multiple-choice questions with 4 tools of varying reliability.

Usage:
    julia domains/qa_benchmark/host.jl                          # 20 seeds, fast agents only
    julia domains/qa_benchmark/host.jl --seeds 3                # quick test
    julia domains/qa_benchmark/host.jl --seeds 20 --include-llm # with LLM agents
"""

# --- Setup ---
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Random

include(joinpath(@__DIR__, "environment.jl"))
include(joinpath(@__DIR__, "metrics.jl"))

# --- Load DSL ---
const BDSL_PATH = joinpath(@__DIR__, "agent.bdsl")
const DSL_ENV = load_dsl(read(BDSL_PATH, String))
const AGENT_STEP = DSL_ENV[Symbol("agent-step")]
const UPDATE_ON_RESPONSE = DSL_ENV[Symbol("update-on-response")]
const ANSWER_KERNEL = DSL_ENV[Symbol("answer-kernel")]
const RELIABILITY_KERNEL = DSL_ENV[Symbol("reliability-kernel")]

# --- Bayesian agent ---

function run_bayesian_seed(tools::Vector{SimulatedTool}, seed::Int)
    rng = MersenneTwister(seed)
    questions = get_questions(; seed)
    n_tools = length(tools)
    n_cats = length(CATEGORIES)

    # Per-tool per-category reliability: flat matrix of Betas, all Beta(1,1)
    rel_betas = [BetaMeasure() for _ in 1:n_tools, _ in 1:n_cats]

    records = QuestionResult[]
    total_reward = 0.0
    total_tool_cost = 0.0
    t_start = time()

    for q in questions
        cat_idx = findfirst(==(q.category), CATEGORIES)
        answer_measure = CategoricalMeasure(Finite(Float64[0, 1, 2, 3]))
        available = collect(1:n_tools)
        tool_responses = Dict{Int,Int}()
        q_cost = 0.0

        while true
            # Extract reliability measures for available tools in this category
            rel_measures = [rel_betas[t, cat_idx] for t in available]
            costs_jl = Float64[tools[t].cost for t in available]

            result = AGENT_STEP(answer_measure, rel_measures, costs_jl,
                                REWARD_CORRECT, REWARD_ABSTAIN, PENALTY_WRONG)
            action_type = Int(result[1])
            action_arg = Int(result[2])

            if action_type == 2  # query tool
                tool_local_idx = action_arg + 1  # DSL is 0-indexed
                tool_idx = available[tool_local_idx]
                q_cost += tools[tool_idx].cost

                response = query_tool(tools[tool_idx], q, rng)
                tool_responses[tool_idx] = response

                # Update answer belief via DSL
                k = ANSWER_KERNEL(rel_betas[tool_idx, cat_idx], 4.0)
                answer_measure = UPDATE_ON_RESPONSE(answer_measure, k, Float64(response))

                filter!(!=(tool_idx), available)

            elseif action_type == 0  # submit
                submitted = action_arg
                was_correct = submitted == q.correct_index
                reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
                total_reward += reward
                total_tool_cost += q_cost

                # Update reliability beliefs with ground truth
                for (t, resp) in tool_responses
                    rel_betas[t, cat_idx] = condition(
                        rel_betas[t, cat_idx], RELIABILITY_KERNEL,
                        resp == q.correct_index ? 1.0 : 0.0)
                end

                push!(records, QuestionResult(q.id, q.category,
                    collect(keys(tool_responses)), tool_responses,
                    submitted, was_correct, reward, q_cost))
                break

            else  # abstain
                total_reward += REWARD_ABSTAIN
                total_tool_cost += q_cost

                # Update reliability beliefs with ground truth
                for (t, resp) in tool_responses
                    rel_betas[t, cat_idx] = condition(
                        rel_betas[t, cat_idx], RELIABILITY_KERNEL,
                        resp == q.correct_index ? 1.0 : 0.0)
                end

                push!(records, QuestionResult(q.id, q.category,
                    collect(keys(tool_responses)), tool_responses,
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

        was_correct = response == q.correct_index
        reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
        total_reward += reward
        push!(records, QuestionResult(q.id, q.category, [tool_idx],
            Dict(tool_idx => response),
            response, was_correct, reward, cost))
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

        was_correct = response == q.correct_index
        reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
        total_reward += reward
        push!(records, QuestionResult(q.id, q.category, [t],
            Dict(t => response),
            response, was_correct, reward, cost))
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
        responses = Dict{Int,Int}()
        cost = 0.0
        for t in 1:length(tools)
            responses[t] = query_tool(tools[t], q, rng)
            cost += tools[t].cost
        end
        total_tool_cost += cost

        # Majority vote
        counts = Dict{Int,Int}()
        for v in values(responses); counts[v] = get(counts, v, 0) + 1; end
        submitted = first(sort(collect(counts); by=last, rev=true))[1]

        was_correct = submitted == q.correct_index
        reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
        total_reward += reward
        push!(records, QuestionResult(q.id, q.category, collect(1:length(tools)),
            responses, submitted, was_correct, reward, cost))
    end

    wall_time = time() - t_start
    SeedResult(seed, records, total_reward - total_tool_cost,
               total_reward, total_tool_cost, wall_time)
end

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
        elseif arg == "--ablation"
            println(stderr, "Ablation mode not yet implemented")
        end
    end
    if isempty(models); models = ["llama3.1"]; end
    (; n_seeds, include_llm, models)
end

# --- Main ---

function main()
    args = parse_args()
    tools = make_spec_tools()

    mode = args.include_llm ? "fast + LLM agents" : "fast agents only"
    println(stderr, "QA Benchmark: $(args.n_seeds) seeds, $(length(QUESTION_BANK)) questions, $mode")
    flush(stderr)

    all_results = Dict{String,Vector{SeedResult}}()

    # Fast agents
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

    # LLM agents (optional)
    if args.include_llm
        include(joinpath(@__DIR__, "llm_agent.jl"))
        for model in args.models
            println(stderr, "\n--- LLM agent: $model ---"); flush(stderr)
            for seed in 0:args.n_seeds-1
                backend = if startswith(model, "claude")
                    AnthropicBackend(model, get(ENV, "ANTHROPIC_API_KEY", ""))
                else
                    OllamaBackend(model, "http://localhost:11434")
                end
                result = run_llm_seed(backend, tools, seed)
                push!(get!(all_results, model, SeedResult[]), result)
                score = @sprintf("%+.1f", result.total_score)
                println(stderr, "  $model seed $seed: score=$score"); flush(stderr)
            end
            println(stderr, "\n--- Results after $model ---")
            println(stderr, summary_table(all_results)); flush(stderr)
        end
    end

    # Final table to stdout
    println(summary_table(all_results)); flush(stdout)

    # Save raw results
    results_path = joinpath(@__DIR__, "results", "results.json")
    save_results(results_path, all_results)
end

main()

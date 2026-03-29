"""
    host.jl — QA benchmark host driver.

Runs the Bayesian agent (agent.bdsl) and baselines against 50
simulated multiple-choice questions with 4 tools of varying reliability.

All agents see identical tool responses for a given seed: responses are
pre-generated in a response table before any agent runs.

Results stored in SQLite at results/benchmark.db.

Usage:
    julia domains/qa_benchmark/host.jl                          # 20 seeds, fast agents only
    julia domains/qa_benchmark/host.jl --seeds 3                # quick test
    julia domains/qa_benchmark/host.jl --seeds 20 --include-llm # with LLM agents
    julia domains/qa_benchmark/host.jl --seeds 20 --llm-only --model claude-haiku-4-5-20251001
"""

# --- Setup ---
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Random

include(joinpath(@__DIR__, "environment.jl"))
include(joinpath(@__DIR__, "metrics.jl"))
include(joinpath(@__DIR__, "llm_agent.jl"))

const DB_PATH = joinpath(@__DIR__, "results", "benchmark.db")

# --- Load DSL ---
const BDSL_PATH = joinpath(@__DIR__, "agent.bdsl")
const DSL_ENV = load_dsl(read(BDSL_PATH, String))
const AGENT_STEP = DSL_ENV[Symbol("agent-step")]
const UPDATE_ON_RESPONSE = DSL_ENV[Symbol("update-on-response")]
const ANSWER_KERNEL = DSL_ENV[Symbol("answer-kernel")]
const RELIABILITY_KERNEL = DSL_ENV[Symbol("reliability-kernel")]

# --- Bayesian agent ---

function run_bayesian_seed(tools::Vector{SimulatedTool},
                           questions::Vector{Question},
                           response_table::Matrix{Int};
                           use_voi::Bool=true, learn::Bool=true,
                           allow_abstain::Bool=true, greedy::Bool=false)
    n_tools = length(tools)
    n_cats = length(CATEGORIES)

    rel_betas = [BetaMeasure() for _ in 1:n_tools, _ in 1:n_cats]

    records = QuestionResult[]
    total_reward = 0.0
    total_tool_cost = 0.0
    t_start = time()

    for (qi, q) in enumerate(questions)
        q_t0 = time()
        cat_idx = findfirst(==(q.category), CATEGORIES)
        answer_measure = CategoricalMeasure(Finite(Float64[0, 1, 2, 3]))
        available = collect(1:n_tools)
        tool_responses = Dict{Int,Int}()
        q_cost = 0.0

        if greedy
            # Query tool with highest E[reliability], submit its answer directly
            best_t = argmax(t -> mean(rel_betas[t, cat_idx]), 1:n_tools)
            q_cost += tools[best_t].cost
            response = response_table[qi, best_t]
            tool_responses[best_t] = response
            submitted = response
            was_correct = submitted == q.correct_index
            reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
            total_reward += reward
            total_tool_cost += q_cost
            push!(records, QuestionResult(q.id, q.category, [best_t], tool_responses,
                submitted, was_correct, reward, q_cost, time() - q_t0, 0, 0))

        elseif !use_voi
            # Query all tools cheapest-first, update answer belief, then submit/abstain
            for t in sortperm([tools[t].cost for t in 1:n_tools])
                q_cost += tools[t].cost
                response = response_table[qi, t]
                tool_responses[t] = response
                k = ANSWER_KERNEL(rel_betas[t, cat_idx], 4.0)
                answer_measure = UPDATE_ON_RESPONSE(answer_measure, k, Float64(response))
            end
            # Submit argmax or abstain based on EU
            w = weights(answer_measure)
            best_idx = argmax(w)
            submitted = Int(answer_measure.space.values[best_idx])
            p_correct = w[best_idx]
            eu_submit = p_correct * REWARD_CORRECT + (1 - p_correct) * PENALTY_WRONG
            if eu_submit > 0 || !allow_abstain
                was_correct = submitted == q.correct_index
                reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
                total_reward += reward
                total_tool_cost += q_cost
                push!(records, QuestionResult(q.id, q.category,
                    collect(keys(tool_responses)), tool_responses,
                    submitted, was_correct, reward, q_cost, time() - q_t0, 0, 0))
            else
                total_reward += REWARD_ABSTAIN
                total_tool_cost += q_cost
                push!(records, QuestionResult(q.id, q.category,
                    collect(keys(tool_responses)), tool_responses,
                    nothing, nothing, REWARD_ABSTAIN, q_cost, time() - q_t0, 0, 0))
            end

        else
            # Full VOI loop
            while true
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

                    response = response_table[qi, tool_idx]
                    tool_responses[tool_idx] = response

                    k = ANSWER_KERNEL(rel_betas[tool_idx, cat_idx], 4.0)
                    answer_measure = UPDATE_ON_RESPONSE(answer_measure, k, Float64(response))

                    filter!(!=(tool_idx), available)

                elseif action_type == 0  # submit
                    submitted = action_arg
                    was_correct = submitted == q.correct_index
                    reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
                    total_reward += reward
                    total_tool_cost += q_cost
                    push!(records, QuestionResult(q.id, q.category,
                        collect(keys(tool_responses)), tool_responses,
                        submitted, was_correct, reward, q_cost, time() - q_t0, 0, 0))
                    break

                else  # abstain
                    if !allow_abstain
                        w = weights(answer_measure)
                        submitted = Int(answer_measure.space.values[argmax(w)])
                        was_correct = submitted == q.correct_index
                        reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
                        total_reward += reward
                        total_tool_cost += q_cost
                        push!(records, QuestionResult(q.id, q.category,
                            collect(keys(tool_responses)), tool_responses,
                            submitted, was_correct, reward, q_cost, time() - q_t0, 0, 0))
                        break
                    end

                    total_reward += REWARD_ABSTAIN
                    total_tool_cost += q_cost
                    push!(records, QuestionResult(q.id, q.category,
                        collect(keys(tool_responses)), tool_responses,
                        nothing, nothing, REWARD_ABSTAIN, q_cost, time() - q_t0, 0, 0))
                    break
                end
            end
        end

        # Reliability learning (skip if learn=false)
        if learn
            for (t, resp) in tool_responses
                rel_betas[t, cat_idx] = condition(
                    rel_betas[t, cat_idx], RELIABILITY_KERNEL,
                    resp == q.correct_index ? 1.0 : 0.0)
            end
        end
    end

    wall_time = time() - t_start
    total_score = total_reward - total_tool_cost
    SeedResult(0, records, total_score, total_reward, total_tool_cost, wall_time)
end

# --- Baseline agents ---

function run_single_best_seed(tools::Vector{SimulatedTool},
                              questions::Vector{Question},
                              response_table::Matrix{Int};
                              tool_idx::Int=1)
    records = QuestionResult[]
    total_reward = 0.0
    total_tool_cost = 0.0
    t_start = time()

    for (qi, q) in enumerate(questions)
        q_t0 = time()
        response = response_table[qi, tool_idx]
        cost = tools[tool_idx].cost
        total_tool_cost += cost

        was_correct = response == q.correct_index
        reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
        total_reward += reward
        push!(records, QuestionResult(q.id, q.category, [tool_idx],
            Dict(tool_idx => response),
            response, was_correct, reward, cost, time() - q_t0, 0, 0))
    end

    wall_time = time() - t_start
    SeedResult(0, records, total_reward - total_tool_cost,
               total_reward, total_tool_cost, wall_time)
end

function run_random_seed(tools::Vector{SimulatedTool},
                         questions::Vector{Question},
                         response_table::Matrix{Int},
                         rng)
    records = QuestionResult[]
    total_reward = 0.0
    total_tool_cost = 0.0
    t_start = time()

    for (qi, q) in enumerate(questions)
        q_t0 = time()
        t = rand(rng, 1:length(tools))
        response = response_table[qi, t]
        cost = tools[t].cost
        total_tool_cost += cost

        was_correct = response == q.correct_index
        reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
        total_reward += reward
        push!(records, QuestionResult(q.id, q.category, [t],
            Dict(t => response),
            response, was_correct, reward, cost, time() - q_t0, 0, 0))
    end

    wall_time = time() - t_start
    SeedResult(0, records, total_reward - total_tool_cost,
               total_reward, total_tool_cost, wall_time)
end

function run_all_tools_seed(tools::Vector{SimulatedTool},
                            questions::Vector{Question},
                            response_table::Matrix{Int})
    records = QuestionResult[]
    total_reward = 0.0
    total_tool_cost = 0.0
    t_start = time()

    for (qi, q) in enumerate(questions)
        q_t0 = time()
        responses = Dict{Int,Int}()
        cost = 0.0
        for t in 1:length(tools)
            responses[t] = response_table[qi, t]
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
            responses, submitted, was_correct, reward, cost, time() - q_t0, 0, 0))
    end

    wall_time = time() - t_start
    SeedResult(0, records, total_reward - total_tool_cost,
               total_reward, total_tool_cost, wall_time)
end

# --- CLI parsing ---

function parse_args()
    n_seeds = 20
    include_llm = false
    llm_only = false
    ablation = false
    models = String[]
    for (i, arg) in enumerate(ARGS)
        if arg == "--include-llm"
            include_llm = true
        elseif arg == "--llm-only"
            include_llm = true
            llm_only = true
        elseif arg == "--seeds" && i < length(ARGS)
            n_seeds = parse(Int, ARGS[i + 1])
        elseif startswith(arg, "--seeds=")
            n_seeds = parse(Int, split(arg, "=")[2])
        elseif arg == "--model" && i < length(ARGS)
            push!(models, ARGS[i + 1])
        elseif startswith(arg, "--model=")
            push!(models, split(arg, "=")[2])
        elseif arg == "--ablation"
            ablation = true
        end
    end
    if isempty(models); models = ["llama3.1"]; end
    (; n_seeds, include_llm, llm_only, ablation, models)
end

# --- Main ---

function main()
    args = parse_args()
    tools = make_spec_tools()

    mode = args.llm_only ? "LLM only" : args.include_llm ? "fast + LLM agents" : "fast agents only"
    println(stderr, "QA Benchmark: $(args.n_seeds) seeds, $(length(QUESTION_BANK)) questions, $mode")
    flush(stderr)

    all_results = Dict{String,Vector{SeedResult}}()

    # Fast agents
    if !args.llm_only
    println(stderr, "\n--- Fast agents ---")
    for seed in 0:args.n_seeds-1
        rng = MersenneTwister(seed)
        questions = get_questions(; seed)
        response_table = generate_response_table(tools, questions, rng)

        random_rng = MersenneTwister(seed + 1_000_000)

        for (name, runner) in [
            ("bayesian",    () -> run_bayesian_seed(tools, questions, response_table)),
            ("single_best", () -> run_single_best_seed(tools, questions, response_table)),
            ("random",      () -> run_random_seed(tools, questions, response_table, random_rng)),
            ("all_tools",   () -> run_all_tools_seed(tools, questions, response_table)),
        ]
            result = runner()
            result = SeedResult(seed, result.records, result.total_score,
                               result.total_reward, result.total_tool_cost, result.wall_time_s)
            push!(get!(all_results, name, SeedResult[]), result)
        end
        println(stderr, "  Seed $seed done"); flush(stderr)
    end

    println(stderr, "\n--- Fast agent results ---")
    println(stderr, summary_table(all_results)); flush(stderr)

    # Save fast agents to SQLite
    for (name, runs) in all_results
        save_results(DB_PATH, name, runs)
    end
    end # !llm_only

    # Ablation variants (Bayesian only)
    if args.ablation
        ablation_variants = [
            ("ablation_no_voi",       (use_voi=false,  learn=true,  allow_abstain=true,  greedy=false)),
            ("ablation_no_learning",  (use_voi=true,   learn=false, allow_abstain=true,  greedy=false)),
            ("ablation_no_abstain",   (use_voi=true,   learn=true,  allow_abstain=false, greedy=false)),
            ("ablation_greedy",       (use_voi=true,   learn=true,  allow_abstain=true,  greedy=true)),
        ]
        println(stderr, "\n--- Ablation variants ---"); flush(stderr)
        for (name, kwargs) in ablation_variants
            println(stderr, "  Running $name..."); flush(stderr)
            for seed in 0:args.n_seeds-1
                rng = MersenneTwister(seed)
                questions = get_questions(; seed)
                response_table = generate_response_table(tools, questions, rng)
                result = run_bayesian_seed(tools, questions, response_table; kwargs...)
                result = SeedResult(seed, result.records, result.total_score,
                                   result.total_reward, result.total_tool_cost, result.wall_time_s)
                push!(get!(all_results, name, SeedResult[]), result)
            end
        end
        println(stderr, "\n--- Ablation results ---")
        println(stderr, summary_table(all_results)); flush(stderr)

        # Save ablation results to SQLite
        for (name, _) in ablation_variants
            save_results(DB_PATH, name, all_results[name])
        end
    end

    # LLM agents (optional)
    if args.include_llm
        for model in args.models
            println(stderr, "\n--- LLM agent: $model ---"); flush(stderr)
            llm_runs = SeedResult[]
            for seed in 0:args.n_seeds-1
                questions = get_questions(; seed)
                rng = MersenneTwister(seed)
                response_table = generate_response_table(tools, questions, rng)

                backend = if startswith(model, "claude")
                    AnthropicBackend(model, get(ENV, "ANTHROPIC_API_KEY", ""))
                else
                    OllamaBackend(model, "http://localhost:11434")
                end
                result = run_llm_seed(backend, tools, questions, response_table, seed)
                push!(llm_runs, result)
                push!(get!(all_results, model, SeedResult[]), result)
                score = @sprintf("%+.1f", result.total_score)
                println(stderr, "  $model seed $seed: score=$score"); flush(stderr)
            end
            # Save this LLM agent to SQLite
            save_results(DB_PATH, model, llm_runs; model)

            println(stderr, "\n--- Results after $model ---")
            println(stderr, summary_table(all_results)); flush(stderr)
        end
    end

    # Final table to stdout
    println(summary_table(all_results)); flush(stdout)
end

main()

# Role: brain-side application
"""
    host.jl — QA benchmark host driver.

Runs the Bayesian agent (agent.bdsl) and baselines against 50
simulated multiple-choice questions with 4 tools of varying reliability.

All agents see identical tool responses for a given seed: responses are
pre-generated in a response table before any agent runs.

Results stored in SQLite at results/benchmark.db.

Usage:
    julia apps/julia/qa_benchmark/host.jl                          # 20 seeds, fast agents only
    julia apps/julia/qa_benchmark/host.jl --seeds 3                # quick test
    julia apps/julia/qa_benchmark/host.jl --seeds 20 --include-llm # with LLM agents
    julia apps/julia/qa_benchmark/host.jl --seeds 20 --llm-only --model claude-haiku-4-5-20251001
"""

# --- Setup ---
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
using Credence
using Random

include(joinpath(@__DIR__, "environment.jl"))
include(joinpath(@__DIR__, "metrics.jl"))
include(joinpath(@__DIR__, "llm_agent.jl"))
include(joinpath(@__DIR__, "category_inference.jl"))
include(joinpath(@__DIR__, "category_update.jl"))

const DB_PATH = joinpath(@__DIR__, "results", "benchmark.db")

# --- Load DSL ---
const BDSL_PATH = joinpath(@__DIR__, "agent.bdsl")
const DSL_ENV = load_dsl(read(BDSL_PATH, String))
const AGENT_STEP = DSL_ENV[Symbol("agent-step")]
const UPDATE_ON_RESPONSE = DSL_ENV[Symbol("update-on-response")]
const ANSWER_KERNEL = DSL_ENV[Symbol("answer-kernel")]
const RELIABILITY_KERNEL = DSL_ENV[Symbol("reliability-kernel")]
const HORIZON_STEP = DSL_ENV[Symbol("horizon-step")]

# --- Inferred-category posteriors (offline embedding fixture) ---

const FIXTURE_DIR = joinpath(@__DIR__, "fixtures")

"""
    category_posteriors_from_fixture() -> Dict{String, Vector{Float64}}

Load the committed sentence-embedding fixture and compute the
leave-one-out category posterior for every question (id → soft posterior
aligned to CATEGORIES). Used to run the Bayesian agent under *inferred*
("fair") categories. Fully offline — sentence-transformers is not needed.
"""
function category_posteriors_from_fixture()
    bank = JSON3.read(read(joinpath(FIXTURE_DIR, "question_bank.json"), String))
    emb = JSON3.read(read(joinpath(FIXTURE_DIR, "question_embeddings.json"), String))
    byid = emb.embeddings
    N = length(bank)
    D = emb.dim
    X = Matrix{Float64}(undef, N, D)
    cats = Vector{Symbol}(undef, N)
    ids = Vector{String}(undef, N)
    for (i, q) in enumerate(bank)
        X[i, :] = Float64.(byid[Symbol(q.id)])
        cats[i] = Symbol(q.category)
        ids[i] = String(q.id)
    end
    posts = loo_category_inference(X, cats)
    out = Dict{String,Vector{Float64}}()
    for i in 1:N
        out[ids[i]] = [get(posts[i], Symbol(c), 0.0) for c in CATEGORIES]
    end
    out
end

# --- Bayesian agent ---

function run_bayesian_seed(tools::Vector{SimulatedTool},
                           questions::Vector{Question},
                           response_table::Matrix{Int};
                           use_voi::Bool=true, learn::Bool=true,
                           allow_abstain::Bool=true, greedy::Bool=false,
                           category_posteriors::Union{Nothing,Dict{String,Vector{Float64}}}=nothing,
                           credit::Symbol=:post,
                           init_reliability::Union{Nothing,AbstractMatrix}=nothing)
    n_tools = length(tools)
    n_cats = length(CATEGORIES)
    credit in (:post, :soft) ||
        error("run_bayesian_seed: credit must be :post or :soft, got :$credit")

    # When `category_posteriors` is supplied (id → soft posterior over
    # CATEGORIES), the agent runs under *inferred* categories (Paper 1 fair
    # conditions): decisions marginalise reliability over the posterior and
    # learning is the posterior-weighted update (`credit=:post`, the deployed
    # default; issue #111) or the soft-credit B2c update (`credit=:soft`, the
    # paper's committed numbers). Otherwise it uses the given category
    # (one-hot) — the v1 path, bit-for-bit, unaffected by `credit`.
    inferred = category_posteriors !== nothing

    # init_reliability seeds the per-(tool,category) reliability beliefs. With
    # learn=false this gives the *known-reliability ceiling* — myopic-EU acting
    # on the true θ table (the asymptote of perfect exploration). Default prior
    # Beta(1,1) is the normal learning start.
    rel_betas = init_reliability === nothing ?
        [BetaPrevision(1.0, 1.0) for _ in 1:n_tools, _ in 1:n_cats] :
        copy(init_reliability)

    records = QuestionResult[]
    total_reward = 0.0
    total_tool_cost = 0.0
    t_start = time()

    for (qi, q) in enumerate(questions)
        q_t0 = time()
        # Category context: given (one-hot) or inferred soft posterior π.
        cat_idx = inferred ? 0 : findfirst(==(q.category), CATEGORIES)
        π = inferred ? category_posteriors[q.id] : Float64[]
        # Per-tool reliability belief at this question: a single Beta under
        # the given category, or the π-marginalised MixturePrevision (B2c)
        # when categories are inferred. rel_betas is constant within a
        # question (learning happens after the decision).
        rel(t) = inferred ? marginal_reliability(rel_betas[t, :], π) : rel_betas[t, cat_idx]
        relmean(t) = inferred ? expect(rel(t), r -> r) : mean(rel_betas[t, cat_idx])
        answer_measure = CategoricalMeasure(Finite(Float64[0, 1, 2, 3]))
        available = collect(1:n_tools)
        tool_responses = Dict{Int,Int}()
        q_cost = 0.0

        if greedy
            # Query tool with highest E[reliability], submit its answer directly
            best_t = argmax(relmean, 1:n_tools)  # credence-lint: allow — precedent:baseline-comparison — greedy baseline: argmax over mean reliability, intentionally non-Bayesian for paper comparison
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
                k = ANSWER_KERNEL(rel(t), 4.0)
                answer_measure = UPDATE_ON_RESPONSE(answer_measure, k, Float64(response))
            end
            # Submit argmax or abstain based on EU
            w = weights(answer_measure)
            best_idx = argmax(w)
            submitted = Int(answer_measure.space.values[best_idx])
            utility_vec = [Float64(v == submitted ? REWARD_CORRECT : PENALTY_WRONG)
                           for v in support(answer_measure.space)]
            eu_submit = expect(answer_measure, Tabular(utility_vec))
            if eu_submit > 0 || !allow_abstain  # credence-lint: allow — precedent:compute-on-weights — EU correctly computed via expect(m, Tabular); comparison is the host decision threshold
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
                rel_measures = [rel(t) for t in available]
                costs_jl = Float64[tools[t].cost for t in available]
                # qa_benchmark assumes all tools always respond (no coverage
                # uncertainty in this domain), so pass uniform 1.0 — preserves
                # the pre-coverage agent-step behaviour.
                cov_probs_jl = ones(Float64, length(available))

                result = AGENT_STEP(answer_measure, rel_measures, costs_jl,
                                    cov_probs_jl,
                                    REWARD_CORRECT, REWARD_ABSTAIN, PENALTY_WRONG)
                action_type = Int(result[1])
                action_arg = Int(result[2])

                if action_type == 2  # query tool
                    tool_local_idx = action_arg + 1  # DSL is 0-indexed
                    tool_idx = available[tool_local_idx]
                    q_cost += tools[tool_idx].cost

                    response = response_table[qi, tool_idx]
                    tool_responses[tool_idx] = response

                    k = ANSWER_KERNEL(rel(tool_idx), 4.0)
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
            if inferred
                # Posterior-weighted (`:post`, issue #111) or soft-credit
                # (`:soft`, B2c) update across all categories. Both route every
                # belief change through `condition` (no host-side Bayes).
                credit_update = credit === :soft ? update_reliability : post_update
                for (t, resp) in tool_responses
                    rel_betas[t, :] = credit_update(
                        rel_betas[t, :], π, resp == q.correct_index ? 1 : 0)
                end
            else
                for (t, resp) in tool_responses
                    rel_betas[t, cat_idx] = condition(
                        rel_betas[t, cat_idx], RELIABILITY_KERNEL,
                        resp == q.correct_index ? 1.0 : 0.0)
                end
            end
        end
    end

    wall_time = time() - t_start
    total_score = total_reward - total_tool_cost
    SeedResult(0, records, total_score, total_reward, total_tool_cost, wall_time)
end

# --- Baseline agents ---

# --- Horizon-aware VOI agent (Paper 1 B4) ---

"""
    run_horizon_seed(tools, questions, response_table; init_reliability=nothing)

Horizon-aware VOI agent (DSL `horizon-step`): per question, submit one tool and
optionally probe one tool for information, choosing the pair that maximises the
expected best-submit over the remaining same-category questions (pure EU-max over
the horizon — exploration emergent). Oracle condition (categories given): a single
Beta per (tool, category), per-category horizon. Beats greedy here (the
exploration the myopic `agent-step` lacks); under inferred categories the
advantage is denied by attribution noise — see scripts/paper1-horizon-inferred.jl.
"""
function run_horizon_seed(tools::Vector{SimulatedTool},
                          questions::Vector{Question},
                          response_table::Matrix{Int};
                          init_reliability::Union{Nothing,AbstractMatrix}=nothing)
    n_tools = length(tools); n_cats = length(CATEGORIES)
    rel_betas = init_reliability === nothing ?
        [BetaPrevision(1.0, 1.0) for _ in 1:n_tools, _ in 1:n_cats] : copy(init_reliability)
    costs = Float64[tools[t].cost for t in 1:n_tools]
    cat_of = [findfirst(==(q.category), CATEGORIES) for q in questions]
    # remaining same-category questions incl. current (stream order) = the horizon
    remaining = zeros(Int, length(questions)); seen = zeros(Int, n_cats)
    for qi in length(questions):-1:1
        c = cat_of[qi]; seen[c] += 1; remaining[qi] = seen[c]
    end
    records = QuestionResult[]; total_reward = 0.0; total_tool_cost = 0.0; t_start = time()
    for (qi, q) in enumerate(questions)
        q_t0 = time(); ci = cat_of[qi]
        rels = [rel_betas[t, ci] for t in 1:n_tools]
        res = HORIZON_STEP(rels, costs, Float64(remaining[qi]),
                           REWARD_CORRECT, REWARD_ABSTAIN, PENALTY_WRONG)
        s = Int(res[1]) + 1; r = Int(res[2])
        tool_responses = Dict{Int,Int}()
        q_cost = costs[s]
        submitted = response_table[qi, s]; tool_responses[s] = submitted
        was_correct = submitted == q.correct_index
        reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
        rel_betas[s, ci] = condition(rel_betas[s, ci], RELIABILITY_KERNEL, was_correct ? 1.0 : 0.0)
        if r >= 0
            rt = r + 1; q_cost += costs[rt]
            resp_r = response_table[qi, rt]; tool_responses[rt] = resp_r
            rel_betas[rt, ci] = condition(rel_betas[rt, ci], RELIABILITY_KERNEL,
                                          (resp_r == q.correct_index) ? 1.0 : 0.0)
        end
        total_reward += reward; total_tool_cost += q_cost
        push!(records, QuestionResult(q.id, q.category, collect(keys(tool_responses)),
            tool_responses, submitted, was_correct, reward, q_cost, time() - q_t0, 0, 0))
    end
    SeedResult(0, records, total_reward - total_tool_cost,
               total_reward, total_tool_cost, time() - t_start)
end

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
    credit = :post   # inferred-category credit rule (issue #111); :soft repro's the paper
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
        elseif arg == "--credit" && i < length(ARGS)
            credit = Symbol(ARGS[i + 1])
        elseif startswith(arg, "--credit=")
            credit = Symbol(split(arg, "=")[2])
        end
    end
    credit in (:post, :soft) ||
        error("--credit must be 'post' (default) or 'soft' (paper repro), got '$credit'")
    if isempty(models); models = ["llama3.1"]; end
    (; n_seeds, include_llm, llm_only, ablation, models, credit)
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
    # Inferred-category posteriors from the offline embedding fixture (if
    # present): adds the "bayesian_inferred" agent — the Bayesian agent
    # under fair (inferred, not given) categories.
    cat_post = isfile(joinpath(FIXTURE_DIR, "question_embeddings.json")) ?
        category_posteriors_from_fixture() : nothing
    cat_post === nothing && println(stderr, "  (no embedding fixture — skipping bayesian_inferred)")
    cat_post !== nothing && println(stderr,
        "  inferred-category credit rule: :$(args.credit)" *
        (args.credit === :soft ? " (B2c — reproduces the paper's committed numbers)" :
                                 " (posterior-weighted, issue #111 default; --credit soft for paper repro)"))
    for seed in 0:args.n_seeds-1
        rng = MersenneTwister(seed)
        questions = get_questions(; seed)
        response_table = generate_response_table(tools, questions, rng)

        random_rng = MersenneTwister(seed + 1_000_000)

        agents = Tuple{String,Function}[
            ("bayesian",    () -> run_bayesian_seed(tools, questions, response_table)),
            ("horizon",     () -> run_horizon_seed(tools, questions, response_table)),
            ("single_best", () -> run_single_best_seed(tools, questions, response_table)),
            ("random",      () -> run_random_seed(tools, questions, response_table, random_rng)),
            ("all_tools",   () -> run_all_tools_seed(tools, questions, response_table)),
        ]
        if cat_post !== nothing
            # Paper 1 fair-condition roster: every category-using agent run
            # under *inferred* (not given) categories, paired on the same seed.
            # The oracle counterparts (`bayesian`, `ablation_*`) stay only as a
            # legacy "price of inference" skyline; they are NOT on the fair
            # Pareto. The LLM agents (raw question text, no category) are reused
            # from the saved DB — see docs/paper1 B4.
            push!(agents, ("bayesian_inferred",
                () -> run_bayesian_seed(tools, questions, response_table; category_posteriors=cat_post, credit=args.credit)))
            push!(agents, ("greedy_inferred",
                () -> run_bayesian_seed(tools, questions, response_table; greedy=true, category_posteriors=cat_post, credit=args.credit)))
            push!(agents, ("no_voi_inferred",
                () -> run_bayesian_seed(tools, questions, response_table; use_voi=false, category_posteriors=cat_post, credit=args.credit)))
            push!(agents, ("no_learning_inferred",
                () -> run_bayesian_seed(tools, questions, response_table; learn=false, category_posteriors=cat_post, credit=args.credit)))
        end
        for (name, runner) in agents
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

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

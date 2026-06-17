# Role: eval
"""
    tb_dominance.jl — routing-dominance on the REAL Terminal-Bench matrix.

The empirical sibling of routing_dominance.jl: same EU-max mechanism, same brain
(`RoutingBrain.route` over K per-tier `StructureBMA` posteriors), same foil set —
but the synthetic IRT oracle is replaced by the MEASURED capability×cost matrix
(tb_matrix.jsonl: real `resolved` from each task's deterministic test suite, real
`cost_usd` = claude's own bill, real `num_turns`). And cost is CONTEXT-DEPENDENT:
E[cost | tier, X] = E[turns | tier, X] · c̄_tier, where E[turns] is read from a
Poisson-Gamma turns belief (the thrashing signal — a cheap model that loops burns
more turns) and c̄_tier is the measured \$/turn for that tier. This is the thesis
the toy could not show: the cheapest-priced model is not the cheapest-cost model.

Claim (held-out tasks, scored on realized outcomes — non-causal read-out, the
eval carve-out): the per-profile EU-max router weakly dominates every fixed
single-tier policy and the deployable foils (argmax-accuracy, best-fixed table,
RouteLLM-style threshold), and trails only the clairvoyant FrugalGPT cascade
(which knows each rung's success before paying). Every belief update is
`condition`; the routing argmax is the single `optimise` inside `route`.

Run:
    julia --project=<repo-root> apps/credence-pi/eval/tb_dominance.jl \
        apps/credence-pi/eval/live_ab/results/tb_matrix_full.jsonl \
        [--seeds 20] [--train-frac 0.6] [--out .../tb_dominance.summary.json]
"""

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence
using Random: MersenneTwister, shuffle!
using Printf
using JSON3

_mean(v) = isempty(v) ? 0.0 : sum(v) / length(v)

include(joinpath(@__DIR__, "..", "brain", "feature_brain.jl"))
using .FeatureBrain: StructureBMA, build_model, build_prior, observe, context_from_features
include(joinpath(@__DIR__, "..", "brain", "routing_brain.jl"))
using .RoutingBrain: route, posterior_accuracy

const TIERS = ["haiku", "sonnet", "opus"]   # cheap → dear (per-token price order)

# Feature schema for the routing belief. difficulty/category are the task's declared
# correlates of hardness; length is a prompt-only bucket (so the belief never sees the
# label). The StructureBMA auto-discovers which of these drive correctness.
const FEATURES = ["difficulty", "category", "length"]
const ALPHA0, BETA0, PEDGE = 2.0, 2.0, 0.5

# Profiles = the dollar value of a correct solve (Savage utility; one belief, these
# differ). Swept to trace the cost/quality frontier — each fixed policy is one point,
# the EU-max router adapts across the whole sweep (the Wald complete-class core).
const PROFILES = [("cost-hawk", 0.25), ("balanced", 1.0), ("quality-hawk", 5.0)]

# Poisson-Gamma turns-belief prior (weakly-informative): Gamma(α,β), E[turns]=α/β.
# α0=1, β0=0.1 ⇒ prior mean 10 turns, washed out by data. NOT tuned to the result.
const TURNS_A0, TURNS_B0 = 1.0, 0.1

struct TaskRow
    id::String
    difficulty::String
    category::String
    length::String
    resolved::Vector{Bool}     # per tier
    cost::Vector{Float64}      # per tier (realized; lower bound on timeouts)
    turns::Vector{Float64}     # per tier
end

length_bucket(instr_len) = instr_len >= 500 ? "long" : "short"

featuredict(t::TaskRow) = Dict("difficulty" => t.difficulty,
                               "category" => t.category, "length" => t.length)

# ── Load the measured matrix → one TaskRow per task with all K tiers present ──
function load_matrix(path::String)
    rows = Dict{String, Dict{String, Any}}()   # task -> tier -> row
    feats = Dict{String, NTuple{3, String}}()
    for ln in eachline(path)
        isempty(strip(ln)) && continue
        r = JSON3.read(ln)
        task = String(r.task); tier = String(r.tier)
        get!(rows, task, Dict{String, Any}())[tier] = r
        feats[task] = (String(something(r.difficulty, "unknown")),
                       String(something(r.category, "unknown")),
                       length_bucket(get(r, :instr_len, 0)))
    end
    tasks = TaskRow[]
    skipped = String[]
    for (task, byt) in rows
        if !all(t -> haskey(byt, t), TIERS)
            push!(skipped, task); continue
        end
        d, c, l = feats[task]
        resolved = Bool[byt[t].resolved for t in TIERS]
        cost = Float64[something(byt[t].cost_usd, 0.0) for t in TIERS]
        turns = Float64[Float64(something(byt[t].num_turns, 0)) for t in TIERS]
        push!(tasks, TaskRow(task, d, c, l, resolved, cost, turns))
    end
    isempty(skipped) || @warn "tasks missing some tiers (skipped)" tasks=skipped
    tasks
end

# Feature value-sets observed in the data (so the schema matches what's present).
function feature_values(tasks::Vector{TaskRow})
    diffs = sort(unique(t.difficulty for t in tasks))
    cats  = sort(unique(t.category for t in tasks))
    lens  = sort(unique(t.length for t in tasks))
    [diffs, cats, lens]
end

# ── Beliefs ──────────────────────────────────────────────────────────────────
# Resolved: K per-tier StructureBMA posteriors, trained by the canonical `observe`
# (condition under the hood) on the train tasks' real outcomes.
function train_tops(model::StructureBMA, train::Vector{TaskRow})
    tops = [build_prior(model) for _ in TIERS]
    for t in train, a in eachindex(TIERS)
        tops[a] = observe(model, tops[a], context_from_features(model, featuredict(t)),
                          t.resolved[a] ? 1 : 0)
    end
    tops
end

# Turns: Poisson-Gamma per (tier, difficulty) cell — the thrashing belief. condition the
# Gamma prior on each train task's observed integer turn-count via the Poisson kernel;
# E[turns|tier,cell] = expect(posterior, Identity). Cost rate c̄_tier = Σcost/Σturns over
# train (measured \$/turn). E[cost|tier,X] = E[turns|tier,cell]·c̄_tier — the
# context-dependent cost fed to `route` (a cheap model that thrashes costs more).
_poisson_kernel() = Kernel(PositiveReals(), Finite(collect(0:1000)), λ -> λ,
                           (h, o) -> 0.0; likelihood_family = Poisson())

struct CostBelief
    turns_mean::Dict{Tuple{Int, String}, Float64}   # (tier, difficulty) -> E[turns]
    rate::Vector{Float64}                            # c̄_tier ($/turn)
end

function train_cost(train::Vector{TaskRow}, diffs::Vector{String})
    k = _poisson_kernel()
    turns_mean = Dict{Tuple{Int, String}, Float64}()
    for a in eachindex(TIERS), d in diffs
        g = GammaPrevision(TURNS_A0, TURNS_B0)
        for t in train
            t.difficulty == d || continue
            n = round(Int, t.turns[a])
            n >= 0 && (g = condition(g, k, n))
        end
        turns_mean[(a, d)] = Float64(expect(g, Identity()))   # E[turns] = α/β
    end
    # measured $/turn per tier (Σcost/Σturns over train; total, not per cell)
    rate = Float64[]
    for a in eachindex(TIERS)
        sc = sum(t.cost[a] for t in train; init = 0.0)
        st = sum(t.turns[a] for t in train; init = 0.0)
        push!(rate, st > 0 ? sc / st : 0.0)
    end
    CostBelief(turns_mean, rate)
end

# E[cost|tier,X] vector for a task (context-dependent: turns belief × tier rate).
costs_at(cb::CostBelief, t::TaskRow) =
    Float64[get(cb.turns_mean, (a, t.difficulty), 0.0) * cb.rate[a] for a in eachindex(TIERS)]

# ── Arms → routed tier index per test task ───────────────────────────────────
# EU-max: the brain's `route`, fed the context-dependent cost vector (Invariant 1:
# argmax is the single `optimise`; cost is prepared utility data).
eu_routed(model, tops, cb, test, reward) =
    Int[route(model, tops, context_from_features(model, featuredict(t)), costs_at(cb, t), reward)
        for t in test]

# Deployable foils (baseline-comparison precedent): they see the SAME features and the
# SAME train data; they differ only in the DECISION RULE. Built from train empirics.
function empirical(train::Vector{TaskRow}, diffs, cats, lens)
    # per (tier, full-cell) accuracy and mean cost from train
    cells = [(d, c, l) for d in diffs for c in cats for l in lens]
    acc = Dict{Tuple{Int, NTuple{3, String}}, Float64}()
    cost = Dict{Tuple{Int, NTuple{3, String}}, Float64}()
    for a in eachindex(TIERS), ck in cells
        rs = [t for t in train if (t.difficulty, t.category, t.length) == ck]
        acc[(a, ck)]  = isempty(rs) ? 0.5 : _mean(Float64[t.resolved[a] for t in rs])
        cost[(a, ck)] = isempty(rs) ? 0.0 : _mean(Float64[t.cost[a] for t in rs])
    end
    acc, cost, cells
end

cellof(t::TaskRow) = (t.difficulty, t.category, t.length)

# credence-lint: allow — precedent:baseline-comparison — most-accurate-per-cell foil (cost-blind; ties→cheaper tier)
pick_argmax_acc(acc) = (t) -> argmin([(-acc[(a, cellof(t))], a) for a in eachindex(TIERS)])

# credence-lint: allow — precedent:baseline-comparison — best single profile-blind table (avg over profiles)
function pick_best_fixed(acc, cost)
    score(a, ck) = _mean(Float64[r * acc[(a, ck)] - cost[(a, ck)] for (_, r) in PROFILES])
    (t) -> argmax([score(a, cellof(t)) for a in eachindex(TIERS)])
end

# credence-lint: allow — precedent:baseline-comparison — RouteLLM-style cheap-vs-strong threshold on the cheap tier's accuracy
function pick_threshold(acc, cells)
    ca = Float64[acc[(1, ck)] for ck in cells]
    cut = (minimum(ca) + maximum(ca)) / 2.0
    (t) -> acc[(1, cellof(t))] >= cut ? 1 : length(TIERS)
end

table_routed(pick, test) = Int[pick(t) for t in test]

# ── Welfare read-out: realized, non-causal. Score the routed tier on each test task's
# MEASURED outcome: welfare = reward·resolved − cost. ──
realized_welfare(routed::Vector{Int}, test::Vector{TaskRow}, reward) =
    sum(reward * test[i].resolved[routed[i]] - test[i].cost[routed[i]]
        for i in eachindex(test); init = 0.0)

realized_solves(routed::Vector{Int}, test::Vector{TaskRow}) =
    sum(test[i].resolved[routed[i]] for i in eachindex(test); init = 0)
realized_cost(routed::Vector{Int}, test::Vector{TaskRow}) =
    sum(test[i].cost[routed[i]] for i in eachindex(test); init = 0.0)

# credence-lint: allow — precedent:baseline-comparison — clairvoyant FrugalGPT cascade upper bound (cheapest-first, stop at first real success, cumulative cost)
function cascade(test::Vector{TaskRow}, reward)
    w = 0.0; solves = 0; cost = 0.0
    for t in test
        paid = 0.0; done = false
        for a in eachindex(TIERS)        # cheapest-first
            paid += t.cost[a]
            if t.resolved[a]; done = true; break; end
        end
        cost += paid
        solves += done ? 1 : 0
        w += (done ? reward : 0.0) - paid
    end
    (w, solves, cost)
end

# EU-gated escalation — the credence-pi metareasoning arm. Walk tiers cheapest→
# dearest; at each rung (given cheaper rungs already failed) TRY iff its expected
# utility is positive, reward·E[θ_a|X] ≥ E[cost_a|X] (the same per-tier resolved
# belief + context-cost used by `route`); stop at the first OBSERVED solve. This is
# deployable for testable work (the test suite is the verifier), captures the full
# capability ladder WITHOUT predicting the boundary, and the EU gate stops it
# over-escalating when a solve isn't worth the spend (where the naive cascade
# over-pays). Myopic gate (ignores the option value of still-dearer rungs ⇒
# conservative, under-escalates). Scored on realized outcomes (non-causal read-out,
# mirroring eu_routed; in the live brain the gate is the canonical `optimise`).
function escalation_eu(model, tops, cb, test::Vector{TaskRow}, reward)
    w = 0.0; solves = 0; cost = 0.0
    for t in test
        X = context_from_features(model, featuredict(t))
        ec = costs_at(cb, t)
        paid = 0.0; solved = false
        for a in eachindex(TIERS)                       # cheapest → dearest
            θ = posterior_accuracy(model, tops[a], X)    # E[P(resolved_a | X)]
            reward * θ >= ec[a] || break                 # EU gate: stop escalating
            paid += t.cost[a]                            # pay this rung (realized)
            if t.resolved[a]; solved = true; break; end  # observed solve → stop
        end
        cost += paid; solves += solved ? 1 : 0
        w += (solved ? reward : 0.0) - paid
    end
    (w, solves, cost)
end

# ── Harness ──────────────────────────────────────────────────────────────────
function parse_args(argv)
    a = Dict{String, Any}("path" => "", "seeds" => 20, "train-frac" => 0.6, "out" => "")
    i = 1
    while i <= length(argv)
        t = argv[i]
        if t == "--seeds";          a["seeds"] = parse(Int, argv[i+1]);          i += 2
        elseif t == "--train-frac"; a["train-frac"] = parse(Float64, argv[i+1]); i += 2
        elseif t == "--out";        a["out"] = argv[i+1];                        i += 2
        elseif !startswith(t, "--") && isempty(a["path"]); a["path"] = t;        i += 1
        else; error("unknown arg $t"); end
    end
    isempty(a["path"]) && error("usage: tb_dominance.jl <matrix.jsonl> [--seeds N] [--train-frac f] [--out f]")
    a
end

function main()
    args = parse_args(ARGS)
    tasks = load_matrix(args["path"])
    @assert length(tasks) >= 4 "need ≥4 fully-measured tasks, got $(length(tasks))"
    fvals = feature_values(tasks)
    diffs, cats, lens = fvals

    # Arms scored per profile, accumulated over seeds (shuffled train/test splits).
    armnames = ["eu-max", "escalation-eu", "always-haiku", "always-sonnet", "always-opus",
                "argmax-accuracy", "best-fixed", "threshold(RouteLLM)", "oracle-cascade"]
    welf = Dict(p[1] => Dict(a => Float64[] for a in armnames) for p in PROFILES)
    solv = Dict(p[1] => Dict(a => Float64[] for a in armnames) for p in PROFILES)
    cst  = Dict(p[1] => Dict(a => Float64[] for a in armnames) for p in PROFILES)

    ids = [t.id for t in tasks]
    for seed in 0:(args["seeds"] - 1)
        rng = MersenneTwister(seed)
        ord = shuffle!(rng, collect(eachindex(ids)))
        ntr = max(2, round(Int, args["train-frac"] * length(ids)))
        trainset = Set(ids[ord[1:ntr]])
        train = [t for t in tasks if t.id in trainset]
        test  = [t for t in tasks if !(t.id in trainset)]
        isempty(test) && continue

        model = build_model(FEATURES, fvals; alpha0 = ALPHA0, beta0 = BETA0, p_edge = PEDGE)
        tops = train_tops(model, train)
        cb = train_cost(train, diffs)
        acc, cost, cells = empirical(train, diffs, cats, lens)

        picks = Dict(
            "always-haiku"         => fill(1, length(test)),
            "always-sonnet"        => fill(2, length(test)),
            "always-opus"          => fill(3, length(test)),
            "argmax-accuracy"      => table_routed(pick_argmax_acc(acc), test),
            "best-fixed"           => table_routed(pick_best_fixed(acc, cost), test),
            "threshold(RouteLLM)"  => table_routed(pick_threshold(acc, cells), test),
        )
        for (pname, reward) in PROFILES
            picks["eu-max"] = eu_routed(model, tops, cb, test, reward)
            for a in armnames
                if a == "oracle-cascade"
                    w, s, c = cascade(test, reward)
                elseif a == "escalation-eu"
                    w, s, c = escalation_eu(model, tops, cb, test, reward)
                else
                    r = picks[a]
                    w = realized_welfare(r, test, reward)
                    s = realized_solves(r, test); c = realized_cost(r, test)
                end
                push!(welf[pname][a], w); push!(solv[pname][a], Float64(s)); push!(cst[pname][a], c)
            end
        end
    end
    report(args, tasks, welf, solv, cst, armnames)
end

avg(v) = isempty(v) ? 0.0 : sum(v) / length(v)

function report(args, tasks, welf, solv, cst, armnames)
    bar = "="^86
    println("\n", bar)
    println("TERMINAL-BENCH ROUTING DOMINANCE  (real matrix; welfare = reward·solved − \$cost)")
    println(bar)
    println("tasks (all tiers measured): ", length(tasks),
            "   tiers: ", join(TIERS, " < "), "   seeds: ", args["seeds"],
            "   train-frac: ", args["train-frac"])
    nb = count(t -> any(t.resolved) && !all(t.resolved), tasks)
    println("tasks with a capability SPREAD (some tier solves, some fails): ", nb)
    for (pname, reward) in PROFILES
        println("\n[", pname, "]  reward(\$/solve)=", reward,
                "   (mean over seeds, held-out tasks)")
        println("  ", rpad("arm", 22), rpad("welfare", 12), rpad("solves", 9), "cost")
        ranked = sort(armnames; by = a -> -avg(welf[pname][a]))
        wbest = maximum(avg(welf[pname][a]) for a in armnames)
        for a in ranked
            mark = a == "eu-max" ? " ← EU-max" :
                   (avg(welf[pname][a]) ≈ wbest ? " ← best" : "")
            println("  ", rpad(a, 22),
                    rpad(@sprintf("%+.4f", avg(welf[pname][a])), 12),
                    rpad(@sprintf("%.1f", avg(solv[pname][a])), 9),
                    @sprintf("\$%.4f", avg(cst[pname][a])), mark)
        end
    end

    # CROSS-PROFILE dominance (the Wald complete-class claim): a deployment serves a
    # MIX of profiles; no single fixed policy is optimal across them, but one adaptive
    # arm can be. Scored by MINIMAX REGRET — each arm's worst-case shortfall vs the
    # best arm for that profile (scale-fair, unlike a raw sum which quality-hawk's big
    # numbers would dominate). Lowest worst-case regret = most robust single policy.
    best_w = Dict(pname => maximum(avg(welf[pname][a]) for a in armnames) for (pname, _) in PROFILES)
    worst_regret(a) = maximum(best_w[pname] - avg(welf[pname][a]) for (pname, _) in PROFILES)
    println("\nCROSS-PROFILE — worst-case regret over {", join((p for (p, _) in PROFILES), ", "),
            "} (one policy serving all profiles; lower = more robust):")
    for a in sort(armnames; by = worst_regret)
        mark = a in ("escalation-eu", "eu-max") ? "  ← credence-pi" : ""
        println("  ", rpad(a, 22), @sprintf("worst-regret %.4f", worst_regret(a)), mark)
    end

    # Per-profile margins for the headline credence-pi arm (EU-gated escalation) vs each
    # other arm (≥0 ⇒ escalation-eu ≥ arm; % = seed win-rate).
    head = "escalation-eu"
    println("\nDOMINANCE — ", head, " mean-welfare margin vs each arm, per profile:")
    print("  ", rpad("arm", 22)); for (p, _) in PROFILES; print(rpad(p, 16)); end; println()
    for a in armnames
        a == head && continue
        print("  ", rpad(a, 22))
        for (pname, _) in PROFILES
            rs = welf[pname][head] .- welf[pname][a]
            win = count(>=(-1e-12), rs) / length(rs)
            print(rpad(@sprintf("%+.4f(%.0f%%)", avg(rs), 100win), 16))
        end
        println()
    end
    println("  (oracle-cascade is the CLAIRVOYANT upper bound — knows each rung's success before paying;")
    println("   escalation-eu pays each rung it tries but the EU gate halts it where a solve isn't worth it.)")
    println(bar, "\n")

    if !isempty(args["out"])
        out = Dict{String, Any}(
            "config" => Dict("tasks" => length(tasks), "tiers" => TIERS,
                             "features" => FEATURES, "profiles" => Dict(PROFILES),
                             "seeds" => args["seeds"], "train_frac" => args["train-frac"]),
            "welfare_mean" => Dict(p => Dict(a => avg(welf[p][a]) for a in armnames) for (p, _) in PROFILES),
            "solves_mean"  => Dict(p => Dict(a => avg(solv[p][a]) for a in armnames) for (p, _) in PROFILES),
            "cost_mean"    => Dict(p => Dict(a => avg(cst[p][a]) for a in armnames) for (p, _) in PROFILES),
            "cross_profile_worst_regret" => Dict(a =>
                maximum(maximum(avg(welf[p][x]) for x in armnames) - avg(welf[p][a])
                        for (p, _) in PROFILES) for a in armnames),
        )
        mkpath(dirname(args["out"]))
        open(args["out"], "w") do io; JSON3.pretty(io, out); end
        println("summary → ", args["out"])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# ─────────────────────────────────────────────────────────────────────────────
# Principled INFERRED horizon-VOI — does B deliver the deployed (headline) win?
#
# Fair condition: categories inferred (LOO classifier π per question), reliability
# per-category (rel_betas[t][c]), B2c full-posterior-weighted update. The agent
# does NOT know future categories. Principled horizon-VOI: value a probe by its
# expected effect on EACH category's future questions, weighted by the expected
# remaining count of that category:
#
#   total(action) = marginal-submit-EU(current)
#                 + Σ_c (R'·p_c) · E_outcomes[ submit-value_c(rows after action) ]
#
# where R' = remaining-after-current, p_c = benchmark category mix (the agent
# knows the domain composition — a stated, generous assumption, the inferred
# analogue of the oracle-horizon idealisation), and the hypothetical updates use
# the B2c update_reliability(row, π, outcome). Reduces to the oracle mechanism
# when π is one-hot. greedy_inferred anchor must reproduce the host's 149.6.
# ─────────────────────────────────────────────────────────────────────────────
push!(LOAD_PATH, joinpath(normpath(joinpath(@__DIR__, "..")), "src"))
using Credence, Random, JSON3
using Credence: BetaPrevision
const QA = joinpath(normpath(joinpath(@__DIR__, "..")), "apps", "julia", "qa_benchmark")
include(joinpath(QA, "environment.jl"))
include(joinpath(QA, "category_inference.jl"))
include(joinpath(QA, "category_update.jl"))

const TOOLS = make_spec_tools(); const NT = length(TOOLS)
const COSTS = Float64[t.cost for t in TOOLS]
const NCAT = length(CATEGORIES)
const PMIX = Float64[count(q -> q.category == c, QUESTION_BANK) for c in CATEGORIES] ./ length(QUESTION_BANK)
avg(v) = sum(v) / length(v)

# Same LOO posteriors the host's inferred agents use.
function category_posteriors()
    bank = JSON3.read(read(joinpath(QA, "fixtures", "question_bank.json"), String))
    emb = JSON3.read(read(joinpath(QA, "fixtures", "question_embeddings.json"), String))
    byid = emb.embeddings; N = length(bank); D = emb.dim
    X = Matrix{Float64}(undef, N, D); cats = Vector{Symbol}(undef, N); ids = Vector{String}(undef, N)
    for (i, q) in enumerate(bank)
        X[i, :] = Float64.(byid[Symbol(q.id)]); cats[i] = Symbol(q.category); ids[i] = String(q.id)
    end
    posts = loo_category_inference(X, cats)
    Dict(ids[i] => [get(posts[i], Symbol(c), 0.0) for c in CATEGORIES] for i in 1:N)
end
const POST = category_posteriors()

# Credit-assignment rules (the upstream axis). soft = B2c (credit every category
# by π_c); hard = unit update on the argmax-π category only (no fractional leak).
using Credence: BetaPrevision
function hard_update(row::Vector{BetaPrevision}, π::Vector{Float64}, outcome)
    c = argmax(π); nr = copy(row)
    nr[c] = condition(row[c], WEIGHTED_RELIABILITY_KERNEL, (Float64(outcome), 1.0))
    nr
end
# posterior-weighted credit: the de Finettian refinement B2c approximates. Weight
# each category by π_c · P(outcome | θ_{t,c}) — a tool's correct answer credits the
# category where it's reliable, not the misclassified one. Reduces bimodal leakage
# without discarding π (unlike hard-credit).
function post_update(row::Vector{BetaPrevision}, π::Vector{Float64}, outcome)
    o = Float64(outcome)
    L = Float64[o == 1 ? mean(row[c]) : 1 - mean(row[c]) for c in 1:NCAT]
    w = π .* L; s = sum(w); w = s > 0 ? w ./ s : copy(π)
    BetaPrevision[condition(row[c], WEIGHTED_RELIABILITY_KERNEL, (o, w[c])) for c in 1:NCAT]
end
creditupd(row, π, o, credit) =
    credit == :hard ? hard_update(row, π, o) :
    credit == :post ? post_update(row, π, o) : update_reliability(row, π, o)

mrow(row) = Float64[mean(row[c]) for c in 1:NCAT]
marg_mean(row, π) = sum(π[c] * mean(row[c]) for c in 1:NCAT)           # marginal predictive correctness
# best single-submit EU among tools, for a category-c question, given rows
function scv(rows, c)
    best = -Inf
    for s in 1:NT
        m = mean(rows[s][c]); v = m * REWARD_CORRECT + (1 - m) * PENALTY_WRONG - COSTS[s]
        v > best && (best = v)
    end
    best
end

# principled horizon-VOI decision: returns (submit s, probe r or 0)
# λ scales the horizon (probe-intensity) term: λ=0 ⇒ never probe ⇒ greedy.
function decide(rows, π, Rprime; λ = 1.0)
    # precompute marginal predictive per tool
    mt = Float64[marg_mean(rows[t], π) for t in 1:NT]
    exprem = (λ * Rprime) .* PMIX                                      # expected remaining per category
    function future(rows2)                                            # Σ_c exprem_c · scv_c
        s = 0.0; for c in 1:NCAT; s += exprem[c] * scv(rows2, c); end; s
    end
    best = -Inf; bs = 1; br = 0
    for s in 1:NT
        immS = mt[s] * REWARD_CORRECT + (1 - mt[s]) * PENALTY_WRONG - COSTS[s]
        rows_sc = copy(rows); rows_sc[s] = update_reliability(rows[s], π, 1)
        rows_sw = copy(rows); rows_sw[s] = update_reliability(rows[s], π, 0)
        # submit-only s
        v = immS + mt[s] * future(rows_sc) + (1 - mt[s]) * future(rows_sw)
        v > best && (best = v; bs = s; br = 0)
        # submit s + probe r
        for r in 1:NT
            r == s && continue
            immP = immS - COSTS[r]
            rc = update_reliability(rows[r], π, 1); rw = update_reliability(rows[r], π, 0)
            rows_sc_rc = copy(rows_sc); rows_sc_rc[r] = rc
            rows_sc_rw = copy(rows_sc); rows_sc_rw[r] = rw
            rows_sw_rc = copy(rows_sw); rows_sw_rc[r] = rc
            rows_sw_rw = copy(rows_sw); rows_sw_rw[r] = rw
            v2 = immP + mt[s]*(mt[r]*future(rows_sc_rc) + (1-mt[r])*future(rows_sc_rw)) +
                       (1-mt[s])*(mt[r]*future(rows_sw_rc) + (1-mt[r])*future(rows_sw_rw))
            v2 > best && (best = v2; bs = s; br = r)
        end
    end
    (bs, br)
end

# cost-BLIND submit value for category c (greedy's submit rule): pick argmax-mean
# tool, score by its reliability − its cost.
function scv_blind(rows, c)
    sbest = 1; mbest = -Inf
    for s in 1:NT; m = mean(rows[s][c]); m > mbest && (mbest = m; sbest = s); end
    mbest * REWARD_CORRECT + (1 - mbest) * PENALTY_WRONG - COSTS[sbest]
end

# horizon-VOI with greedy's cost-blind submit + optional single horizon-probe.
# Submit = argmax marginal mean (identical to greedy). Probe r chosen by its
# horizon value (improvement to future cost-blind submits) − fee; probe iff > 0.
function decide_blind(rows, π, Rprime; τ = 0.0, credit = :soft)
    mt = Float64[marg_mean(rows[t], π) for t in 1:NT]
    s = argmax(mt)
    maximum(π) < τ && return (s, 0)        # confidence gate: don't probe when category is uncertain
    exprem = Rprime .* PMIX
    base = sum(exprem[c] * scv_blind(rows, c) for c in 1:NCAT)
    bestr = 0; bestv = 0.0   # no-probe baseline = 0 marginal value
    for r in 1:NT
        r == s && continue
        rc = creditupd(rows[r], π, 1, credit); rw = creditupd(rows[r], π, 0, credit)
        rows_c = copy(rows); rows_c[r] = rc; rows_w = copy(rows); rows_w[r] = rw
        fut = mt[r] * sum(exprem[c]*scv_blind(rows_c,c) for c in 1:NCAT) +
              (1-mt[r]) * sum(exprem[c]*scv_blind(rows_w,c) for c in 1:NCAT)
        v = fut - base - COSTS[r]
        v > bestv && (bestv = v; bestr = r)
    end
    (s, bestr)
end

function run(mode; seeds = 0:19, λ = 1.0, credit = :soft)   # mode = :greedy | :horizon | :hblind
    scores = Float64[]; accs = Float64[]; probes = Int[]
    for seed in seeds
        rng = MersenneTwister(seed); questions = get_questions(; seed)
        rt = generate_response_table(TOOLS, questions, rng)
        rows = [[BetaPrevision(1.0, 1.0) for _ in 1:NCAT] for _ in 1:NT]
        reward = 0.0; cost = 0.0; correct = 0; pq = 0; N = length(questions)
        for (qi, q) in enumerate(questions)
            π = POST[q.id]
            if mode == :greedy
                mt = Float64[marg_mean(rows[t], π) for t in 1:NT]; s = argmax(mt); r = 0
            elseif mode == :hblind
                (s, r) = decide_blind(rows, π, Float64(N - qi); τ = λ, credit = credit)
            else
                (s, r) = decide(rows, π, Float64(N - qi); λ = λ)     # R' = remaining after current
            end
            scor = rt[qi, s] == q.correct_index
            reward += scor ? REWARD_CORRECT : PENALTY_WRONG; cost += COSTS[s]; correct += scor ? 1 : 0
            rows[s] = creditupd(rows[s], π, scor ? 1 : 0, credit)
            if r != 0
                cost += COSTS[r]; pq += 1
                rows[r] = creditupd(rows[r], π, (rt[qi, r] == q.correct_index) ? 1 : 0, credit)
            end
        end
        push!(scores, reward - cost); push!(accs, correct / N); push!(probes, pq)
    end
    (score = avg(scores), acc = avg(accs), probes = avg(probes))
end

g = run(:greedy)
println("INFERRED (fair) condition, 20 seeds:")
println("  greedy_inferred (anchor)  score=$(round(g.score,digits=1))  acc=$(round(100g.acc,digits=1))%   [host: 149.6]")
println("  reference: bayesian_inferred (myopic VOI) 110.4\n")
println("  cost-AWARE-submit horizon-VOI, probe-intensity sweep (λ=0 = cost-aware no-probe):")
for λ in [0.0, 0.25, 0.5, 1.0]
    h = run(:horizon; λ = λ)
    println("    λ=$λ  score=$(round(h.score,digits=1))  acc=$(round(100h.acc,digits=1))%  probe-Q/seed=$(round(h.probes,digits=1))  Δvs-greedy=$(round(h.score-g.score,digits=1))")
end
println("\n  THE CREDIT-RULE TEST (the upstream axis, expert loop 3) — cost-blind submit + probing:")
for credit in [:soft, :hard, :post]
    gg = run(:greedy; credit = credit)
    hb = run(:hblind; λ = 0.0, credit = credit)
    gap = hb.score - gg.score
    flag = gap > 0 ? "  horizon > greedy ✓ (B REVIVES under this credit rule)" : "  greedy wins"
    println("    $(rpad(string(credit),5))  greedy=$(round(gg.score,digits=1))   horizon=$(round(hb.score,digits=1))  (probe-Q/seed=$(round(hb.probes,digits=1)))  gap=$(round(gap,digits=1))$flag")
end

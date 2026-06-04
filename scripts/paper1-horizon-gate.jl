# ─────────────────────────────────────────────────────────────────────────────
# Horizon-aware VOI — the DECOUPLED gate (expert loop 2; the binding run).
#
# The single-query gate tested the wrong MDP: it COUPLED probe and submit (query
# one tool, submit its answer, eat the −5 if wrong), which structurally cannot
# express VOI's defining move — probe-without-submit — while expressing greedy's
# move exactly (hence its 189.4 match). The 248 headroom was itself produced by a
# DECOUPLED mechanism (explore-then-exploit submits the best-LEARNED answer). So
# this run uses the deployable action space:
#
#   per question: submit tool s (pay cost_s, learn s, score by s's correctness)
#                 + optionally probe one tool r≠s (pay cost_r, learn r, DON'T submit r)
#
# Exploration now costs the probe FEE, not the answer risk. Exact finite-horizon
# backward induction (no truncation, no myopic-tail bootstrap). Probe-budget K per
# category swept to confirm non-binding (⇒ effectively the unbounded ceiling).
#
# PRE-COMMITTED STOPPING RULE (binding either way):
#   benchmark decoupled optimum  > 189  ⇒ B alive: build the depth-d DSL version.
#   benchmark decoupled optimum  ≤ 189  ⇒ ship A (stronger: the true decoupled
#                                         ceiling can't beat optimism in this regime).
# ─────────────────────────────────────────────────────────────────────────────
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "apps/julia/qa_benchmark/environment.jl"))
using Random

const TOOLS  = make_spec_tools()
const NT     = length(TOOLS)
const COSTS  = Float64[t.cost for t in TOOLS]
const CATIDX = Dict(c => i for (i, c) in enumerate(CATEGORIES))
const HORIZONS = [count(q -> q.category == c, QUESTION_BANK) for c in CATEGORIES]
avg(v) = sum(v) / length(v)

imm_submit(p, s) = p * REWARD_CORRECT + (1 - p) * PENALTY_WRONG - COSTS[s]
@inline pred(b, t) = b[2t-1] / (b[2t-1] + b[2t])
@inline succ(b, t) = ntuple(i -> i == 2t-1 ? b[i] + 1 : b[i], 8)
@inline fail(b, t) = ntuple(i -> i == 2t   ? b[i] + 1 : b[i], 8)
@inline upd(b, t, correct) = correct ? succ(b, t) : fail(b, t)
@inline pack(b, h) = (UInt64(b[1]) | UInt64(b[2])<<5 | UInt64(b[3])<<10 | UInt64(b[4])<<15 |
                      UInt64(b[5])<<20 | UInt64(b[6])<<25 | UInt64(b[7])<<30 | UInt64(b[8])<<35 |
                      UInt64(h)<<40)

# Per-(H,K) exact solve. probes_used at (b,h) = (Σb−8) − (H−h); probe allowed iff < K.
# Globals set before each solve.
HH::Int = 0; KK::Int = 0
const MEMO = Dict{UInt64,Float64}()

function Vd(b::NTuple{8,Int}, h::Int)
    h == 0 && return 0.0
    key = pack(b, h)
    hit = get(MEMO, key, NaN); isnan(hit) || return hit
    probes_used = (sum(b) - 8) - (HH - h)
    best = -Inf
    for s in 1:NT
        ps = pred(b, s); base = imm_submit(ps, s)
        # submit-only (always available)
        q = base + ps * Vd(succ(b, s), h - 1) + (1 - ps) * Vd(fail(b, s), h - 1)
        q > best && (best = q)
        # submit s + probe r (if budget remains)
        if probes_used < KK
            for r in 1:NT
                r == s && continue
                pr = pred(b, r); c = base - COSTS[r]
                bss = upd(b, s, true);  bsf = upd(b, s, false)
                q2 = c +
                    ps*pr           * Vd(upd(bss, r, true),  h-1) +
                    ps*(1-pr)       * Vd(upd(bss, r, false), h-1) +
                    (1-ps)*pr       * Vd(upd(bsf, r, true),  h-1) +
                    (1-ps)*(1-pr)   * Vd(upd(bsf, r, false), h-1)
                q2 > best && (best = q2)
            end
        end
    end
    MEMO[key] = best
    best
end

# Optimal action (s, r) at (b,h); r==0 means no probe. Reuses memo.
function act_d(b::NTuple{8,Int}, h::Int)
    probes_used = (sum(b) - 8) - (HH - h)
    best = -Inf; bs = 1; br = 0
    for s in 1:NT
        ps = pred(b, s); base = imm_submit(ps, s)
        q = base + ps * Vd(succ(b, s), h - 1) + (1 - ps) * Vd(fail(b, s), h - 1)
        q > best && (best = q; bs = s; br = 0)
        if probes_used < KK
            for r in 1:NT
                r == s && continue
                pr = pred(b, r); c = base - COSTS[r]
                bss = upd(b, s, true); bsf = upd(b, s, false)
                q2 = c + ps*pr*Vd(upd(bss,r,true),h-1) + ps*(1-pr)*Vd(upd(bss,r,false),h-1) +
                         (1-ps)*pr*Vd(upd(bsf,r,true),h-1) + (1-ps)*(1-pr)*Vd(upd(bsf,r,false),h-1)
                q2 > best && (best = q2; bs = s; br = r)
            end
        end
    end
    (bs, br)
end

# Solve all distinct horizons at budget K; return Dict H => frozen policy memo.
# We keep MEMO populated per H during simulation, so simulate immediately per H.
function run_K(K::Int; seeds = 0:19)
    global KK = K
    scores = Float64[]; accs = Float64[]; probe_qs = Int[]
    pulls = zeros(Float64, NT)
    # Pre-solve each distinct horizon once (fills MEMO), keep until done with K.
    solved = Dict{Int,Bool}()
    for seed in seeds
        rng = MersenneTwister(seed); questions = get_questions(; seed)
        rt = generate_response_table(TOOLS, questions, rng)
        bycat = [Int[] for _ in 1:length(CATEGORIES)]
        for (qi, q) in enumerate(questions); push!(bycat[CATIDX[q.category]], qi); end
        reward = 0.0; cost = 0.0; correct = 0; pq = 0
        for (ci, qis) in enumerate(bycat)
            H = length(qis); global HH = H
            if !get(solved, H, false); Vd(ntuple(_->1,8), H); solved[H] = true; end
            b = ntuple(_->1, 8)
            for (i, qi) in enumerate(qis)
                h = H - i + 1; global HH = H
                (s, r) = act_d(b, h)
                scor = rt[qi, s] == questions[qi].correct_index
                reward += scor ? REWARD_CORRECT : PENALTY_WRONG; cost += COSTS[s]
                correct += scor ? 1 : 0; pulls[s] += 1; b = upd(b, s, scor)
                if r != 0
                    rcor = rt[qi, r] == questions[qi].correct_index
                    cost += COSTS[r]; pulls[r] += 1; pq += 1; b = upd(b, r, rcor)
                end
            end
        end
        push!(scores, reward - cost); push!(accs, correct / length(questions)); push!(probe_qs, pq)
    end
    empty!(MEMO)  # free before next K
    (score = avg(scores), acc = avg(accs), probes = avg(probe_qs), pulls = pulls ./ length(seeds))
end

# argmax-mean greedy (no probe) under matched prior — the true greedy baseline.
function synth_greedy(; rng, N = 3000)
    s = 0.0
    for _ in 1:N
        θ = rand(rng, NT, length(CATEGORIES)); total = 0.0
        for (ci, H) in enumerate(HORIZONS)
            b = ntuple(_->1, 8)
            for _ in 1:H
                t = argmax(Float64[pred(b, j) for j in 1:NT])
                scor = rand(rng) < θ[t, ci]
                total += scor ? REWARD_CORRECT : PENALTY_WRONG; total -= COSTS[t]
                b = upd(b, t, scor)
            end
        end
        s += total
    end
    s / N
end

# Matched-prior synthetic for the decoupled model (θ ~ Beta(1,1)); decoupled
# optimal should beat argmax-greedy by MORE than the single-query +9.2 (cheap
# probing helps more when the prior is right).
function synth_decoupled(K::Int; rng, N = 3000)
    global KK = K; so = 0.0
    solved = Dict{Int,Bool}()
    for _ in 1:N
        θ = rand(rng, NT, length(CATEGORIES))
        total = 0.0
        for (ci, H) in enumerate(HORIZONS)
            global HH = H
            if !get(solved, H, false); Vd(ntuple(_->1,8), H); solved[H] = true; end
            b = ntuple(_->1, 8)
            for i in 1:H
                h = H - i + 1; global HH = H
                (s, r) = act_d(b, h)
                scor = rand(rng) < θ[s, ci]
                total += scor ? REWARD_CORRECT : PENALTY_WRONG; total -= COSTS[s]
                b = upd(b, s, scor)
                if r != 0
                    rcor = rand(rng) < θ[r, ci]; total -= COSTS[r]; b = upd(b, r, rcor)
                end
            end
        end
        so += total
    end
    empty!(MEMO); so / N
end

# ── Run ──────────────────────────────────────────────────────────────────────
println("DECOUPLED gate — exact backward induction, probe-without-submit.")
println("category horizons: ", join(["$c=$h" for (c, h) in zip(CATEGORIES, HORIZONS)], "  "), "\n")
println("benchmark, 20 seeds (greedy=189.4 from gate 1; K=0 = single-query optimum 180.8):")
results = Tuple{Int,Float64,Float64,Float64}[]
@time for K in 0:6
    r = run_K(K)
    push!(results, (K, r.score, r.acc, r.probes))
    flag = K == 0 ? "  ⟵ single-query optimum (no probe)" : (r.score > 189.4 ? "  ⟵ > greedy 189.4" : "")
    println("  K=$K  score=$(round(r.score,digits=1))  acc=$(round(100r.acc,digits=1))%  probe-Q/seed=$(round(r.probes,digits=1))$flag")
    K == 6 && println("    pulls/seed [web,KB,calc,llm]: ", round.(r.pulls, digits=2))
end

best = maximum(x -> x[2], results)
println("\n  decoupled optimum (best over K) = $(round(best, digits=1))   vs greedy 189.4")

println("\nmatched-prior control (θ ~ Beta(1,1)); single-query gave decoupled-opt−greedy = +9.2,")
println("expect LARGER here (cheap probing helps more when the prior is right):")
let
    g = synth_greedy(; rng = MersenneTwister(777))
    o = synth_decoupled(6; rng = MersenneTwister(777))
    println("  argmax-greedy = $(round(g,digits=1))   decoupled optimal = $(round(o,digits=1))   Δ = $(round(o-g,digits=1))")
end

println("\nVERDICT:")
if best > 189.4
    println("  decoupled optimum $(round(best,digits=1)) > 189.4  ⇒  B ALIVE — build the depth-d DSL version.")
else
    println("  decoupled optimum $(round(best,digits=1)) ≤ 189.4  ⇒  SHIP A (airtight: true decoupled ceiling can't beat optimism).")
end

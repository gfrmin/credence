# ─────────────────────────────────────────────────────────────────────────────
# Depth-d horizon-VOI — de-risk the DSL build before writing DSL machinery.
#
# The exact gate (d=∞) reaches 211.8. The DSL can only run a BOUNDED depth-d
# receding-horizon lookahead with a myopic tail (the expert's prime suspect).
# This sweeps d to answer: does a shallow d (tractable in the DSL) already clear
# greedy (189.4), or does only deep d work? If d≈2 clears greedy, the DSL build
# is viable; if not, I need a better tail before building.
#
# Same decoupled mechanism (submit s + optional probe r). Receding-horizon: at
# each question the agent re-plans a fresh depth-d lookahead, acts, observes the
# real outcome, re-plans. Myopic tail at depth exhaustion: (h remaining)·best-submit.
# ─────────────────────────────────────────────────────────────────────────────
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "apps/julia/qa_benchmark/environment.jl"))
using Random
const TOOLS = make_spec_tools(); const NT = length(TOOLS)
const COSTS = Float64[t.cost for t in TOOLS]
const CATIDX = Dict(c => i for (i, c) in enumerate(CATEGORIES))
avg(v) = sum(v) / length(v)

@inline pred(b, t) = b[2t-1] / (b[2t-1] + b[2t])
@inline succ(b, t) = ntuple(i -> i == 2t-1 ? b[i] + 1 : b[i], 8)
@inline fail(b, t) = ntuple(i -> i == 2t   ? b[i] + 1 : b[i], 8)
@inline upd(b, t, c) = c ? succ(b, t) : fail(b, t)
imm_submit(p, s) = p * REWARD_CORRECT + (1 - p) * PENALTY_WRONG - COSTS[s]
# best myopic one-question submit value (the tail building block)
submitval(b) = maximum(imm_submit(pred(b, s), s) for s in 1:NT)

const MEMO = Dict{Tuple{NTuple{8,Int},Int,Int},Float64}()

# Depth-limited lookahead value of being at (belief b, h questions left) with d
# further questions of exact lookahead remaining; myopic tail beyond d.
function LA(b::NTuple{8,Int}, h::Int, d::Int)
    h == 0 && return 0.0
    d == 0 && return h * submitval(b)               # myopic tail
    key = (b, h, d); hit = get(MEMO, key, NaN); isnan(hit) || return hit
    best = -Inf
    for s in 1:NT
        ps = pred(b, s); base = imm_submit(ps, s)
        v = base + ps * LA(succ(b, s), h-1, d-1) + (1-ps) * LA(fail(b, s), h-1, d-1)
        v > best && (best = v)
        for r in 1:NT
            r == s && continue
            pr = pred(b, r); bss = succ(b, s); bsf = fail(b, s)
            v2 = base - COSTS[r] +
                ps*pr*LA(succ(bss,r),h-1,d-1) + ps*(1-pr)*LA(fail(bss,r),h-1,d-1) +
                (1-ps)*pr*LA(succ(bsf,r),h-1,d-1) + (1-ps)*(1-pr)*LA(fail(bsf,r),h-1,d-1)
            v2 > best && (best = v2)
        end
    end
    MEMO[key] = best; best
end

# Receding-horizon action at (b,h) with planning depth D: argmax over actions of
# [immediate + E LA(·, h-1, D-1)]. Returns (submit s, probe r or 0).
function act(b::NTuple{8,Int}, h::Int, D::Int)
    best = -Inf; bs = 1; br = 0
    for s in 1:NT
        ps = pred(b, s); base = imm_submit(ps, s)
        v = base + ps*LA(succ(b,s),h-1,D-1) + (1-ps)*LA(fail(b,s),h-1,D-1)
        v > best && (best = v; bs = s; br = 0)
        for r in 1:NT
            r == s && continue
            pr = pred(b, r); bss = succ(b,s); bsf = fail(b,s)
            v2 = base - COSTS[r] +
                ps*pr*LA(succ(bss,r),h-1,D-1) + ps*(1-pr)*LA(fail(bss,r),h-1,D-1) +
                (1-ps)*pr*LA(succ(bsf,r),h-1,D-1) + (1-ps)*(1-pr)*LA(fail(bsf,r),h-1,D-1)
            v2 > best && (best = v2; bs = s; br = r)
        end
    end
    (bs, br)
end

function simulate(D; seeds = 0:19)
    scores = Float64[]; accs = Float64[]; probes = Int[]
    for seed in seeds
        rng = MersenneTwister(seed); questions = get_questions(; seed)
        rt = generate_response_table(TOOLS, questions, rng)
        bycat = [Int[] for _ in 1:length(CATEGORIES)]
        for (qi, q) in enumerate(questions); push!(bycat[CATIDX[q.category]], qi); end
        reward = 0.0; cost = 0.0; correct = 0; pq = 0
        for qis in bycat
            b = ntuple(_->1, 8); H = length(qis)
            for (i, qi) in enumerate(qis)
                h = H - i + 1; (s, r) = act(b, h, D)
                scor = rt[qi, s] == questions[qi].correct_index
                reward += scor ? REWARD_CORRECT : PENALTY_WRONG; cost += COSTS[s]
                correct += scor ? 1 : 0; b = upd(b, s, scor)
                if r != 0
                    cost += COSTS[r]; pq += 1
                    b = upd(b, r, rt[qi, r] == questions[qi].correct_index)
                end
            end
        end
        push!(scores, reward - cost); push!(accs, correct/length(questions)); push!(probes, pq)
    end
    (score = avg(scores), acc = avg(accs), probes = avg(probes))
end

println("depth-d horizon-VOI (oracle, 20 seeds) — greedy=189.4, exact-ceiling=211.8\n")
for D in 1:5
    r = simulate(D)
    flag = r.score > 189.4 ? "  > greedy ✓" : "  ≤ greedy"
    println("  D=$D  score=$(round(r.score,digits=1))  acc=$(round(100r.acc,digits=1))%  probe-Q/seed=$(round(r.probes,digits=1))  cap=$(round(100*(r.score-180.8)/(211.8-180.8),digits=0))% of [180.8→211.8]$flag")
end
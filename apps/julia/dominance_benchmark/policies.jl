# Role: brain-side application
"""
    policies.jl — The five meta-selection policies of the dominance benchmark

Each factory returns a closure `(scored::Dict{Symbol,Float64}, step::Int) → Symbol` plugged
into `run_agent(meta_policy=…)` (the Phase-3 seam). `eu_max` is the agent; the other four are
research baselines that deliberately implement non-Bayesian selection for empirical contrast
(dominance-design §2). Scope of Invariant 1 is the agent, not its baselines.

Baselines may act on non-positive scores — that waste is what the benchmark measures.
"""

const GROWTH_OPS = Symbol[:gw_explore, :gw_add_feature, :gw_perturb_grammar]
const ESCAPE_OPS = Symbol[:gw_enumerate_more, :gw_deepen]
# The random baseline's op set excludes :gw_deepen: each hit raises the GLOBAL enumeration
# depth, so a score-blind policy that keeps drawing it compounds exponentially and its
# wall-clock is unbounded by construction (measured: minutes vs seconds per cell). No other
# policy ever reaches deepen either — the EU-max tie order is breadth-before-depth — so the
# exclusion changes no comparison the gate makes.
const RANDOM_OPS = Symbol[:gw_enumerate_more, :gw_perturb_grammar, :gw_explore, :gw_add_feature]

"""The agent: the deterministic EU-max argmax with the act-now floor (no pragma — this IS the policy under test)."""
make_eu_max() = (scored, step) -> default_eu_max_policy(scored)

"""
Retired random explorer, act-rate family: at each opportunity, with probability `p` take a
uniform non-idle op; otherwise do nothing. Score-blind by declaration (the seam skips the
exact lookaheads it would never read). Swept over `p` by the harness — best-tuned reported
(anti-strawman, symmetric with the fixed schedule's k sweep). The always-act literal (p=1)
is subsumed as the sweep's endpoint but is wall-clock-pathological: score-blind growth
executions bloat the enumeration without bound.
"""
function make_random(seed::Int, p::Float64)
    rng = MersenneTwister(seed + 1_000_000)
    # credence-lint: allow — precedent:baseline-comparison — random: score-blind rate-p uniform op choice, the retired random explorer
    ScoreBlind((scored, step) -> rand(rng) < p ?
        RANDOM_OPS[rand(rng, 1:length(RANDOM_OPS))] : :gw_do_nothing)
end

"""
Fixed schedule: on every k-th step, take exactly one growth-ladder op (cycling
explore → add_feature → perturb → enumerate), regardless of VOI; otherwise do nothing.
Swept over k by the harness; the best-tuned configuration is what the gate compares
against (anti-strawman, dominance-design §6).
"""
function make_fixed_schedule(k::Int)
    cycle = Symbol[:gw_explore, :gw_add_feature, :gw_perturb_grammar, :gw_enumerate_more]
    idx = Ref(0)
    last_acted_step = Ref(-1)
    # credence-lint: allow — precedent:baseline-comparison — fixed_schedule: VOI-blind periodic exploration, the hand-tuned schedule
    ScoreBlind((scored, step) -> begin
        (step % k == 0 && step != last_acted_step[]) || return :gw_do_nothing
        last_acted_step[] = step
        idx[] += 1
        cycle[mod1(idx[], length(cycle))]
    end)
end

"""
Scope-A floor: EU-max with grammar/feature growth masked off, keeping the SAME
learned-returns escape ops as eu_max. The headline gap eu_max − never_explore
isolates exploration's value with the heuristic held constant on both sides —
the one comparison the softest score in the policy cannot contaminate (§6).
"""
function make_never_explore()
    # credence-lint: allow — precedent:baseline-comparison — never_explore: growth ops vetoed, the Scope-A floor de-confounder
    (scored, step) -> begin
        masked = copy(scored)
        for op in GROWTH_OPS
            masked[op] = -Inf
        end
        default_eu_max_policy(masked)
    end
end

"""
Clairvoyant ceiling: eu_max PLUS the task's ground truth (the regime schedule). It behaves
exactly like eu_max except inside an adaptation window after each regime start, where it
eagerly forces the growth ladder — one :gw_explore then one :gw_add_feature attempt per
window — without waiting for the residual buffer to justify them (the ops no-op harmlessly
when nothing clears). Strictly more informed than eu_max, never less capable: a genuine
soft ceiling on adaptation timing. The window length is ground-truth task data.
"""
function make_clairvoyant(regime_change_steps::Vector{Int}; window::Int=15)
    starts = [1; regime_change_steps]
    forced = Dict{Int, Int}()   # regime start → forced-op count this window
    # credence-lint: allow — precedent:baseline-comparison — clairvoyant: regime-schedule ground truth, the adaptation-timing ceiling
    (scored, step) -> begin
        rs = findlast(s -> s <= step < s + window, starts)
        if rs !== nothing
            start = starts[rs]
            n = get(forced, start, 0)
            if n < 2
                forced[start] = n + 1
                return n == 0 ? :gw_explore : :gw_add_feature
            end
        end
        default_eu_max_policy(scored)
    end
end

"""
Wrap a policy with a growth-op recorder: appends (step, op) to `log` whenever the wrapped
policy takes a grammar/feature-growth action. Used for the behaviour-verified inversions
report (concrete steps where one policy grows and another does not).
"""
function make_recording(policy::Function, log::Vector{Tuple{Int, Symbol}})
    rec = (scored, step) -> begin
        chosen = policy(scored, step)::Symbol
        chosen in GROWTH_OPS && push!(log, (step, chosen))
        chosen
    end
    # Preserve the score-blindness declaration through the wrapper.
    score_blind(policy) ? ScoreBlind(rec) : rec
end

# test_growth_returns.jl — belief-derived meta-action valuation, engine half
# (docs/exploration-budget/belief-derived-valuation-design.md, RATIFIED 2026-07-02).
#
# Sections:
#   §1  growth_value — the horizon-completed growth valuation: the H == n_buf nested-
#       special-case pin (window-total score reproduced EXACTLY), horizon proportionality,
#       the one-time prior term (outside plateau AND horizon), the empty-window guard.
#   §2  Reciprocal — the Gamma closed form E[1/λ] = β/(α−1), and the α ≤ 1 fail-loud guard.
#   §3  Returns cells — conjugate exactness through the one learning mechanism (condition
#       with the declared Exponential kernel): fresh-context optimism, the zero-yield decay
#       sequence 1 → 1/2 → 1/3, a positive-yield update, exact α/β, and the context split
#       (changed vs unchanged cells hold their own evidence; note_space_change! flips the
#       bit, never resets a cell).
#   §4  growth_yield — the yield observable: posterior mass of the injected tags in nats,
#       exactly 0.0 on a dedup no-op, exact against the coherent-injection transition.
#
# The selection-seam integration (escape ops fire on fresh optimism, self-extinguish under
# zero-yield evidence, rejoin the one argmax) is asserted by test_grid_world_meta.jl.
#
# Run: julia test/test_growth_returns.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: growth_value, GrowthReturns, initial_growth_returns, expected_growth_yield,
                observe_growth_yield!, note_space_change!, growth_yield,
                Reciprocal, GammaPrevision, expect, params,
                MixturePrevision, TaggedBetaPrevision, BetaPrevision, Prevision, weights,
                probability, TagSet, Interval,
                Grammar, ProductionRule, Program, AgentState, compile_kernel,
                ExploreObservation, add_programs_to_state!, reset_grammar_counter!

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("growth returns — belief-derived meta-action valuation (engine half)")
println("="^64)

# ── §1  growth_value: the canalised horizon completion ──
let
    # The nested-special-case pin: H == n_buf reproduces the window-total score EXACTLY
    # (H/n_buf == 1.0 in floats — any n_buf, not just powers of two).
    for n in (7, 12, 30, 100)
        Δℓ, pl, prior, cost = 1.7342, 0.8317, -log(2.0), 0.11
        check("§1 nested pin: H == n_buf (n=$n) reproduces plateau·Δℓ + prior − cost exactly",
              growth_value(Δℓ, n, pl, Float64(n); prior_term = prior, compute_cost = cost) ==
              pl * Δℓ + prior - cost)
    end
    # Horizon proportionality: doubling H doubles the fit term exactly (2n/n == 2.0).
    check("§1 horizon proportionality: H = 2·n_buf doubles the fit term exactly",
          growth_value(0.9, 12, 0.5, 24.0) == 2.0 * (0.5 * 0.9))
    # The prior term is ONE-TIME: outside plateau and outside the horizon.
    check("§1 prior term rides outside plateau AND horizon (zero fit ⇒ prior − cost alone)",
          growth_value(0.0, 10, 0.3, 1000.0; prior_term = -log(2.0)) == -log(2.0))
    check("§1 prior term not horizon-multiplied (huge H, placement pin)",
          growth_value(1.0, 10, 1.0, 1000.0; prior_term = -log(2.0)) == 100.0 - log(2.0))
    # Empty window: no measured gain — prior − cost, no NaN from 0/0.
    check("§1 n_buf == 0 guard: prior_term − compute_cost exactly",
          growth_value(0.0, 0, 1.0, 5.0; prior_term = -0.2, compute_cost = 0.3) == -0.5)
end

# ── §2  Reciprocal: the Gamma closed form ──
let
    check("§2 E_Gamma(2,1)[1/λ] == 1.0 (the fresh-cell prior optimism)",
          expect(GammaPrevision(2.0, 1.0), Reciprocal()) == 1.0)
    check("§2 E_Gamma(3,1)[1/λ] == 0.5", expect(GammaPrevision(3.0, 1.0), Reciprocal()) == 0.5)
    check("§2 E_Gamma(4,2.6)[1/λ] == 2.6/3 exactly",
          expect(GammaPrevision(4.0, 2.6), Reciprocal()) == 2.6 / 3.0)
    threw = false
    try
        expect(GammaPrevision(1.0, 1.0), Reciprocal())
    catch
        threw = true
    end
    check("§2 α ≤ 1 fails loud (the expectation diverges)", threw)
end

# ── §3  Returns cells: conjugate exactness, one learning mechanism, context split ──
let
    gr = initial_growth_returns()
    check("§3 never-fired op reads the fresh changed-context cell: E[yield] == 1.0",
          expected_growth_yield(gr, :enumerate) == 1.0)

    # First firing conditions the (op, changed) cell; with no space change the context flips
    # to (op, unchanged) — its OWN fresh cell (a different hypothesis, not a reset).
    observe_growth_yield!(gr, :enumerate, 0.0)
    check("§3 after first firing (no space change): the unchanged-context cell is fresh (1.0)",
          expected_growth_yield(gr, :enumerate) == 1.0)

    # The zero-yield decay sequence IN one cell: 1 → 1/2 → 1/3 (β/(α−1) under α += 1, β += 0).
    observe_growth_yield!(gr, :enumerate, 0.0)
    check("§3 one zero yield in the unchanged cell: expectation halves (== 0.5)",
          expected_growth_yield(gr, :enumerate) == 0.5)
    observe_growth_yield!(gr, :enumerate, 0.0)
    check("§3 two zero yields: == 1/3", expected_growth_yield(gr, :enumerate) == 1.0 / 3.0)

    # Exact α/β through the one learning mechanism (three observations in the unchanged cell:
    # 0.0, 0.0, then 0.4).
    observe_growth_yield!(gr, :enumerate, 0.4)
    cell = gr.cells[(:enumerate, false)]
    pr = params(cell)   # credence-lint: allow — precedent:test-oracle — exact conjugate α/β against the manual Gamma×Exponential update
    check("§3 exact conjugate state: α == 2+3, β == 1+0.4",
          pr.alpha == 5.0 && pr.beta == 1.4, "params=$pr")
    check("§3 expected yield == β/(α−1) == 1.4/4 exactly",
          expected_growth_yield(gr, :enumerate) == 1.4 / 4.0)

    # Context split: a space change flips the bit; the changed cell holds ITS evidence
    # (the one first-firing observation), untouched by the unchanged cell's four.
    note_space_change!(gr)
    check("§3 after note_space_change!: the changed-context cell retains its own 1 obs (== 0.5)",
          expected_growth_yield(gr, :enumerate) == 0.5)
    check("§3 cells never reset: the unchanged cell still holds its posterior",
          expect(gr.cells[(:enumerate, false)], Reciprocal()) == 1.4 / 4.0)

    # Independent op: its own cells, untouched.
    check("§3 a different op reads its own fresh cell (== 1.0)",
          expected_growth_yield(gr, :deepen) == 1.0)
end

# ── §4  growth_yield: the yield observable ──
let
    # Hand-built mixture: uniform over 4 tagged components.
    m = MixturePrevision(Prevision[TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0)) for i in 1:4],
                         zeros(4))
    check("§4 mass of tags {3,4} on the uniform 4-mixture == 0.5 (probability, Tier-1 read)",
          probability(m, TagSet(Interval(0.0, 1.0), Set([3, 4]))) == 0.5)
    check("§4 yield == −log(1 − mass) exactly",
          growth_yield(m, [3, 4]) == -log1p(-0.5))
    check("§4 empty tag set (dedup no-op) yields exactly 0.0",
          growth_yield(m, Int[]) == 0.0)

    # Against the real transition: inject via add_programs_to_state! (coherent injection,
    # empty window at t=0) and read the injected block's mass.
    reset_grammar_counter!()
    g1 = Grammar(Set([:red]), ProductionRule[], 1)
    prog = Program(Credence.IfExpr(Credence.GTExpr(Credence.FeatureRef(:red), 0.5),
                                   Credence.ActionExpr(:food), Credence.ActionExpr(:enemy)), 3, 1)
    state = AgentState(MixturePrevision(Prevision[TaggedBetaPrevision(1, BetaPrevision(1.0, 1.0))],
                                        [0.0]),
                       [(1, 1)], [compile_kernel(prog, g1, 1)], Program[prog],
                       Dict(1 => g1), 2)
    g2 = Grammar(Set([:red, :speed]), ProductionRule[], 2)
    len_before = length(state.compiled_kernels)
    n_added = add_programs_to_state!(state, g2, 2; observations = ExploreObservation[],
                                     action_space = Symbol[:food, :enemy])
    check("§4 precondition: the enumeration injected components", n_added > 0, "n_added=$n_added")
    tags = (len_before + 1):length(state.compiled_kernels)
    w = weights(state.belief)
    mass = sum(w[t] for t in tags)   # credence-lint: allow — precedent:test-oracle — manual mass sum against probability(·, TagSet)
    check("§4 yield of the injected block == −log(1 − Σ injected weights) exactly",
          growth_yield(state.belief, tags) == -log1p(-mass))

    # Dedup no-op: the same grammar again adds nothing and yields exactly 0.0.
    len_before2 = length(state.compiled_kernels)
    n2 = add_programs_to_state!(state, g2, 2; observations = ExploreObservation[],
                                action_space = Symbol[:food, :enemy])
    check("§4 dedup no-op: n_added == 0 and yield == 0.0",
          n2 == 0 && growth_yield(state.belief, (len_before2 + 1):length(state.compiled_kernels)) == 0.0)
end

println("="^64)
println("ALL CHECKS PASSED — growth returns (engine half)")
println("="^64)

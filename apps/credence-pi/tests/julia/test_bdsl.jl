#!/usr/bin/env julia
"""
    test_bdsl.jl — Step 2 of credence-pi: Pass-1 BDSL programs.

Loads the five files under apps/credence-pi/bdsl/ in dependency order,
then exercises the public surface decide-action / observe-response /
followup-after-response. The architectural claim of Pass 1 is that at
Beta(2,2) under symmetric proceed/block utilities, voi(ask) > 0 while
EU(proceed) = EU(block) = 0, so ask wins by EU-maximisation rather
than by being forced. The numerical values are derived analytically
below.

Tolerance semantics (see src/ontology.jl `expect(::BetaMeasure,
::Function)` for the underlying primitive): the post-step-2 Gauss-
Jacobi quadrature is *polynomial-exact* in the mathematical sense
(captures pass1-pref's degree-1 integrand exactly), and *FP-precision-
bounded* in the computational sense — accumulated rounding in the
weighted sum leaves residual error well under 1e-13. The 1e-12 atol
below accommodates that FP rounding, not quadrature error. A failure
here is a design surprise, not a test bug.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence
using Credence.Eval: EffectorDecl

const BDSL_DIR = joinpath(@__DIR__, "..", "..", "bdsl")

function load_pass1_env()
    env = Credence.Eval.default_env()
    env[:__toplevel__] = true
    stdlib_path = joinpath(@__DIR__, "..", "..", "..", "..", "src", "stdlib.bdsl")
    for expr in Credence.Parse.parse_all(read(stdlib_path, String))
        Credence.Eval.eval_dsl(expr, env)
    end
    for fname in ("capabilities.bdsl", "features.bdsl",
                  "prior.bdsl", "kernel.bdsl", "decide.bdsl")
        for expr in Credence.Parse.parse_all(read(joinpath(BDSL_DIR, fname), String))
            Credence.Eval.eval_dsl(expr, env)
        end
    end
    env
end

const PASSED = String[]
function ok(name)
    push!(PASSED, name)
    println("PASSED: ", name)
end

const ENV_ = load_pass1_env()
ok("all five Pass-1 BDSL files load in dependency order")

# ── 1. Manifest yields three-action space ───────────────────────────────

let manifest = ENV_[:manifest]
    @assert length(manifest) == 3
    @assert all(d -> d isa EffectorDecl, manifest)
    @assert [d.name for d in manifest] == [:ask, :proceed, :block]
    ok("manifest holds three EffectorDecls in declaration order")
end

let action_space = ENV_[Symbol("action-space")]
    @assert action_space isa Finite
    @assert support(action_space) == [:ask, :proceed, :block]
    ok("(apply space :finite (effector-names manifest)) yields the three-action Finite space")
end

# ── 2. Cold-start numerical claims ──────────────────────────────────────
#
# Analytical derivation (see SPEC.md "Pass 1 BDSL" / decide.bdsl).
#
# Prior: Beta(α, β) on θ ∈ [0,1]; E[θ] = α/(α+β).
# pass1-pref:
#   pref(θ, proceed) = 2θ - 1
#   pref(θ, block)   = 1 - 2θ
#   pref(θ, ask)     = 0
#
# Under expect, EU under pass1-pref:
#   EU(proceed) = 2*E[θ] - 1 = (α-β)/(α+β)
#   EU(block)   = 1 - 2*E[θ] = (β-α)/(α+β)
#   EU(ask)     = 0
# So value(m) = max(|α-β|/(α+β), 0).
#
# Bernoulli observation kernel: P(o=1) = E[θ] = α/(α+β).
# Beta-Bernoulli conjugate: o=1 → Beta(α+1, β); o=0 → Beta(α, β+1).
#
# At Beta(2, 2): E[θ] = 1/2, value(prior) = 0.
#   o=1 → Beta(3,2): value = 1/5
#   o=0 → Beta(2,3): value = 1/5
#   E_o[value(post)] = (1/2)(1/5) + (1/2)(1/5) = 1/5
#   voi(ask) = 1/5 - 0 = 1/5 = 0.2
#
# General: at Beta(α, α), voi(ask) = 1/(2α + 1).
#   α=2 → 1/5;  α=3 → 1/7;  α=4 → 1/9.
# voi decreases as the symmetric posterior concentrates.

const TOL = 1e-12   # credence-lint: allow — precedent:test-oracle — analytical EVPI vs DSL voi computed by stdlib.bdsl

let prior   = ENV_[Symbol("make-prior")](),
    pref    = ENV_[Symbol("pass1-pref")],
    actions = ENV_[Symbol("action-space")],
    kernel  = ENV_[Symbol("approve-kernel")]

    eu_proceed = expect(prior, theta -> pref(theta, :proceed))
    eu_block   = expect(prior, theta -> pref(theta, :block))
    eu_ask     = expect(prior, theta -> pref(theta, :ask))

    @assert isapprox(eu_proceed, 0.0; atol=TOL)  # credence-lint: allow — precedent:test-oracle — analytical EU(proceed)=0 at Beta(2,2)
    @assert isapprox(eu_block,   0.0; atol=TOL)  # credence-lint: allow — precedent:test-oracle — analytical EU(block)=0 at Beta(2,2)
    @assert eu_ask == 0.0                         # ask is constant 0 — no quadrature involved
    ok("EU(proceed) = EU(block) = 0 at Beta(2,2) under pass1-pref (polynomial-exact, FP-bounded ≤ 1e-12)")

    # voi(ask) computed by stdlib.bdsl's voi composition.
    voi_fn = ENV_[:voi]
    voi_ask = voi_fn(prior, kernel, actions, pref, [0, 1])
    @assert isapprox(voi_ask, 0.2; atol=TOL)  # credence-lint: allow — precedent:test-oracle — analytical EVPI = 1/5 at Beta(2,2)
    ok("voi(ask) = 1/5 at Beta(2,2) (matches analytical EVPI within 1e-12)")

    # decide-action returns :ask (corollary of the two assertions above).
    decide = ENV_[Symbol("decide-action")]
    @assert decide(prior) === :ask
    ok("decide-action returns :ask at cold start")
end

# ── 3. Observations update the posterior ───────────────────────────────

let prior  = ENV_[Symbol("make-prior")](),
    obs_fn = ENV_[Symbol("observe-response")]

    p1 = obs_fn(prior, 1)
    @assert p1 isa BetaMeasure
    @assert p1.alpha == 3.0 && p1.beta == 2.0
    ok("observe-response 1 turns Beta(2,2) into Beta(3,2) (exact)")

    p0 = obs_fn(prior, 0)
    @assert p0.alpha == 2.0 && p0.beta == 3.0
    ok("observe-response 0 turns Beta(2,2) into Beta(2,3) (exact)")

    p_yes_no = obs_fn(obs_fn(prior, 1), 0)
    @assert p_yes_no.alpha == 3.0 && p_yes_no.beta == 3.0
    ok("observe-response 1 then 0 yields Beta(3,3)")
end

# ── 4. voi decreases as symmetric posterior concentrates ───────────────

let voi_fn  = ENV_[:voi],
    actions = ENV_[Symbol("action-space")],
    pref    = ENV_[Symbol("pass1-pref")],
    kernel  = ENV_[Symbol("approve-kernel")]

    voi_at(α, β) = voi_fn(BetaMeasure(α, β), kernel, actions, pref, [0, 1])

    v22 = voi_at(2.0, 2.0)
    v33 = voi_at(3.0, 3.0)
    v44 = voi_at(4.0, 4.0)

    @assert isapprox(v22, 1/5; atol=TOL)  # credence-lint: allow — precedent:test-oracle — voi(Beta(2,2)) = 1/(2α+1)
    @assert isapprox(v33, 1/7; atol=TOL)  # credence-lint: allow — precedent:test-oracle — voi(Beta(3,3)) = 1/(2α+1)
    @assert isapprox(v44, 1/9; atol=TOL)  # credence-lint: allow — precedent:test-oracle — voi(Beta(4,4)) = 1/(2α+1)
    @assert v22 > v33 > v44 > 0
    ok("voi decreases as symmetric posterior concentrates: 1/5 > 1/7 > 1/9")
end

# ── 5. After enough one-sided evidence, proceed wins ───────────────────
#
# At Beta(4, 2) (two yeses on top of Beta(2,2)):
#   EU(proceed) = (4-2)/6 = 1/3
#   EU(block)   = -1/3
#   value(prior) = 1/3
#   o=1 → Beta(5,2): value = 3/7
#   o=0 → Beta(4,3): value = 1/7
#   P(o=1) = 4/6 = 2/3, P(o=0) = 1/3
#   E_o[value(post)] = (2/3)(3/7) + (1/3)(1/7) = 7/21 = 1/3
#   voi(ask) = 1/3 - 1/3 = 0
# So decide-action picks :proceed (EU 1/3 > 0).

let obs_fn = ENV_[Symbol("observe-response")],
    decide = ENV_[Symbol("decide-action")],
    prior  = ENV_[Symbol("make-prior")]()

    p_two_yes = obs_fn(obs_fn(prior, 1), 1)
    @assert p_two_yes.alpha == 4.0 && p_two_yes.beta == 2.0
    @assert decide(p_two_yes) === :proceed
    ok("after two yeses (Beta(4,2)), decide-action returns :proceed")

    p_two_no = obs_fn(obs_fn(prior, 0), 0)
    @assert p_two_no.alpha == 2.0 && p_two_no.beta == 4.0
    @assert decide(p_two_no) === :block
    ok("after two nos (Beta(2,4)), decide-action returns :block")
end

# ── 6. followup-after-response ─────────────────────────────────────────

let followup = ENV_[Symbol("followup-after-response")]
    @assert followup(Dict("response" => "yes")) === :proceed
    @assert followup(Dict("response" => "no"))  === :block
    @assert followup(Dict(:response => "yes")) === :proceed   # symbol-keyed too
    @assert followup(Dict("response" => "timeout")) === :nothing
    ok("followup-after-response: yes→proceed, no→block, timeout→:nothing")
end

println()
println("=" ^ 60)
println("ALL ", length(PASSED), " ASSERTIONS PASSED")
println("=" ^ 60)

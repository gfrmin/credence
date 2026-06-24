# test_routing.jl — the engine routing stdlib (decouple Move 4, src/routing.jl). Self-contained
# (no apps/ dependency): builds a tiny inline routing fixture and pins the lifted ops directly
# through the engine's public names. The end-to-end behavioural oracle is
# apps/credence-pi/tests/julia/test_routing.jl (passes unmodified through the shim); this file
# pins the engine primitives. Asserts:
#   (1) Wald flip — route's EU-max model changes with `reward` ALONE (cost-hawk → cheap,
#       quality-hawk → best-believed), exact index;
#   (2) escalation gate boundary — escalation_next opens iff reward·θ ≥ cost, at cost = reward·θ ± 1e-6;
#   (3) w_time=0 is bit-identical to the pre-time router;
#   (4) the route_outcome! soft path reduces EXACTLY to a hard structure_observe at the
#       certain-signal corners (the substrate-equivalence guard);
#   (5) the coupled-EM route_outcome!: successes raise θ; a pure-flakiness stream leaves θ
#       ~flat and drops the learned reliability ρ̄ (confound-partialling); a seeded stream
#       identifies ρ toward the truth.
#
# Run from repo root:  julia test/test_routing.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: build_structure_model, build_structure_prior, structure_observe,
                structure_observe_soft, context_from_features,
                RoutingState, EmissionBelief, route, route_eu, escalation_next,
                posterior_accuracy, route_outcome!, decode_correctness, _ctx_key,
                MixturePrevision, mean
using Random: MersenneTwister

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("routing engine stdlib — lifted route / escalate / route_outcome! (Move 4)")
println("="^64)

# ── Fixture: one feature (prompt length), two models. A = cheap + lower accuracy, B = dear +
# higher accuracy. Seed each model's per-feature StructureBMA posterior by hand. ──
model = build_structure_model(["len"], [["short", "long"]]; alpha0 = 2.0, beta0 = 2.0, p_edge = 0.5)
X = ["short"]
feats = Dict("len" => "short")
seed(top, n1, n0) = begin
    for _ in 1:n1; top = structure_observe(model, top, X, 1); end
    for _ in 1:n0; top = structure_observe(model, top, X, 0); end
    top
end
topA = seed(build_structure_prior(model), 3, 7)    # low θ
topB = seed(build_structure_prior(model), 7, 3)    # high θ
tops = MixturePrevision[topA, topB]
costs = [0.01, 0.10]                               # A cheap, B dear
θA = posterior_accuracy(model, topA, X)
θB = posterior_accuracy(model, topB, X)
check("fixture: θ_B > θ_A (B better-believed)", θB > θA, "$θA vs $θB")

# ── (1) Wald flip: cheap at low reward, best-believed at high reward ──
check("route picks the CHEAP model at low reward (cost-hawk)", route(model, tops, X, costs, 0.01) == 1)
check("route picks the BEST-BELIEVED model at high reward (quality-hawk)", route(model, tops, X, costs, 100.0) == 2)

# ── (2) escalation gate boundary: opens iff reward·θ ≥ cost ──
check("escalation opens when cost just below reward·θ (tier 1)",
      escalation_next(model, tops, X, [1.0 * θA - 1e-6, 9.9], 1.0, Set{Int}()) == 1,
      "θA=$θA")
check("escalation STOPs when every rung's cost just above its reward·θ",
      escalation_next(model, tops, X, [1.0 * θA + 1e-6, 9.9], 1.0, Set{Int}()) == 0)

# ── (3) w_time=0 bit-identical ──
base = route(model, tops, X, costs, 1.0)
check("w_time=0 with times == the time-blind router (bit-identical)",
      route(model, tops, X, costs, 1.0; w_time = 0.0, times = [0.5, 0.9]) == base)
check("w_time=0 with times=nothing == the time-blind router (bit-identical)",
      route(model, tops, X, costs, 1.0; w_time = 0.0, times = nothing) == base)

# ── (4) route_outcome! soft path ≡ hard structure_observe at certain-signal corners ──
let top0 = build_structure_prior(model)
    h1 = posterior_accuracy(model, structure_observe(model, top0, X, 1), X)
    s1 = posterior_accuracy(model, structure_observe_soft(model, top0, X, 1.0, 0.0), X)
    h0 = posterior_accuracy(model, structure_observe(model, top0, X, 0), X)
    s0 = posterior_accuracy(model, structure_observe_soft(model, top0, X, 0.0, 1.0), X)
    check("soft (r,w)=(1,0) ≡ hard obs=1 (exact)", s1 == h1, "$s1 vs $h1")
    check("soft (r,w)=(0,1) ≡ hard obs=0 (exact)", s0 == h0, "$s0 vs $h0")
end

# ── (5) the coupled-EM route_outcome! ──
mkstate() = RoutingState(model, MixturePrevision[seed(build_structure_prior(model), 20, 30),
                                                 seed(build_structure_prior(model), 20, 30)],
                         Dict{String, MixturePrevision}(), EmissionBelief(),
                         ["a", "b"], ["p", "p"], ["id-a", "id-b"], [0.01, 0.10], 0.5, 0.0, nothing)
ρ̄of(rt) = mean(get(rt.emission.rho_cells, _ctx_key(X), rt.emission.rho0))

# successes raise the routed model's θ
let rt = mkstate()
    before = posterior_accuracy(rt.model, rt.tops[1], X)
    for _ in 1:30; route_outcome!(rt, "id-a", feats, true); end
    check("clean successes raise the routed model's θ", posterior_accuracy(rt.model, rt.tops[1], X) > before,
          "$before → $(posterior_accuracy(rt.model, rt.tops[1], X))")
end

# confound-partialling: a pure-flakiness stream (every call fails) leaves θ ~flat and is
# absorbed by the learned reliability ρ̄ (correct calls failed too ⇒ the tool is flaky).
let rt = mkstate()
    before = [posterior_accuracy(rt.model, rt.tops[a], X) for a in 1:2]
    for _ in 1:60, a in 1:2; route_outcome!(rt, ["id-a", "id-b"][a], feats, false); end
    drift = maximum(abs.([posterior_accuracy(rt.model, rt.tops[a], X) for a in 1:2] .- before))
    ρ̄_drop = 0.667 - ρ̄of(rt)
    # The confound-partialling signature: the flakiness lands in ρ̄ (which falls sharply),
    # NOT in θ (which barely moves) — ρ̄'s drop dwarfs θ's drift.
    check("flakiness drops the learned reliability ρ̄ sharply (≫ θ drift)",
          ρ̄_drop > 0.1 && ρ̄_drop > 3 * drift, "ρ̄_drop=$ρ̄_drop  drift=$drift")
end

# seeded identifiability: a true emission process (ρ_true high, σ_true low) is recovered —
# ρ̄ rises from the 2/3 prior toward ρ_true while θ stays anchored.
let rt = mkstate(), rng = MersenneTwister(20260624)
    ρ_true, σ_true, θ_true = 0.9, 0.25, 0.5
    θ0 = [posterior_accuracy(rt.model, rt.tops[a], X) for a in 1:2]
    for _ in 1:1000
        a = rand(rng, 1:2)
        C = rand(rng) < θ_true
        e = rand(rng) < (C ? ρ_true : σ_true)
        route_outcome!(rt, ["id-a", "id-b"][a], feats, e)
    end
    Δθ = maximum(abs.([posterior_accuracy(rt.model, rt.tops[a], X) for a in 1:2] .- θ0))
    check("seeded EM lifts ρ̄ from prior toward ρ_true (0.667 → >0.74)", ρ̄of(rt) > 0.74, "ρ̄=$(ρ̄of(rt))")
    check("seeded EM keeps θ anchored (Δθ < 0.1)", Δθ < 0.1, "Δθ=$Δθ")
end

println("="^64)
println("ALL CHECKS PASSED — routing engine stdlib exact")
println("="^64)

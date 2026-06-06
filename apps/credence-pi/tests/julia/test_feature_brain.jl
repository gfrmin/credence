# Role: tests
# test_feature_brain.jl — the Route-B feature-conditioned brain (Pass 2, Move 3).
#
# Asserts the typed Julia brain reproduces the G1 spike oracle EXACTLY through its
# public path (build_prior → observe → belief_at_context / decide), that it builds
# the real 5-feature model, that cold-start asks by computation, that the surgical
# win fires (block the loop, spare the novel call), and the degenerate reductions.
#
# Oracle: the G1 spike (exact-fraction reference) — structure marginal likelihoods
# S0=1/105, S1=3/175, S2=1/25, S3=3/100; predictives 131/406 and 136/203.
#
# Run from repo root:
#     julia --project=. apps/credence-pi/tests/julia/test_feature_brain.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: Identity, weights, mean, expect
include(joinpath(@__DIR__, "..", "..", "brain", "feature_brain.jl"))
using .FeatureBrain

function check(name, cond, detail="")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("assertion failed: $name")
    end
end

println("="^64)
println("feature brain (Route B) — Move 3")
println("="^64)

# ── 1. Spike-oracle reproduction: 2 features (tool, rep), 4 structures. ──
# build_model enumerates structures by bit-mask: ∅, {tool}, {rep}, {tool,rep} —
# matching the spike's S0..S3.
let model = build_model(["tool", "rep"], [["bash", "read"], ["rep0", "rep3"]];
                        alpha0 = 2.0, beta0 = 2.0, p_edge = 0.5)
    check("2-feature model has 4 structures", length(model.structures) == 4,
          "got $(length(model.structures))")
    check("globally-unique tags: 1 + 2 + 2 + 4 = 9 cells",
          sum(length(d) for d in model.cell_tag) == 9,
          "got $(sum(length(d) for d in model.cell_tag))")

    top = build_prior(model)
    check("prior structure weights uniform (p=0.5)",
          all(isapprox(w, 0.25; atol=1e-12) for w in weights(top)), "got $(weights(top))")
    check("prior predictive P(approve|bash,rep3) = 0.5",
          isapprox(expect(belief_at_context(model, top, ["bash", "rep3"]), Identity()), 0.5; atol=1e-12))

    OBS = [(["read", "rep0"], 1), (["bash", "rep0"], 1), (["bash", "rep3"], 0),
           (["bash", "rep3"], 0), (["bash", "rep3"], 0), (["read", "rep0"], 1)]
    for (X, o) in OBS
        top = observe(model, top, X, o)
    end

    w = weights(top)
    check("structure posterior S0 = 20/203", isapprox(w[1], 0.09852216748768473; atol=1e-9), "got $(w[1])")
    check("structure posterior S1 = 36/203", isapprox(w[2], 0.17733990147783252; atol=1e-9), "got $(w[2])")
    check("structure posterior S2 = 84/203", isapprox(w[3], 0.41379310344827586; atol=1e-9), "got $(w[3])")
    check("structure posterior S3 = 63/203", isapprox(w[4], 0.31034482758620690; atol=1e-9), "got $(w[4])")
    # credence-lint: allow — precedent:test-oracle — asserts the rep-parent structure posterior dominates
    check("rep-parent structures S2+S3 dominate (>0.7)", w[3] + w[4] > 0.7, "got $(w[3]+w[4])")

    pbr = expect(belief_at_context(model, top, ["bash", "rep3"]), Identity())
    prr = expect(belief_at_context(model, top, ["read", "rep0"]), Identity())
    check("P(approve|bash,rep3) = 131/406 ≈ 0.3227 (deny the loop)",
          isapprox(pbr, 0.32266009852216746; atol=1e-9), "got $pbr")
    check("P(approve|read,rep0) = 136/203 ≈ 0.6700 (approve the novel call)",
          isapprox(prr, 0.66995073891625620; atol=1e-9), "got $prr")
    check("surgical: predictive(bash,rep3) < 0.5 < predictive(read,rep0)",
          # credence-lint: allow — precedent:test-oracle — surgical-win comparison vs the 0.5 threshold
          pbr < 0.5 < prr, "bash,rep3=$pbr read,rep0=$prr")

    # ── 2. The decision: surgical win at λ=1 (θ* = 0.5), c=1. ──
    c, λ, q = 1.0, 1.0, 0.05
    d_loop  = decide(model, top, ["bash", "rep3"], c; aversion = λ, interrupt_cost = q)
    d_novel = decide(model, top, ["read", "rep0"], c; aversion = λ, interrupt_cost = q)
    check("decide blocks the repeated loop (bash,rep3)", d_loop === :block, "got $d_loop")
    check("decide does NOT block the novel call (read,rep0)", d_novel !== :block, "got $d_novel")

    # ── 3. Decision-margin instrumentation (expert ask). ──
    # θ* = 1/(1+λ). Replay the loop one obs at a time and report when the decision
    # flips proceed/ask → block at (bash,rep3), with the head-room below θ*.
    θ_star = 1.0 / (1.0 + λ)
    println("--- decision-margin trajectory at (bash,rep3), λ=$λ ⇒ θ*=$θ_star ---")
    t2 = build_prior(model)
    flip_at = 0
    for (i, (X, o)) in enumerate(OBS)
        t2 = observe(model, t2, X, o)
        θ = expect(belief_at_context(model, t2, ["bash", "rep3"]), Identity())
        # credence-lint: allow — precedent:test-oracle — rep-parent mass for the margin-trajectory readout
        repmass = weights(t2)[3] + weights(t2)[4]
        dec = decide(model, t2, ["bash", "rep3"], c; aversion = λ, interrupt_cost = q)
        dec === :block && flip_at == 0 && (flip_at = i)
        println("  after obs $i ($(OBS[i][1]) →$(OBS[i][2])): θ=$(round(θ,digits=4)) " *
                "rep-mass=$(round(repmass,digits=4)) decision=$dec")
    end
    check("block fires within the 6-obs loop (margin reachable)", 1 <= flip_at <= 6,
          "flip_at=$flip_at")
    final_θ = expect(belief_at_context(model, top, ["bash", "rep3"]), Identity())
    println("  head-room below threshold: θ*−θ_final = $(round(θ_star - final_θ, digits=4)) " *
            "(rep-mass 0.72)")
    check("final predictive sits below θ* with head-room", final_θ < θ_star, "θ=$final_θ θ*=$θ_star")
end

# ── 4. Cold-start asks by computation (calibration-friendly). ──
# At the default λ=1.0 (θ*=0.5), cold-start θ=0.5 gives EU(proceed)=EU(block)=0,
# while one yes/no observation has positive value (a 'no' would flip to block):
# voi=0.1 ⇒ EU(ask)=voi−q=0.05 > 0, so ask wins BY COMPUTATION (not a constant).
let model = build_model(["tool", "rep"], [["bash", "read"], ["rep0", "rep3"]]; p_edge = 0.5)
    top = build_prior(model)
    d = decide(model, top, ["bash", "rep3"], 1.0; aversion = 1.0, interrupt_cost = 0.05)
    check("cold-start asks by computation (voi-gated)", d === :ask, "got $d")
end

# ── 5. Degenerate reduction: point mass on ∅ ≡ a single global Beta. ──
let model = build_model(["tool", "rep"], [["bash", "read"], ["rep0", "rep3"]]; p_edge = 0.5)
    # Force point mass on the ∅ structure (index 1) by a near-degenerate prior.
    # Easiest: build the prior, then assert the ∅ structure alone reproduces global Beta.
    top = build_prior(model)
    OBS = [(["read", "rep0"], 1), (["bash", "rep0"], 1), (["bash", "rep3"], 0),
           (["bash", "rep3"], 0), (["bash", "rep3"], 0), (["read", "rep0"], 1)]
    for (X, o) in OBS
        top = observe(model, top, X, o)
    end
    # The ∅ structure's single cell saw all six outcomes (1,1,0,0,0,1) → Beta(5,5) → mean 0.5,
    # context-independent.
    s0 = top.components[1]                 # ProductPrevision with one cell
    cell = s0.factors[1]
    check("∅-structure cell ≡ global Beta(5,5): mean 0.5",
          isapprox(mean(cell.beta), 0.5; atol=1e-12), "got $(mean(cell.beta))")
    # Its predictive is context-independent (same at every X).
    bx_a = expect(MixturePrevision([cell], [0.0]), Identity())
    check("∅-structure predictive context-independent = 0.5", isapprox(bx_a, 0.5; atol=1e-12))
end

# ── 6. The real 5-feature model builds (32 structures, 8448 all-edges cells). ──
let
    names = ["tool-name", "working-directory-relative", "parent-tool-call-name",
             "recent-repetition-count", "time-since-last-user-message"]
    vals = [["read","write","edit","bash","exec","process","apply_patch","grep","find","ls","other"],
            ["project-root","subdirectory","outside-project","no-path"],
            ["read","write","edit","bash","exec","process","apply_patch","grep","find","ls","other","none"],
            ["rep-0","rep-1","rep-2","rep-3plus"],
            ["lt-30s","lt-2m","lt-10m","gt-10m"]]
    model = build_model(names, vals; p_edge = 0.5)
    check("5-feature model has 32 structures", length(model.structures) == 32,
          "got $(length(model.structures))")
    # all-edges structure (the last, mask=11111) has 11·4·12·4·4 = 8448 cells.
    check("all-edges structure has 8448 cells",
          length(model.cell_tag[end]) == 11*4*12*4*4, "got $(length(model.cell_tag[end]))")
    top = build_prior(model)
    check("5-feature prior builds and normalises",
          # credence-lint: allow — precedent:test-oracle — structure-prior normalisation check
          isapprox(sum(weights(top)), 1.0; atol=1e-12), "got $(sum(weights(top)))")
    # cold-start predictive is 0.5 at an arbitrary context.
    X = ["bash","project-root","edit","rep-3plus","lt-30s"]
    check("5-feature cold-start predictive = 0.5",
          isapprox(expect(belief_at_context(model, top, X), Identity()), 0.5; atol=1e-12))
end

println("="^64)
println("ALL CHECKS PASSED — feature brain (Route B)")
println("="^64)

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
    # Uses the DENSE reference prior so the cell is reachable via `.factors`
    # (build_prior is sparse; equivalence is asserted in test_sparse_structure_equivalence.jl).
    top = build_prior_dense(model)
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

# ── 6. The real 6-feature model builds (64 structures, 33792 all-edges cells). ──
let
    names = ["tool-name", "working-directory-relative", "parent-tool-call-name",
             "recent-repetition-count", "recent-identical-call-count",
             "time-since-last-user-message"]
    vals = [["read","write","edit","bash","exec","process","apply_patch","grep","find","ls","other"],
            ["project-root","subdirectory","outside-project","no-path"],
            ["read","write","edit","bash","exec","process","apply_patch","grep","find","ls","other","none"],
            ["rep-0","rep-1","rep-2","rep-3plus"],
            ["ident-0","ident-1","ident-2","ident-3plus"],
            ["lt-30s","lt-2m","lt-10m","gt-10m"]]
    model = build_model(names, vals; p_edge = 0.5)
    check("6-feature model has 64 structures", length(model.structures) == 64,
          "got $(length(model.structures))")
    # all-edges structure (the last, mask=111111) has 11·4·12·4·4·4 = 33792 cells.
    check("all-edges structure has 33792 cells",
          length(model.cell_tag[end]) == 11*4*12*4*4*4, "got $(length(model.cell_tag[end]))")
    top = build_prior(model)
    check("6-feature prior builds and normalises",
          # credence-lint: allow — precedent:test-oracle — structure-prior normalisation check
          isapprox(sum(weights(top)), 1.0; atol=1e-12), "got $(sum(weights(top)))")
    # cold-start predictive is 0.5 at an arbitrary context.
    X = ["bash","project-root","edit","rep-3plus","ident-3plus","lt-30s"]
    check("6-feature cold-start predictive = 0.5",
          isapprox(expect(belief_at_context(model, top, X), Identity()), 0.5; atol=1e-12))
end

# ── 7. Multi-outcome decide: H=0 reduces to single-outcome; harm couples in. ──
# Two posteriors (waste P(approve|Xw), harm P(unsafe|Xh)); EU(block)=c[1−(1+λ)θ_a]+H·θ_u.
# Verifies the backward-compat reduction (H=0 ≡ `decide`) and the coupling no OR of
# thresholds can express (harm flips a clearly-wanted call to block). See multi_outcome.jl.
let
    wm = build_model(["ctx"], [["c"]]); hm = build_model(["ctx"], [["c"]])
    function mk(model, a, d)
        t = build_prior(model)
        for _ in 1:a; t = observe(model, t, ["c"], 1); end
        for _ in 1:d; t = observe(model, t, ["c"], 0); end
        t
    end
    X = ["c"]; c, λ, q = 0.5, 1.0, 0.02
    wt_hi = mk(wm, 16, 2)   # θ_a ≈ 0.82 (clearly wanted)
    ht_lo = mk(hm, 0, 8)    # θ_u ≈ 0.17 (safe)
    ht_hi = mk(hm, 8, 2)    # θ_u ≈ 0.71 (harmful)

    # H=0 ⇒ decide_multi is EXACTLY the single-outcome decide, for any harm belief.
    for ht in (ht_lo, ht_hi)
        dm0 = decide_multi(wm, wt_hi, hm, ht, X, X, c; aversion=λ, interrupt_cost=q, harm_cost=0.0)
        ds  = decide(wm, wt_hi, X, c; aversion=λ, interrupt_cost=q)
        check("decide_multi(H=0) ≡ single-outcome decide", dm0 === ds, "multi=$dm0 single=$ds")
    end

    # H>0: a clearly-wanted call proceeds when safe, BLOCKS when harmful — the coupling.
    d_safe = decide_multi(wm, wt_hi, hm, ht_lo, X, X, c; aversion=λ, interrupt_cost=q, harm_cost=1.0)
    d_harm = decide_multi(wm, wt_hi, hm, ht_hi, X, X, c; aversion=λ, interrupt_cost=q, harm_cost=1.0)
    check("multi-outcome: clearly-wanted + safe → proceed", d_safe === :proceed, "got $d_safe")
    check("multi-outcome: clearly-wanted + harmful → block (harm couples in)", d_harm === :block, "got $d_harm")
    check("harm flips the SAME wanted call (proceed→block)", d_safe !== d_harm)
end

# ── 8. Tail-aware EU (multi-turn look-ahead): expected_repeats m scales the waste tail. ──
# m = E[further identical repeats a block prevents] — supplied live by the tail brain as
# expect(continuation_belief, GeometricTail()). EU(block) = c·(1+m)·(1−θ) − cλθ, so block ⟺
# θ < (1+m)/((1+m)+λ); m=0 recovers the myopic 1/(1+λ). Asking inherits the (1+m)× VOI.
# Oracles below recompute EU(block) independently of the brain's arithmetic (test-oracle).
let
    wm = build_model(["ctx"], [["c"]]); X = ["c"]
    function mk(a, d)             # θ = (2+a)/(4+a+d) via the Beta(2,2) cell prior
        t = build_prior(wm)
        for _ in 1:a; t = observe(wm, t, X, 1); end
        for _ in 1:d; t = observe(wm, t, X, 0); end
        t
    end

    # (a) m=0 is BIT-IDENTICAL to the default (no kwarg) — the safe reduction, over real
    #     contexts and several profiles, with the full three-way decision (ask in play).
    let model = build_model(["tool","rep"], [["bash","read"],["rep0","rep3"]]; p_edge=0.5)
        top = build_prior(model)
        for (Xc,o) in [(["read","rep0"],1),(["bash","rep3"],0),(["bash","rep3"],0)]
            top = observe(model, top, Xc, o)
        end
        allsame = true
        for Xc in (["bash","rep3"], ["read","rep0"]), λ in (0.25,1.0,4.0), q in (0.02,1.0)
            d_def = decide(model, top, Xc, 1.0; aversion=λ, interrupt_cost=q)
            d_m0  = decide(model, top, Xc, 1.0; aversion=λ, interrupt_cost=q, expected_repeats=0.0)
            d_def === d_m0 || (allsame = false)
        end
        check("expected_repeats=0.0 ≡ default decide (bit-identical reduction)", allsame)
    end

    # (b) Threshold oracle (ask suppressed by a huge q): across θ, λ, m, decide blocks IFF
    #     EU(block)=c·[(1+m)(1−θ)−λθ] > 0. m=0 column = the myopic 1/(1+λ). Tie-safe (skip |EU|≈0).
    for (a,d) in [(4,2),(8,2),(2,8),(3,3),(16,2),(2,2),(10,1)]
        wt = mk(a, d); θ = expect(belief_at_context(wm, wt, X), Identity())
        for λ in (0.25, 1.0, 4.0), m in (0.0, 0.5, 2.0, 10.0)
            tf = 1.0 + m
            eu_block = tf*(1.0 - θ) - λ*θ        # EU(block)/c, proceed baseline 0
            abs(eu_block) < 1e-9 && continue      # skip the knife-edge tie
            dec = decide(wm, wt, X, 1.0; aversion=λ, interrupt_cost=1e6, expected_repeats=m)
            # credence-lint: allow — precedent:test-oracle — block boundary vs independently-computed EU(block)
            check("block ⟺ EU(block)>0 [θ=$(round(θ,digits=3)) λ=$λ m=$m]",
                  (dec === :block) == (eu_block > 0.0),
                  "dec=$dec θ=$θ eu_block=$(round(eu_block,digits=4))")
        end
    end

    # (c) HEADLINE myopia fix: a call the myopic EU lets RUN, the look-ahead blocks.
    #     θ=0.6 (a=4,d=2), λ=1 ⇒ myopic θ*=0.5 (0.6>0.5: no block); m=2 ⇒ θ*=0.75 (0.6<0.75: block).
    let wt = mk(4, 2), θ = expect(belief_at_context(wm, wt, X), Identity())
        check("θ=0.6 setup", isapprox(θ, 0.6; atol=1e-9), "got $θ")
        d_my = decide(wm, wt, X, 1.0; aversion=1.0, interrupt_cost=0.05, expected_repeats=0.0)
        d_la = decide(wm, wt, X, 1.0; aversion=1.0, interrupt_cost=0.05, expected_repeats=2.0)
        check("myopic EU does NOT block θ=0.6 (waves the loop through)", d_my !== :block, "got $d_my")
        check("tail-aware EU (m=2) BLOCKS the same call", d_la === :block, "got $d_la")
    end

    # (d) Attention-precious profile (q=1.0): myopic PROCEEDS on the loop; the look-ahead
    #     makes it GOVERN — the limitation-#1 fix (EU/VOI inherit the (1+m)× tail).
    let wt = mk(4, 2)
        d_my = decide(wm, wt, X, 1.0; aversion=1.0, interrupt_cost=1.0, expected_repeats=0.0)
        d_la = decide(wm, wt, X, 1.0; aversion=1.0, interrupt_cost=1.0, expected_repeats=3.0)
        check("attention-precious + myopic → proceed (under-governs the loop)", d_my === :proceed, "got $d_my")
        check("attention-precious + tail-aware → governs (≠proceed)", d_la !== :proceed, "got $d_la")
    end

    # (d-time) TIME coordinate (the MVP's time dial in governance): a call the time-blind EU
    #     PROCEEDS on, valuing the user's wall-clock GOVERNS. θ=0.6, λ=1 ⇒ block EU=−0.2 (<0,
    #     proceed); a block also SAVES E[time] seconds (here w_time·E[time]=0.02·30=0.60 > the
    #     0.20 gap) ⇒ block wins. w_time=0 ⇒ bit-identical regardless of exp_time. q=1.0 isolates
    #     proceed-vs-block (ask suppressed), mirroring (d).
    let wt = mk(4, 2)
        d_blind = decide(wm, wt, X, 1.0; aversion=1.0, interrupt_cost=1.0, w_time=0.0, exp_time=30.0)
        d_ref   = decide(wm, wt, X, 1.0; aversion=1.0, interrupt_cost=1.0)
        d_time  = decide(wm, wt, X, 1.0; aversion=1.0, interrupt_cost=1.0, w_time=0.02, exp_time=30.0)
        check("time-blind (w_time=0) ≡ default decide regardless of exp_time", d_blind === d_ref, "blind=$d_blind ref=$d_ref")
        check("time-blind → proceed (under-values the loop's wall-clock)", d_blind === :proceed, "got $d_blind")
        check("valuing time (w_time·E[time]=0.60 saved) → governs (≠proceed)", d_time !== :proceed, "got $d_time")
    end

    # (e) decide_multi: m=0 ≡ default; m>0 scales the waste tail (harm term untouched).
    let hm = build_model(["ctx"], [["c"]])
        function mkh(a, d)
            t = build_prior(hm)
            for _ in 1:a; t = observe(hm, t, X, 1); end
            for _ in 1:d; t = observe(hm, t, X, 0); end
            t
        end
        wt = mk(4, 2); ht = mkh(0, 8)   # θ_a=0.6, θ_u≈0.17 (safe)
        dm_def = decide_multi(wm, wt, hm, ht, X, X, 1.0; aversion=1.0, interrupt_cost=0.05, harm_cost=1.0)
        dm_m0  = decide_multi(wm, wt, hm, ht, X, X, 1.0; aversion=1.0, interrupt_cost=0.05, harm_cost=1.0, expected_repeats=0.0)
        check("decide_multi expected_repeats=0.0 ≡ default", dm_def === dm_m0, "def=$dm_def m0=$dm_m0")
        dm_tail = decide_multi(wm, wt, hm, ht, X, X, 1.0; aversion=1.0, interrupt_cost=0.05, harm_cost=1.0, expected_repeats=2.0)
        check("decide_multi tail-aware blocks the waste tail (safe ctx, m=2)", dm_tail === :block, "got $dm_tail")
    end
    println("tail-aware EU: threshold oracle + myopia-fix demonstrations PASS")
end

println("="^64)
println("ALL CHECKS PASSED — feature brain (Route B)")
println("="^64)

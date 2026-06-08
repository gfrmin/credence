# Role: eval
#
# regex_impossible.jl — three things the credence-pi governor does that NO regex can,
# each run through the REAL brain (feature_brain.jl) + Tier-1 ops (condition/expect/
# optimise). The point is not "a regex scores lower" — it is that these behaviours are
# STRUCTURALLY outside what any fixed rule can express:
#
#   A3 (learning):    the SAME action gets DIFFERENT decisions for different users,
#                     because `condition` learned each user's feedback. A regex is frozen.
#   A1 (calibration): the brain emits a calibrated P(approve|X) that tracks the empirical
#                     rate; a regex emits a binary at one operating point — no probability.
#   A2 (the dial):    EU-max flips proceed→ask→block as the user's risk/cost dial turns;
#                     a regex has no dial — one rule, one decision, forever.
#
# Run from repo root:
#   julia --project=. apps/credence-pi/eval/regex_impossible.jl

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..", "..", "..", "src")))
using Credence
using Credence: Identity, expect, mean
include(joinpath(@__DIR__, "brain_env.jl"))
include(joinpath(@__DIR__, "..", "brain", "feature_brain.jl"))
using .FeatureBrain: build_model, build_prior, observe, decide, belief_at_context

# A tiny but realistic waste model: (tool, identical-call-count). The classic case is a
# re-run of a build command — waste for a user stuck in a loop, legitimate for a user who
# edits-then-rebuilds. A regex keying on "identical-count ≥ 1" cannot tell them apart.
const MODEL = build_model(["tool", "ident"],
                          [["build", "other"], ["ident-0", "ident-1", "ident-2plus"]])
theta(top, X) = expect(belief_at_context(MODEL, top, X), Identity())   # P(approve|X) via Tier-1

# The strongest fixed rule a regex governor could use here: block any repeated call.
regex_decision(X) = X[2] == "ident-0" ? :proceed : :block

hr() = println("─"^78)

# ── A3 — LEARNING: same action, different decision per user (condition) ──
function demo_learning()
    println("\n████ A3  LEARNING — the same action, governed differently per user ████")
    X = ["build", "ident-2plus"]   # a re-run of a build command
    println("action context X = $X  (a repeated build command)\n")

    # User A edit-then-rebuilds: they keep APPROVING the re-run (it is productive).
    topA = build_prior(MODEL)
    # User B is stuck in a loop: they keep DENYING the re-run (it is waste).
    topB = build_prior(MODEL)
    println(rpad("feedback round", 16), rpad("userA θ→dec", 26), "userB θ→dec")
    for r in 0:4
        decA = decide(MODEL, topA, X, 0.5; aversion = 1.0, interrupt_cost = 0.02)
        decB = decide(MODEL, topB, X, 0.5; aversion = 1.0, interrupt_cost = 0.02)
        println(rpad(r, 16),
                rpad("θ=$(round(theta(topA,X);digits=3)) → $decA", 26),
                "θ=$(round(theta(topB,X);digits=3)) → $decB")
        topA = observe(MODEL, topA, X, 1)   # A approves the re-run
        topB = observe(MODEL, topB, X, 0)   # B denies the re-run
    end
    println("\nregex governor (block if identical-count ≥ 1): $(regex_decision(X)) for BOTH users.")
    println("→ A regex CANNOT learn: it blocks user A's legitimate rebuild forever.")
    println("  The brain diverged from one shared prior using only `condition` on feedback.")
end

# ── A1 — CALIBRATION: the brain's P tracks the empirical rate ──
function demo_calibration()
    println("\n████ A1  CALIBRATION — a posterior P, not a binary ████")
    println("Feed each context a feedback stream with a KNOWN approval rate; read back θ.\n")
    println(rpad("context", 22), rpad("true approve-rate", 19), rpad("brain θ=P(approve|X)", 22), "regex output")
    # three contexts, three true rates; deterministic streams matching each rate
    cases = [(["other", "ident-0"], 0.9, 18, 2),
             (["build", "ident-1"], 0.5, 10, 10),
             (["build", "ident-2plus"], 0.1, 2, 18)]
    for (X, rate, n1, n0) in cases
        top = build_prior(MODEL)
        for _ in 1:n1; top = observe(MODEL, top, X, 1); end
        for _ in 1:n0; top = observe(MODEL, top, X, 0); end
        println(rpad(string(X), 22), rpad(rate, 19),
                rpad(round(theta(top, X); digits=3), 22), "$(regex_decision(X)) (no P)")
    end
    println("\n→ The brain returns a calibrated probability that tracks the empirical rate.")
    println("  A regex returns a label. EU-max needs the probability; a label cannot be")
    println("  thresholded by a continuous cost/risk dial (next).")
end

# ── A2 — THE DIAL: EU-max flips the decision as the risk/cost dial turns ──
function demo_dial()
    println("\n████ A2  THE DIAL — one belief, decision flips as the user's dial turns ████")
    # A CONFIDENT ambiguous belief: many observations averaging 0.5 ⇒ θ=0.5, LOW variance.
    X = ["build", "ident-1"]
    top = build_prior(MODEL)
    for _ in 1:8; top = observe(MODEL, top, X, 1); end
    for _ in 1:8; top = observe(MODEL, top, X, 0); end
    println("Part A — the risk dial (confident belief θ=$(round(theta(top,X);digits=3)), Beta(10,10)):")
    println("  λ = false-block aversion: LOW λ tolerates false-blocks (blocks readily);")
    println("  HIGH λ protects the user's actions (blocks rarely). Threshold θ < 1/(1+λ).\n")
    println("  ", rpad("aversion λ", 16), "EU-max decision")
    for λ in (0.2, 1.0, 4.0)
        println("  ", rpad("λ=$λ", 16), decide(MODEL, top, X, 0.5; aversion = λ, interrupt_cost = 0.02))
    end
    println("\n  → Same belief, decision flips block→proceed as the user turns ONE dial.")
    println("    A regex is one rule with one outcome — there is no dial to turn.\n")

    # Part B: ask is VOI-gated — it depends on the VARIANCE, not just the mean. Same
    # mean θ=0.5, two confidences: the brain asks only when the user's input is worth
    # more than the interruption — which only the full posterior can tell.
    println("Part B — ask is value-of-information-gated (same mean θ=0.5, λ=1, q=\$0.02):")
    println("  ", rpad("belief", 34), "EU-max decision")
    unc = build_prior(MODEL)                                   # uncertain: Beta(2,2), wide
    conf = build_prior(MODEL)
    for _ in 1:8; conf = observe(MODEL, conf, X, 1); end       # confident: Beta(10,10), narrow
    for _ in 1:8; conf = observe(MODEL, conf, X, 0); end
    println("  ", rpad("uncertain θ=0.5 (Beta(2,2), wide)", 34),
            decide(MODEL, unc, X, 0.5; aversion = 1.0, interrupt_cost = 0.02))
    println("  ", rpad("confident θ=0.5 (Beta(10,10), narrow)", 34),
            decide(MODEL, conf, X, 0.5; aversion = 1.0, interrupt_cost = 0.02))
    println("\n  → IDENTICAL mean, OPPOSITE decision: ask when uncertain (the user's input")
    println("    beats the interrupt cost), proceed when confident (asking is not worth it).")
    println("    A regex sees one feature vector → one answer. Even a point-estimate")
    println("    classifier sees θ=0.5 → one answer. Only the FULL posterior asks here.")
end

println("="^78)
println("  credence-pi: three behaviours structurally outside any regex (real brain)")
println("="^78)
demo_learning(); hr()
demo_calibration(); hr()
demo_dial()
println("\n", "="^78)
println("All three are A1/A2/A3 in action: a calibrated belief (Cox), updated by")
println("conditioning (Bayes), driving an EU-max decision (Savage). A regex has none of")
println("these — it is a fixed map from features to a label. That is the irreducible gap.")
println("="^78)

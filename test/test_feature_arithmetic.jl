# test_feature_arithmetic.jl — the numeric sublayer (feature-arithmetic move, coefficient-free).
#
# Sections (feature-arithmetic-design.md §2 test spec):
#   §1  the behaviour-preserving lift: enumerate at max_num_depth=1 reproduces the PRE-change
#       capture (test/fixtures/feature_arithmetic_lift_v1.tsv, SHA 3af9077) — count, show_expr,
#       complexity, prior weight, and the fixed-sequence posterior, all ==.
#   §2  complexity accounting: a product predicate costs 3; a bare-feature atom still costs 1.
#   §3  product discovery: a rule separable ONLY multiplicatively ((a×b) > t) is inexpressible
#       at the depth-1 floor and acquired at max_num_depth=2 — the posterior concentrates on
#       product-lhs programs.
#   §4  determinism: enumeration order reproducible; the numeric roster is decision-free
#       (no NumExpr primitive carries a numeric field — the ConstSlot back-door test).
#   §5  division semantics (§5 Q2): protected Div (x/0 = 0.0) vs analytic quotient AQ, exact.
#
# Run: julia test/test_feature_arithmetic.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, Program, FeatureRef, GTExpr, LTExpr, AndExpr, IfExpr,
                Times, Plus, Minus, Div, AQ, Neg, NumExpr,
                show_expr, show_num, num_complexity, expr_complexity, compile_num,
                enumerate_programs, enumerate_programs_as_measure, weights,
                TaggedBetaPrevision, BetaPrevision, Prevision, MixturePrevision,
                compile_kernel, program_space_observation_kernel, condition

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

println("="^64)
println("feature arithmetic — the numeric sublayer (coefficient-free)")
println("="^64)

# ── §1  the behaviour-preserving lift (capture-before-refactor, == against the golden) ──
let
    g = Grammar(Set([:a, :b]),
                [ProductionRule(:HOT, AndExpr(GTExpr(FeatureRef(:a), 0.7), LTExpr(FeatureRef(:b), 0.3)))], 901)
    as = Symbol[:food, :enemy]
    progs = enumerate_programs(g, 3; action_space = as)
    m = enumerate_programs_as_measure(g, 3; action_space = as)
    w = weights(m.prevision)

    lines = readlines(joinpath(@__DIR__, "fixtures", "feature_arithmetic_lift_v1.tsv"))
    data = [split(l, '\t') for l in lines if !startswith(l, "#")]
    n_expected = parse(Int, split(lines[4], '\t')[2])
    check("§1 program count == golden ($(n_expected))", length(progs) == n_expected,
          "got $(length(progs))")

    # Replay the fixed conditioning sequence exactly as the generator did.
    comps = Prevision[TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0)) for i in eachindex(progs)]
    belief = MixturePrevision(comps, Float64[log(x) for x in w])
    cks = [compile_kernel(p, g, i) for (i, p) in enumerate(progs)]
    for (feats, correct) in [
        (Dict(:a => 0.8, :b => 0.2), Set([:food])),
        (Dict(:a => 0.2, :b => 0.8), Set([:enemy])),
        (Dict(:a => 0.9, :b => 0.1), Set([:food])),
        (Dict(:a => 0.1, :b => 0.9), Set([:enemy])),
        (Dict(:a => 0.75, :b => 0.25), Set([:food])),
        (Dict(:a => 0.3, :b => 0.6), Set([:enemy])),
    ]
        k = program_space_observation_kernel(cks, feats, Dict{Symbol, Any}(), correct)
        belief = condition(belief, k, 1.0)
    end
    post = weights(belief)

    ok = true
    why = ""
    for row in data
        i = parse(Int, row[1])
        if show_expr(progs[i].expr) != row[2]
            ok = false; why = "show_expr[$i]: $(show_expr(progs[i].expr)) != $(row[2])"; break
        end
        if progs[i].complexity != parse(Int, row[3])
            ok = false; why = "complexity[$i]"; break
        end
        # Bit-exact: the lift is the same computation (shortest-roundtrip decimal parse).
        if w[i] != parse(Float64, row[4])
            ok = false; why = "prior[$i]: $(w[i]) != $(row[4])"; break
        end
        if post[i] != parse(Float64, row[5])
            ok = false; why = "posterior[$i]: $(post[i]) != $(row[5])"; break
        end
    end
    check("§1 per-program show_expr, complexity, prior, posterior all == golden", ok, why)
end

# ── §2  complexity accounting ──
let
    prod = GTExpr(Times(FeatureRef(:a), FeatureRef(:b)), 0.25)
    check("§2 product predicate costs 3", expr_complexity(prod) == 3,
          "got $(expr_complexity(prod))")
    bare = GTExpr(FeatureRef(:a), 0.5)
    check("§2 bare-feature atom still costs 1", expr_complexity(bare) == 1)
    check("§2 num_complexity: nested (a×b)−c costs 5",
          num_complexity(Minus(Times(FeatureRef(:a), FeatureRef(:b)), FeatureRef(:c))) == 5)
end

# ── §3  product discovery: multiplicative separability acquired at depth 2 ──
let
    g = Grammar(Set([:x_norm, :y_norm]), ProductionRule[], 902)
    as = Symbol[:food, :enemy]

    # The floor cannot express the product…
    floor_progs = enumerate_programs(g, 2; action_space = as, max_num_depth = 1)
    has_product(p) = occursin("(*", show_expr(p.expr))
    check("§3 floor (max_num_depth=1) contains no product", !any(has_product, floor_progs))

    # …depth 2 can.
    progs = enumerate_programs(g, 2; action_space = as, max_num_depth = 2)
    check("§3 depth 2 contains product atoms", any(has_product, progs),
          "no (* …) among $(length(progs)) programs")

    # BOTH quotient operators enumerate (§5 Q2's operative close: the prior selects per
    # hypothesis — which requires both in the hypothesis space).
    check("§3 depth 2 contains BOTH Div and AQ atoms (Q2 two-operator resolution)",
          any(p -> occursin("(/ ", show_expr(p.expr)), progs) &&
          any(p -> occursin("(aq ", show_expr(p.expr)), progs))

    # The NAMED LIMITATION pin (design Q4 ADDENDUM): compound expressions threshold over the
    # seed grid only — per-NumExpr observed-value grids are re-scoped to the
    # escalate-arithmetic-depth design. This assertion breaks CONSCIOUSLY when that lands.
    compound_thresholds = Set(p.expr.predicate.threshold for p in progs
                              if p.expr isa IfExpr && has_product(p) &&
                                 (p.expr.predicate isa GTExpr || p.expr.predicate isa LTExpr))
    check("§3 compound thresholds ⊆ seed grid (the Q4-ADDENDUM limitation, pinned)",
          issubset(compound_thresholds, Set(Credence.THRESHOLDS)),
          "off-grid compound threshold found — the escalate-depth grid attachment landed; re-baseline")

    # True rule: enemy iff x·y > 0.25 — a hyperbola no single axis-aligned threshold matches:
    # every observation pair below straddles each single feature, but the product separates.
    m = enumerate_programs_as_measure(g, 2; action_space = as, max_num_depth = 2)
    w0 = weights(m.prevision)
    comps = Prevision[TaggedBetaPrevision(i, BetaPrevision(1.0, 1.0)) for i in eachindex(progs)]
    belief = MixturePrevision(comps, Float64[log(x) for x in w0])
    cks = [compile_kernel(p, g, i) for (i, p) in enumerate(progs)]
    obs = [
        (0.9, 0.4, :enemy), (0.4, 0.9, :enemy), (0.6, 0.6, :enemy), (0.8, 0.5, :enemy),
        (0.3, 0.3, :food), (0.6, 0.3, :food), (0.3, 0.6, :food), (0.2, 0.8, :food),
        (0.8, 0.2, :food), (0.45, 0.45, :food),
    ]
    for _round in 1:3, (x, y, label) in obs
        k = program_space_observation_kernel(cks,
            Dict(:x_norm => x, :y_norm => y), Dict{Symbol, Any}(), Set([label]))
        belief = condition(belief, k, 1.0)
    end
    post = weights(belief)
    # credence-lint: allow — precedent:test-oracle — posterior-mass split over product vs non-product programs, the acquisition oracle
    product_mass = sum(post[i] for i in eachindex(progs) if has_product(progs[i]))
    check("§3 posterior concentrates on product-lhs programs (mass > 0.9)",
          product_mass > 0.9, "product mass = $product_mass")
end

# ── §4  determinism + the decision-free roster ──
let
    g = Grammar(Set([:a, :b, :c]), ProductionRule[], 903)
    p1 = enumerate_programs(g, 2; action_space = Symbol[:u, :v], max_num_depth = 2)
    p2 = enumerate_programs(g, 2; action_space = Symbol[:u, :v], max_num_depth = 2)
    check("§4 enumeration order deterministic",
          length(p1) == length(p2) &&
          all(show_expr(p1[i].expr) == show_expr(p2[i].expr) for i in eachindex(p1)))

    # The ConstSlot back-door test (design §6.2): no NumExpr primitive carries a numeric
    # field — a baked coefficient would inject an answer, not a prior. Learnable constants
    # arrive only as data-fit slots (the deferred ConstSlot), never literals in primitives.
    numeric_free = all(T -> all(ft -> !(ft <: Number), fieldtypes(T)),
                       (FeatureRef, Times, Plus, Minus, Div, AQ, Neg))
    check("§4 no NumExpr primitive carries a numeric literal field (decision-free-combinator)",
          numeric_free)
end

# ── §5  division semantics (§5 Q2), exact ──
let
    ts = Dict{Symbol, Any}()
    d = compile_num(Div(FeatureRef(:a), FeatureRef(:b)))
    check("§5 protected Div: x/0 == 0.0 exactly (the documented Koza artifact)",
          d(Dict(:a => 3.0, :b => 0.0), ts) == 0.0)
    check("§5 protected Div: ordinary quotient exact", d(Dict(:a => 3.0, :b => 2.0), ts) == 1.5)
    aq = compile_num(AQ(FeatureRef(:a), FeatureRef(:b)))
    # credence-lint: allow — precedent:test-oracle — manual analytic-quotient oracle
    check("§5 AQ: x/√(1+y²) exact and total at y=0",
          aq(Dict(:a => 3.0, :b => 0.0), ts) == 3.0 &&
          aq(Dict(:a => 3.0, :b => 2.0), ts) == 3.0 / sqrt(5.0))
    n = compile_num(Neg(FeatureRef(:a)))
    check("§5 Neg exact", n(Dict(:a => 0.7), ts) == -0.7)
end

println("="^64)
println("ALL CHECKS PASSED — feature arithmetic")
println("="^64)

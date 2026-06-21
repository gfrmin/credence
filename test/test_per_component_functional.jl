# test_per_component_functional.jl — FiringChoice, the Functional-side dual of
# kernel-side FiringByTag. Per-component dispatch over a mixture:
#   expect(mixture, FiringChoice(fired, when_fires, when_not))
#     = Σ_i w_i · expect(component_i, fired[i] ? when_fires : when_not)
#
# Oracle: an independent manual Σ w_i·g_i(θ_i) with θ_i = α_i/(α_i+β_i) computed
# by hand (precedent:test-oracle). Tolerances: closed-form scalar leaves, so the
# only slack is summation order vs the manual loop — assert rtol ≈ 1e-12.

push!(LOAD_PATH, "src")
using Credence
using Credence: BetaPrevision, MixturePrevision

function check(name, cond, detail="")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("assertion failed: $name")
    end
end

println("="^60)
println("FiringChoice — per-component mixture dispatch (issue #39)")
println("="^60)

# Three Beta components with distinct means; weights 0.5 / 0.3 / 0.2.
const A = (2.0, 3.0)   # mean 0.4
const B = (6.0, 4.0)   # mean 0.6
const C = (1.0, 4.0)   # mean 0.2
betamean((α, β)) = α / (α + β)
const M = (betamean(A), betamean(B), betamean(C))   # (0.4, 0.6, 0.2)
const LOGW = [log(0.5), log(0.3), log(0.2)]
const W = (0.5, 0.3, 0.2)                            # already normalised

# Independent oracle: Σ w_i · g_i where g_i picks the fired/not value per component.
oracle(fired, gfire, gnot) =
    sum(W[i] * (fired[i] ? gfire(M[i]) : gnot(M[i])) for i in 1:3)

prev_mix() = MixturePrevision(BetaPrevision[BetaPrevision(A...), BetaPrevision(B...), BetaPrevision(C...)], copy(LOGW))
meas_mix() = MixtureMeasure(Interval(0.0, 1.0),
    [TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaPrevision(A...)),
     TaggedBetaMeasure(Interval(0.0, 1.0), 2, BetaPrevision(B...)),
     TaggedBetaMeasure(Interval(0.0, 1.0), 3, BetaPrevision(C...))],
    copy(LOGW))

# Functional building blocks used at the real sites.
complement   = LinearCombination(Tuple{Float64, TestFunction}[(-1.0, Identity())], 1.0)        # 1 - θ
half_split   = LinearCombination(Tuple{Float64, TestFunction}[(-1.0/2, Identity())], 1.0/2)    # (1 - θ)/(n-1), n=3
const_half   = LinearCombination(Tuple{Float64, TestFunction}[], 0.5)                          # constant 0.5

# ── when_not = 1 - θ  (grid_world p_enemy shape) ──
let
    fired = [true, false, true]
    got = expect(prev_mix(), FiringChoice(fired, Identity(), complement))
    want = oracle(fired, identity, x -> 1 - x)
    check("1-θ split: expect == oracle (prevision)", isapprox(got, want; rtol=1e-12), "got $got want $want")
    check("1-θ split: literal value 0.36", isapprox(got, 0.36; rtol=1e-12), "got $got")  # 0.5·0.4 + 0.3·0.4 + 0.2·0.2
end

# ── when_not = (1-θ)/(n-1)  (skin / email shape) ──
let
    fired = [true, false, false]
    got = expect(prev_mix(), FiringChoice(fired, Identity(), half_split))
    want = oracle(fired, identity, x -> (1 - x) / 2)
    check("(1-θ)/(n-1) split: expect == oracle", isapprox(got, want; rtol=1e-12), "got $got want $want")
end

# ── when_not = constant 0.5  (rss shape, via empty LinearCombination) ──
let
    fired = [true, false, true]
    got = expect(prev_mix(), FiringChoice(fired, Identity(), const_half))
    want = oracle(fired, identity, _ -> 0.5)
    check("const-0.5 split: expect == oracle", isapprox(got, want; rtol=1e-12), "got $got want $want")
    check("const-0.5 split: literal value 0.39", isapprox(got, 0.39; rtol=1e-12), "got $got")  # 0.5·0.4 + 0.3·0.5 + 0.2·0.2
end

# ── affine when_fires / when_not  (grid_world EU shape: enemy 5-10θ, food -5+10θ) ──
let
    fired = [true, false, true]
    eu_fire = LinearCombination(Tuple{Float64, TestFunction}[(-10.0, Identity())], 5.0)
    eu_not  = LinearCombination(Tuple{Float64, TestFunction}[( 10.0, Identity())], -5.0)
    got = expect(prev_mix(), FiringChoice(fired, eu_fire, eu_not))
    want = oracle(fired, x -> 5 - 10x, x -> -5 + 10x)
    check("affine EU split: expect == oracle", isapprox(got, want; rtol=1e-12), "got $got want $want")
end

# ── Measure side mirrors Prevision side exactly ──
let
    fired = [true, false, true]
    gp = expect(prev_mix(), FiringChoice(fired, Identity(), complement))
    gm = expect(meas_mix(), FiringChoice(fired, Identity(), complement))
    check("MixtureMeasure path == MixturePrevision path", isapprox(gp, gm; rtol=1e-12), "prev $gp meas $gm")
end

# ── all-fired and none-fired degenerate to a uniform functional ──
let
    allfire  = expect(prev_mix(), FiringChoice([true, true, true], Identity(), const_half))
    uniform  = expect(prev_mix(), Identity())
    check("all-fired == uniform Identity expect", isapprox(allfire, uniform; rtol=1e-12), "fc $allfire id $uniform")
end

# ── length mismatch errors at dispatch ──
let
    threw = false
    try
        expect(prev_mix(), FiringChoice([true, false], Identity(), complement))
    catch
        threw = true
    end
    check("length-mismatch fired vs components errors", threw)
end

# ── FiringChoice on a non-mixture prevision errors (inherently mixture-level) ──
let
    threw = false
    try
        expect(BetaPrevision(A...), FiringChoice([true], Identity(), complement))
    catch
        threw = true
    end
    check("non-mixture prevision errors", threw)
end

# ── constructor rejects empty fired ──
let
    threw = false
    try
        FiringChoice(Bool[], Identity(), complement)
    catch
        threw = true
    end
    check("empty fired rejected at construction", threw)
end

println("="^60)
println("ALL FiringChoice TESTS PASSED")
println("="^60)

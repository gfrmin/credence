# test_product_bma_routing.jl — per-factor-routed product conditioning, the
# substrate primitive behind the credence-pi feature brain (Move 3).
#
# Validates the DECLARED shape end-to-end: a MixturePrevision over BN-edge
# structures, each structure a ProductPrevision of TaggedBeta cells with
# globally-unique tags, conditioned by a per-context
# FiringByTag(when_fires=BetaBernoulli, when_not=Flat) kernel. The firing
# factor updates via its conjugate; non-firing factors are Flat no-ops; the
# structure stays a product (no flatten), and the enclosing mixture reweights
# each structure by the firing cell's predictive = the chain-rule marginal
# likelihood.
#
# Oracle: the G1 verification spike (wf_97e1b871-883), itself checked against
# an exact-fraction (Python Fraction) reference. Marginal likelihoods
# S0=1/105, S1=3/175, S2=1/25, S3=3/100. We assert the declared shape
# reproduces the spike's host-fold numbers to <1e-9.
#
# Run from the repo root:
#     julia test/test_product_bma_routing.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: BetaPrevision, TaggedBetaPrevision, ProductPrevision, MixturePrevision,
                FiringByTag, BetaBernoulli, Flat, Kernel, Interval, Finite, mean
using Credence.Ontology: wrap_in_measure, condition
const _pll = Credence.Ontology._predictive_ll

function check(name, cond, detail="")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("assertion failed: $name")
    end
end

# ── The 4 BN-edge structures over features tool∈{bash,read}, rep∈{rep0,rep3}.
#    Globally-unique cell tags let ONE FiringByTag kernel route correctly
#    through the whole nested object (within each structure exactly one cell
#    is in `fires`). ──
#   S0 {}        : cell :all          → tag 1
#   S1 {tool}    : bash 2, read 3
#   S2 {rep}     : rep0 4, rep3 5
#   S3 {tool,rep}: bash_rep0 6, bash_rep3 7, read_rep0 8, read_rep3 9
function cell_tag(structure::Int, tool::String, rep::String)
    structure == 0 && return 1
    structure == 1 && return tool == "bash" ? 2 : 3
    structure == 2 && return rep  == "rep0" ? 4 : 5
    return tool == "bash" ? (rep == "rep0" ? 6 : 7) : (rep == "rep0" ? 8 : 9)
end

prior_cell(tag) = TaggedBetaPrevision(tag, BetaPrevision(2.0, 2.0))

fresh_structures() = MixturePrevision(
    [ProductPrevision([prior_cell(1)]),                                   # S0
     ProductPrevision([prior_cell(2), prior_cell(3)]),                    # S1
     ProductPrevision([prior_cell(4), prior_cell(5)]),                    # S2
     ProductPrevision([prior_cell(6), prior_cell(7),
                       prior_cell(8), prior_cell(9)])],                   # S3
    fill(log(0.25), 4),                                                   # uniform p=0.5
)

# Per-context kernel: BetaBernoulli on the cells in F(X), Flat (no-op) on the
# rest. log_density follows the TaggedBeta protocol — it takes the cell MEASURE
# and returns that cell's marginal log-predictive (used by _predictive_ll).
fires_for(tool, rep) = Set(cell_tag(s, tool, rep) for s in 0:3)
approve_kernel(fires::Set{Int}) = Kernel(
    Interval(0.0, 1.0), Finite([0, 1]),
    theta -> theta,                                          # generate stub (unused)
    (m, o) -> (a = mean(m.beta); o == 1 ? log(a) : log(1.0 - a));
    likelihood_family = FiringByTag(fires, BetaBernoulli(), Flat()),
)

# Find cell with `tag` in a conditioned structure (ProductPrevision) and return
# its posterior mean — a manual test-oracle readout (precedent: test-oracle).
function cell_mean(structure::ProductPrevision, tag::Int)
    for f in structure.factors
        f isa TaggedBetaPrevision && f.tag == tag && return mean(wrap_in_measure(f).beta)
    end
    error("no cell with tag $tag")
end

# Σ_i P(Si | data) · mean(Si's cell for this context).
function map_predictive(top::MixturePrevision, tool, rep)
    w = exp.(top.log_weights)                                # normalised at construction
    sum(w[i] * cell_mean(top.components[i], cell_tag(s, tool, rep)) for (i, s) in enumerate(0:3))
end

const OBS = [("read", "rep0", 1), ("bash", "rep0", 1), ("bash", "rep3", 0),
             ("bash", "rep3", 0), ("bash", "rep3", 0), ("read", "rep0", 1)]

println("="^60)
println("per-factor-routed product conditioning — credence-pi Move 3")
println("="^60)

# ── 1. Focused unit test: one structure, one observation. The firing cell
#       updates; every other cell is untouched; the product predictive equals
#       exactly the firing cell's predictive. ──
let s3 = ProductPrevision([prior_cell(6), prior_cell(7), prior_cell(8), prior_cell(9)])
    k = approve_kernel(fires_for("bash", "rep3"))            # fires {1,2,5,7}; in S3 only tag 7
    # predictive of a deny (o=0) at prior Beta(2,2): log(1 - 0.5) = log 0.5
    pll = _pll(wrap_in_measure(s3), k, 0)
    check("product predictive = firing cell's predictive (log 0.5)",
          isapprox(pll, log(0.5); atol=1e-12), "got $pll")

    cond = condition(s3, k, 0)
    check("only the firing cell (tag 7) updated → Beta(2,3) → mean 0.4",
          isapprox(cell_mean(cond, 7), 0.4; atol=1e-12), "got $(cell_mean(cond, 7))")
    check("non-firing cell tag 6 untouched → mean 0.5",
          isapprox(cell_mean(cond, 6), 0.5; atol=1e-12), "got $(cell_mean(cond, 6))")
    check("non-firing cell tag 8 untouched → mean 0.5",
          isapprox(cell_mean(cond, 8), 0.5; atol=1e-12), "got $(cell_mean(cond, 8))")
    check("conditioned structure is still a ProductPrevision (no flatten)",
          cond isa ProductPrevision, "got $(typeof(cond))")
end

# ── 2. Prior predictives are 0.5 (uniform structure prior, all cells Beta(2,2)). ──
let top = fresh_structures()
    check("prior P(approve | bash,rep3) = 0.5",
          isapprox(map_predictive(top, "bash", "rep3"), 0.5; atol=1e-12),
          "got $(map_predictive(top, "bash", "rep3"))")
    check("prior P(approve | read,rep0) = 0.5",
          isapprox(map_predictive(top, "read", "rep0"), 0.5; atol=1e-12),
          "got $(map_predictive(top, "read", "rep0"))")
end

# ── 3. Full BMA over the 6-obs sequence — exact match to the spike oracle. ──
let top = fresh_structures()
    for (tool, rep, o) in OBS
        top = condition(top, approve_kernel(fires_for(tool, rep)), o)
    end
    w = exp.(top.log_weights)
    check("Σ structure posterior = 1", isapprox(sum(w), 1.0; atol=1e-12), "got $(sum(w))")
    check("structure posterior S0 = 20/203", isapprox(w[1], 0.09852216748768473; atol=1e-9), "got $(w[1])")
    check("structure posterior S1 = 36/203", isapprox(w[2], 0.17733990147783252; atol=1e-9), "got $(w[2])")
    check("structure posterior S2 = 84/203", isapprox(w[3], 0.41379310344827586; atol=1e-9), "got $(w[3])")
    check("structure posterior S3 = 63/203", isapprox(w[4], 0.31034482758620690; atol=1e-9), "got $(w[4])")
    # rep-parent structures dominate (the loop signal), as they must.
    check("rep-parent structures S2+S3 dominate (> 0.7)", w[3] + w[4] > 0.7, "got $(w[3]+w[4])")

    pbr = map_predictive(top, "bash", "rep3")
    prr = map_predictive(top, "read", "rep0")
    check("P(approve | bash,rep3) = 131/406 ≈ 0.3227 (deny the loop)",
          isapprox(pbr, 0.32266009852216746; atol=1e-9), "got $pbr")
    check("P(approve | read,rep0) = 136/203 ≈ 0.6700 (approve the novel call)",
          isapprox(prr, 0.66995073891625620; atol=1e-9), "got $prr")
    # The surgical win in one line: the same evidence pushes the repeated call
    # toward deny and the novel call toward approve — impossible for one global Beta.
    check("surgical: predictive(bash,rep3) < 0.5 < predictive(read,rep0)",
          pbr < 0.5 < prr, "bash,rep3=$pbr read,rep0=$prr")
end

# ── 4. Degenerate reduction: point mass on S0 ≡ a single global Beta that
#       ignores context (all 6 outcomes 1,1,0,0,0,1 → Beta(5,5) → 0.5). ──
let top = MixturePrevision(
        [ProductPrevision([prior_cell(1)]),
         ProductPrevision([prior_cell(2), prior_cell(3)]),
         ProductPrevision([prior_cell(4), prior_cell(5)]),
         ProductPrevision([prior_cell(6), prior_cell(7), prior_cell(8), prior_cell(9)])],
        [0.0, -Inf, -Inf, -Inf])                              # point mass on S0
    for (tool, rep, o) in OBS
        top = condition(top, approve_kernel(fires_for(tool, rep)), o)
    end
    w = exp.(top.log_weights)
    check("S0 point mass stays a point mass", isapprox(w[1], 1.0; atol=1e-12), "got $(w[1])")
    check("∅-structure ≡ global Beta(5,5): predictive 0.5 at every context",
          isapprox(map_predictive(top, "bash", "rep3"), 0.5; atol=1e-12) &&
          isapprox(map_predictive(top, "read", "rep0"), 0.5; atol=1e-12),
          "bash,rep3=$(map_predictive(top,"bash","rep3")) read,rep0=$(map_predictive(top,"read","rep0"))")
end

println("="^60)
println("ALL CHECKS PASSED — per-factor-routed product conditioning")
println("="^60)

# test_qa_benchmark_real_embeddings.jl — Paper 1, B2c.
#
# End-to-end check on the committed REAL embedding fixture (not synthetic):
# the Gaussian-NB classifier, with per-dimension standardisation, must
# recover the question categories under leave-one-out well above chance.
#
# This is the regression guard for the standardisation finding: raw
# all-MiniLM embeddings are anisotropic and collapse the classifier to the
# majority class (LOO ≈ 0.30 ≈ chance); per-dim standardisation
# (leak-free, per LOO fold) lifts it to ≈ 0.78. If this test drops toward
# chance, standardisation has regressed.
#
# Run from the repo root:
#     julia test/test_qa_benchmark_real_embeddings.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
import JSON3

include(joinpath(@__DIR__, "..", "apps", "julia", "qa_benchmark", "category_inference.jl"))

function check(name, cond, detail="")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("real-embedding assertion failed: $name")
    end
end

fixdir = joinpath(@__DIR__, "..", "apps", "julia", "qa_benchmark", "fixtures")
bank = JSON3.read(read(joinpath(fixdir, "question_bank.json"), String))
emb = JSON3.read(read(joinpath(fixdir, "question_embeddings.json"), String))
byid = emb.embeddings

N = length(bank)
D = emb.dim
X = Matrix{Float64}(undef, N, D)
cats = Vector{Symbol}(undef, N)
for (i, q) in enumerate(bank)
    X[i, :] = Float64.(byid[Symbol(q.id)])
    cats[i] = Symbol(q.category)
end
K = length(unique(cats))

println("="^60)
println("Paper 1 B2c — real-embedding LOO ($(emb.model))")
println("="^60)

check("fixture shape: N=50, D=384, K=5", N == 50 && D == 384 && K == 5,
      "N=$N D=$D K=$K")

posts = loo_category_inference(X, cats)
acc = count(i -> findmax(posts[i])[2] == cats[i], 1:N) / N
check("real-embedding LOO accuracy clears 1/K + 0.3 (= $(round(1/K + 0.3, digits=2)))",
      acc > 1/K + 0.3, "acc=$(round(acc, digits=3))")

pmass = sum(posts[i][cats[i]] for i in 1:N) / N
check("soft posterior calibrated: mean P(true) clears 1/K + 0.3",
      pmass > 1/K + 0.3, "mean P(true)=$(round(pmass, digits=3))")

println("="^60)
println("REAL-EMBEDDING LOO PASSED — acc=$(round(acc, digits=3)) P(true)=$(round(pmass, digits=3))")
println("="^60)

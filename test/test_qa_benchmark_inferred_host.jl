# test_qa_benchmark_inferred_host.jl — Paper 1, B2c host wiring.
#
# Smoke test: the host runs the Bayesian agent under BOTH given (one-hot)
# and inferred (soft posterior from the offline embedding fixture)
# categories, fully offline. Verifies the B2c MixturePrevision flows
# through VOI / the answer-kernel / learning without error, that both
# paths produce a full 50-question result, and that each is deterministic.
#
# Run from the repo root:
#     julia test/test_qa_benchmark_inferred_host.jl

include(joinpath(@__DIR__, "..", "apps", "julia", "qa_benchmark", "host.jl"))
using Random

function check(name, cond, detail="")
    if cond
        println("PASSED: $name")
    else
        println("FAILED: $name — $detail")
        error("inferred-host assertion failed: $name")
    end
end

println("="^60)
println("Paper 1 B2c — host wiring under inferred categories (offline)")
println("="^60)

tools = make_spec_tools()
questions = get_questions(; seed=0)
rt = generate_response_table(tools, questions, MersenneTwister(0))

# Posteriors from the committed fixture (no network).
cp = category_posteriors_from_fixture()
check("posteriors: one per question (50)", length(cp) == 50, "got $(length(cp))")
check("each posterior sums to 1.0 over CATEGORIES (atol 1e-9)",
      all(abs(sum(v) - 1.0) < 1e-9 for v in values(cp)))
check("each posterior aligned to CATEGORIES (length $(length(CATEGORIES)))",
      all(length(v) == length(CATEGORIES) for v in values(cp)))

# Given-category path: runs, full result, deterministic.
g1 = run_bayesian_seed(tools, questions, rt)
g2 = run_bayesian_seed(tools, questions, rt)
check("given path: 50 records", length(g1.records) == 50)
check("given path: deterministic (==)", g1.total_score == g2.total_score,
      "$(g1.total_score) vs $(g2.total_score)")

# Inferred-category path: runs, full result, deterministic. This is the
# MixturePrevision-through-VOI integration check.
i1 = run_bayesian_seed(tools, questions, rt; category_posteriors=cp)
i2 = run_bayesian_seed(tools, questions, rt; category_posteriors=cp)
check("inferred path: 50 records", length(i1.records) == 50)
check("inferred path: deterministic (==)", i1.total_score == i2.total_score,
      "$(i1.total_score) vs $(i2.total_score)")
check("inferred path: finite score", isfinite(i1.total_score))

println("="^60)
println("HOST WIRING OK — given=$(round(g1.total_score, digits=1)) inferred=$(round(i1.total_score, digits=1))")
println("="^60)

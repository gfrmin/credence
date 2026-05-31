#!/usr/bin/env julia
# scripts/export_question_bank.jl — dump the QA benchmark question bank to
# JSON for offline embedding.
#
# The canonical bank lives in apps/julia/qa_benchmark/environment.jl. This
# exports (id, text, category) so the Python embedder
# (scripts/paper1-embed-questions.py) need not parse Julia. Re-run this
# whenever QUESTION_BANK changes; then regenerate the embedding fixture.
#
# Usage (from repo root):
#     julia scripts/export_question_bank.jl

using JSON3

include(joinpath(@__DIR__, "..", "apps", "julia", "qa_benchmark", "environment.jl"))

out = [Dict("id" => q.id, "text" => q.text, "category" => q.category)
       for q in QUESTION_BANK]

dir = joinpath(@__DIR__, "..", "apps", "julia", "qa_benchmark", "fixtures")
mkpath(dir)
path = joinpath(dir, "question_bank.json")
open(path, "w") do io
    JSON3.write(io, out)
    write(io, "\n")
end

println("wrote $(length(out)) questions to $path")

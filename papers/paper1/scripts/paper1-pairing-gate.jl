# Paper 1 B4 — reuse-validity gate.
#
# Proves the saved no-category Haiku + Llama runs (reused as the fair LLM
# condition) reproduce bit-exact against today's environment: for every recorded
# (seed, question, tool) the simulated tool response in the DB must equal the
# response table regenerated from the same seed. If it passes, reusing the saved
# LLM runs is validly paired with the freshly-run inferred agents. Offline:
#   julia --project=. scripts/paper1-pairing-gate.jl
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
push!(LOAD_PATH, joinpath(ROOT, "src"))
using Credence
using Random
include(joinpath(ROOT, "apps", "julia", "qa_benchmark", "environment.jl"))
using SQLite, DBInterface
import JSON3

const DB = joinpath(ROOT, "apps", "julia", "qa_benchmark", "results", "benchmark.db")
db = SQLite.DB(DB)
tools = make_spec_tools()

function seed_table(seed)
    questions = get_questions(; seed = seed)
    rt = generate_response_table(tools, questions, MersenneTwister(seed))
    rt, Dict(q.id => i for (i, q) in enumerate(questions))
end

checked = 0; mism = 0; rows = 0
for agent in ("claude-haiku-4-5-20251001", "llama3.1")
    data = Tuple{Int,String,String}[]   # SQLite.Row is a streaming cursor — materialise now
    for row in DBInterface.execute(db,
        "SELECT r.seed AS seed, q.question_id AS qid, q.tool_responses AS tr " *
        "FROM questions q JOIN runs r ON q.run_id=r.id WHERE r.agent=?", (agent,))
        push!(data, (Int(row.seed), String(row.qid), String(row.tr)))
    end
    tabs = Dict(s => seed_table(s) for s in unique(first.(data)))
    for (seed, qid, trstr) in data
        global rows += 1
        rt, qpos = tabs[seed]
        if !haskey(qpos, qid)
            println("UNKNOWN qid $qid (seed $seed)"); global mism += 1; continue
        end
        qi = qpos[qid]
        for (tk, rv) in pairs(JSON3.read(trstr))
            global checked += 1
            rt[qi, parse(Int, String(tk))] == Int(rv) ||
                (global mism += 1; println("MISMATCH $agent seed $seed $qid tool $tk"))
        end
    end
end
println("PAIRING GATE: rows=$rows  tool-response checks=$checked  mismatches=$mism")
@assert mism == 0 "reuse INVALID — environment drifted from the saved LLM runs"
println("PASS ✓ — reused Haiku+Llama runs reproduce against today's environment (reuse valid)")

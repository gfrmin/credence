#!/usr/bin/env julia
# Role: tests
"""
    test_manifest.jl — GET /manifest serves the bdsl vocabulary through the ONE library parser.

The de-dup proof (move-4-design §0, §5 Q1): the body has no BDSL parser; the daemon parses
`bdsl/capabilities.bdsl` + `bdsl/features.bdsl` via `Credence.parse_all` and serves the effector /
feature vocabulary the TS body verifies against. Checks: `manifest_dict` round-trips every declared
effector (with its parameter (name,type)s) and every declared feature (with its space); a live HTTP
`GET /manifest` returns the same JSON; and the served effectors are exactly the brain's action-space.

Run from the credence repo root:
    julia --project=. apps/answer-brain/tests/julia/test_manifest.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence
using JSON3
using HTTP

include(joinpath(@__DIR__, "..", "..", "brain", "answer_brain.jl"))
using .AnswerBrain
include(joinpath(@__DIR__, "..", "..", "daemon", "server.jl"))
using .Server

const PASSED = String[]
function check(name::AbstractString, cond::Bool; detail::AbstractString = "")
    if cond
        push!(PASSED, name)
        println("PASSED: ", name)
    else
        println("FAILED: ", name, " — ", detail)
        error("assertion failed: $name")
    end
end

# Look up a served effector / feature by name in the manifest's vectors.
_find(items, name) = (i = findfirst(x -> x["name"] == name, items); i === nothing ? nothing : items[i])
_has_param(eff, pname, ptype) = any(p -> p["name"] == pname && p["type"] == ptype, eff["parameters"])

# ── 1. The pure function: parse the real bdsl ────────────────────────────────────────────
man = Server.Manifest.manifest_dict(
    capabilities_path = Server.CAPABILITIES_PATH,
    features_path = Server.FEATURES_PATH,
)

effs = man["effectors"]
feats = man["features"]

# The served effectors are EXACTLY the brain's declared action-space (capabilities.bdsl).
eff_names = sort([e["name"] for e in effs])
check("effectors are {answer, ask-user, abstain, gather}",
      eff_names == ["abstain", "answer", "ask-user", "gather"];
      detail = "got $(eff_names)")

answer = _find(effs, "answer")
check("answer effector present", answer !== nothing)
check("answer params = value/credence/citations (all string)",
      _has_param(answer, "value", "string") &&
      _has_param(answer, "credence", "string") &&
      _has_param(answer, "citations", "string") &&
      length(answer["parameters"]) == 3;
      detail = "got $(answer["parameters"])")

check("ask-user takes (text string)",
      (a = _find(effs, "ask-user"); a !== nothing && _has_param(a, "text", "string") &&
       length(a["parameters"]) == 1))
check("abstain takes (reason string)",
      (a = _find(effs, "abstain"); a !== nothing && _has_param(a, "reason", "string") &&
       length(a["parameters"]) == 1))
check("gather takes (target string)",
      (a = _find(effs, "gather"); a !== nothing && _has_param(a, "target", "string") &&
       length(a["parameters"]) == 1))

# The four posterior-shape features, each with its declared space (features.bdsl).
expected_features = Dict(
    "posterior-dispersion" => "dispersion-space",
    "leader-credence-band" => "leader-band-space",
    "candidates-era-split" => "era-split-space",
    "owner-scoped"         => "owner-scoped-space",
)
feat_names = sort([f["name"] for f in feats])
check("features are the four posterior-shape signals",
      feat_names == sort(collect(keys(expected_features)));
      detail = "got $(feat_names)")
for (fname, space) in expected_features
    f = _find(feats, fname)
    check("feature $(fname) → $(space)",
          f !== nothing && f["spaceName"] == space;
          detail = f === nothing ? "absent" : "got $(f["spaceName"])")
end

# ── 2. Live HTTP: GET /manifest serves the same JSON ─────────────────────────────────────
const PORT = 8801
server = start_daemon(; port = PORT)
try
    # Give the listener a moment to bind.
    local ok = false
    for _ in 1:50
        try
            r = HTTP.get("http://127.0.0.1:$(PORT)/ready"; retry = false, connect_timeout = 1)
            ok = (r.status == 200); break
        catch
            sleep(0.1)
        end
    end
    check("daemon /ready up", ok)

    resp = HTTP.get("http://127.0.0.1:$(PORT)/manifest"; retry = false)
    check("GET /manifest → 200", resp.status == 200)
    body = JSON3.read(String(resp.body))
    wire_effs = sort([String(e["name"]) for e in body["effectors"]])
    wire_feats = sort([String(f["name"]) for f in body["features"]])
    check("HTTP effectors match the pure function",
          wire_effs == ["abstain", "answer", "ask-user", "gather"];
          detail = "got $(wire_effs)")
    check("HTTP features match the pure function",
          wire_feats == sort(collect(keys(expected_features)));
          detail = "got $(wire_feats)")
    # Spot-check a nested parameter survives the JSON round-trip.
    wire_answer = body["effectors"][findfirst(e -> e["name"] == "answer", body["effectors"])]
    check("HTTP answer carries (value string)",
          any(p -> String(p["name"]) == "value" && String(p["type"]) == "string",
              wire_answer["parameters"]))
finally
    stop_daemon(server)
end

println("\n", repeat("─", 60))
println("test_manifest.jl: ", length(PASSED), " checks passed.")

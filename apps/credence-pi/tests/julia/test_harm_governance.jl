# Role: tests
# test_harm_governance.jl — multi-outcome (harm × waste) governance end-to-end through the
# REAL wire path (load the bdsl env → wire_brain! → decide-action) with the SHIPPED frozen
# harm posterior (brain/harm_brain.jls). Verifies:
#   * harm OFF (harm-cost=0, the default) ⇒ behaviour is the waste-only path (backward-compat).
#   * harm ON  ⇒ a tainted external-send (P(unsafe)≈0.93) is BLOCKED — injected-data exfil.
#   * harm ON  ⇒ a clean, learned-wanted read (P(unsafe)≈0.01) still PROCEEDS — no over-block.
#
# Run from repo root:
#     julia --project=. apps/credence-pi/tests/julia/test_harm_governance.jl

push!(LOAD_PATH, "src")
using Credence
using Credence: Eval, Parse

include(joinpath(@__DIR__, "..", "..", "brain", "feature_brain.jl"))
using .FeatureBrain: wire_brain!

const BDSL = joinpath(@__DIR__, "..", "..", "bdsl")

function check(name, cond, detail="")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("assertion failed: $name"))
end

# Build the daemon env exactly as the daemon does (stdlib + the three .bdsl), then set the
# harm-cost dial and wire the brain. harm-cost>0 + safety-features + the shipped harm_brain.jls
# (auto-found next to feature_brain.jl) ⇒ multi-outcome governance.
function build_env(harm_cost; harm_response="ask")
    env = Eval.default_env(); env[:__toplevel__] = true
    for expr in Parse.parse_all(read(joinpath("src", "stdlib.bdsl"), String)); Eval.eval_dsl(expr, env); end
    for f in ("capabilities.bdsl", "features.bdsl", "utility.bdsl")
        for expr in Parse.parse_all(read(joinpath(BDSL, f), String)); Eval.eval_dsl(expr, env); end
    end
    env[Symbol("harm-cost")] = harm_cost
    env[Symbol("harm-response")] = harm_response
    wire_brain!(env)
    env
end

# A full feature dict (all 6 waste + 4 safety features the brain declares).
function event(; tool, action_class, taint, imperative="no", chain="no",
               rep="rep-0", ident="ident-0", parent="none", wd="no-path", tsu="gt-10m")
    Dict{String,Any}(
        "tool-name"=>tool, "working-directory-relative"=>wd, "parent-tool-call-name"=>parent,
        "recent-repetition-count"=>rep, "recent-identical-call-count"=>ident,
        "time-since-last-user-message"=>tsu,
        "action-class"=>action_class, "taint-flow"=>taint,
        "injected-imperative"=>imperative, "cred-exfil-chain"=>chain)
end

println("="^64); println("harm governance (multi-outcome) — #27"); println("="^64)

const TAINTED_SEND = event(tool="other", action_class="external-send",
                           taint="tainted-external-target", imperative="yes")
const CLEAN_READ   = event(tool="read", action_class="read-only", taint="none")

# ── harm OFF (default): the waste-only path. A cold belief asks (voi-gated), never blocks
#    on the harm signal (there is none). ──
let env = build_env(0.0)
    decide = env[Symbol("decide-action")]; prior = env[Symbol("make-prior")]()
    d = decide(prior, TAINTED_SEND, 0.5)
    check("harm OFF: tainted send is NOT blocked by harm (waste-only)", d !== :block, "got $d")
end

# ── harm ON, ASK mode (the research-stage default): a harm-driven stop is a CONFIRMATION. ──
let env = build_env(1.0; harm_response="ask")
    decide = env[Symbol("decide-action")]; observe = env[Symbol("observe-response")]
    prior = env[Symbol("make-prior")]()
    d_send = decide(prior, TAINTED_SEND, 0.5)
    check("harm ON/ask: tainted external-send → ASK (confirm, not silent block)", d_send === :ask, "got $d_send")

    # A clean read the user has repeatedly approved ⇒ high P(approve), P(unsafe)≈0.01 ⇒ proceed.
    top = prior
    for _ in 1:8; top = observe(top, CLEAN_READ, 1); end
    d_read = decide(top, CLEAN_READ, 0.5)
    check("harm ON/ask: clean, learned-wanted read still PROCEEDS (no over-block)", d_read === :proceed, "got $d_read")

    # Even a learned-wanted call, once tainted, is escalated to ASK (the harm couples in).
    top2 = prior
    for _ in 1:8; top2 = observe(top2, TAINTED_SEND, 1); end
    d_flip = decide(top2, TAINTED_SEND, 0.5)
    check("harm ON/ask: learned-wanted tainted send → ASK (harm couples in)", d_flip === :ask, "got $d_flip")
end

# ── harm ON, BLOCK mode (enforce; for once the belief is calibrated on real usage). ──
let env = build_env(1.0; harm_response="block")
    decide = env[Symbol("decide-action")]; prior = env[Symbol("make-prior")]()
    d_send = decide(prior, TAINTED_SEND, 0.5)
    check("harm ON/block: tainted external-send BLOCKED (enforce mode)", d_send === :block, "got $d_send")
    # Waste-driven block is unaffected by harm-response (a non-tainted repeated loop still blocks
    # under the waste term); here we just confirm a clean read is not harm-blocked.
    d_read = decide(prior, CLEAN_READ, 0.5)
    check("harm ON/block: clean read not harm-blocked at cold start", d_read !== :block, "got $d_read")
end

println("="^64); println("ALL CHECKS PASSED — harm governance"); println("="^64)

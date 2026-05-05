#!/usr/bin/env julia
# Role: tests
"""
    test_bdsl_primitives.jl — Step 1 of credence-pi: primitives audit.

Exercises every BDSL primitive added in step 1 in isolation. Step 2
will compose these into the Pass-1 BDSL programs at apps/credence-pi/
bdsl/; this file only confirms each primitive is reachable from
default_env() and behaves the way SPEC.md presumes.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "src"))
using Credence
using Credence.Eval: EffectorDecl, FeatureDecl

const PASSED = []
function ok(name)
    push!(PASSED, name)
    println("PASSED: ", name)
end

# ── 1. quote ────────────────────────────────────────────────────────────

@assert run_dsl("(quote foo)") === :foo
@assert run_dsl("(quote 42)") == 42        # numeric atoms self-quote
@assert run_dsl("(= (quote foo) (quote foo))") === true
@assert run_dsl("(= (quote foo) (quote bar))") === false
ok("quote returns the atom unevaluated; = compares quoted symbols")

# ── 2. cond ─────────────────────────────────────────────────────────────

@assert run_dsl("""
(cond
  ((= 1 2) (quote nope))
  ((= 1 1) (quote yep))
  (else    (quote fallback)))
""") === :yep
ok("cond returns the first matching clause")

@assert run_dsl("""
(cond
  ((= 1 2) (quote a))
  ((= 2 3) (quote b))
  (else    (quote c)))
""") === :c
ok("cond falls through to else")

let raised = false
    try
        run_dsl("""
        (cond
          ((= 1 2) (quote a))
          ((= 2 3) (quote b)))
        """)
    catch e
        raised = true
        @assert occursin("no clause matched", string(e))
    end
    @assert raised
    ok("cond with no match and no else raises")
end

# ── 3. apply ────────────────────────────────────────────────────────────

let s = run_dsl("(apply space :finite (list 1 2 3))")
    @assert s isa Finite
    @assert support(s) == [1, 2, 3]
    ok("apply splats a list of ints into (space :finite ...)")
end

let s = run_dsl("""
(apply space :finite (list (quote ask) (quote proceed) (quote block)))
""")
    @assert s isa Finite
    @assert support(s) == [:ask, :proceed, :block]
    ok("apply splats a list of quoted symbols into a Finite space")
end

# ── 4. error ────────────────────────────────────────────────────────────

let raised = false
    try
        run_dsl("(error \"boom\")")
    catch e
        raised = true
        @assert occursin("boom", string(e))
    end
    @assert raised
    ok("error raises with the given message")
end

# ── 5. lookup ───────────────────────────────────────────────────────────

# Build a lambda over a dict, then call it from Julia with a Dict literal.
let env = load_dsl("""
    (define get-response
      (lambda (event)
        (lookup event (quote response))))
    """)
    fn = env[Symbol("get-response")]
    @assert fn(Dict("response" => "yes")) == "yes"
    @assert fn(Dict(:response => "no")) == "no"   # symbol-keyed dicts also work
    ok("lookup retrieves dict values by quoted-symbol key (string and symbol keys)")

    let raised = false
        try
            fn(Dict("other" => "x"))
        catch e
            raised = true
            @assert occursin("not found", string(e))
        end
        @assert raised
        ok("lookup raises on missing key")
    end
end

# ── 6. effector + effector-names ────────────────────────────────────────

let env = load_dsl("""
    (define manifest
      (list
        (effector ask
          (parameters (text string)))
        (effector proceed
          (parameters))
        (effector block
          (parameters (reason string)))))
    """)
    manifest = env[:manifest]
    @assert length(manifest) == 3
    @assert all(d -> d isa EffectorDecl, manifest)
    @assert manifest[1].name === :ask
    @assert manifest[1].parameters == [(name = :text, type = :string)]
    @assert manifest[2].name === :proceed
    @assert isempty(manifest[2].parameters)
    @assert manifest[3].parameters == [(name = :reason, type = :string)]
    ok("effector form returns EffectorDecl with parsed parameters")
end

@assert run_dsl("""
(let manifest (list
                (effector ask     (parameters (text string)))
                (effector proceed (parameters))
                (effector block   (parameters (reason string))))
  (effector-names manifest))
""") == [:ask, :proceed, :block]
ok("effector-names extracts symbols in declaration order")

# ── 7. feature + feature-names ──────────────────────────────────────────

let env = load_dsl("""
    (define tool-name-space (space :finite read write bash))
    (define rep-count-space (space :finite rep-0 rep-1 rep-2))
    (define features
      (list
        (feature tool-name           tool-name-space)
        (feature recent-repetition   rep-count-space)))
    """)
    features = env[:features]
    @assert length(features) == 2
    @assert all(d -> d isa FeatureDecl, features)
    @assert features[1].name === Symbol("tool-name")
    @assert features[1].space isa Finite
    @assert support(features[1].space) == [:read, :write, :bash]
    @assert features[2].name === Symbol("recent-repetition")
    ok("feature form returns FeatureDecl carrying the resolved Space")
end

@assert run_dsl("""
(let s (space :finite a b)
  (let features (list
                  (feature one s)
                  (feature two s))
    (feature-names features)))
""") == [:one, :two]
ok("feature-names extracts symbols in declaration order")

# ── 8. Integration: action-space construction the way decide.bdsl will ──

let s = run_dsl("""
(let manifest (list
                (effector ask     (parameters (text string)))
                (effector proceed (parameters))
                (effector block   (parameters (reason string))))
  (apply space :finite (effector-names manifest)))
""")
    @assert s isa Finite
    @assert support(s) == [:ask, :proceed, :block]
    ok("(apply space :finite (effector-names manifest)) yields the three-action space")
end

println()
println("=" ^ 60)
println("ALL ", length(PASSED), " ASSERTIONS PASSED")
println("=" ^ 60)

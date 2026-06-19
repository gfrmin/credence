#!/usr/bin/env julia
# scripts/capture-invariance.jl — Posture 4 Move 0 pre-branch invariance capture.
#
# Walks every `test/test_*.jl` file, instruments three assertion idioms
# (`@assert`, `check(name, cond, detail)`, `@check(name, expr)`), classifies
# each assertion by shape (exact / tolerance / directional / structural /
# failing) per move-0-design.md §3, captures operand values at pre-refactor
# time, and serialises the result to `test/fixtures/posture-3-capture/`.
# Also walks `tools/credence-lint/corpus/<slug>/bad2_*.{jl,py}` as a separate
# structural-invariance channel (Prompt 0 task 3).
#
# Usage:
#   julia --project=scripts scripts/capture-invariance.jl --output DIR
#   julia --project=scripts scripts/capture-invariance.jl --verify
#
# The --verify flag runs the capture twice into /tmp/capture-{a,b}/ and
# diffs them (per move-0-design.md §4 double-run protocol).
#
# Design decisions (resolutions for move-0-design.md §5):
#   Q1: Four shapes (exact / tolerance / directional / structural) orthogonal
#       to stratum (which applies only to tolerance). Membership is a
#       Structural subtype with set-equivalence gating.
#   Q2: Conservative — capture sample sequences downstream of every
#       Random.seed! call site by recording the seed and the test file's
#       entry-time RNG state.
#   Q3: Pin Linux x86_64 + record Julia version + CPU string in manifest.
#   Q4: Capture failing sites truthfully as Failing shape with expression
#       source, captured operands, and failure reason (returned-false vs
#       threw); continue execution within the assertion call (do not halt).
#       Move 10 may upgrade broken→passing when the cleaner foundation fixes.

using MacroTools
using TOML
using SHA
using Serialization
using Random
using Dates

# ═════════════════════════════════════════════════════════════════════════════
# Shape types — the captured-assertion taxonomy from §3
# ═════════════════════════════════════════════════════════════════════════════

abstract type CapturedAssertion end

"""Exact: `lhs == rhs` — post-refactor must match bit-exact."""
struct ExactShape <: CapturedAssertion
    idiom::Symbol                                 # :assert, :check, :check_macro
    key::Dict{String,Any}                         # name-keyed or file+hash-keyed
    expr_source::String
    captured_lhs::Any
    captured_rhs::Any
end

"""Tolerance: isapprox / abs(lhs-rhs)<ε / bare ≈ — verifies by isapprox(atol,rtol)."""
struct ToleranceShape <: CapturedAssertion
    idiom::Symbol
    key::Dict{String,Any}
    expr_source::String
    captured_lhs::Float64
    captured_rhs::Float64
    atol::Float64
    rtol::Float64
    stratum::Int                                   # 1,2,3; ε<=1e-14 => 1; <=1e-12 => 2; <=1e-10 => 3; else 3-by-intent
end

"""Directional: bare `<`, `<=`, `>`, `>=` — verifies by inequality preservation."""
struct DirectionalShape <: CapturedAssertion
    idiom::Symbol
    key::Dict{String,Any}
    expr_source::String
    captured_lhs::Any
    captured_op::Symbol                           # :<, :<=, :>, :>=
    captured_rhs::Any
end

"""Structural: isa / all / haskey / predicate-form — verifies by expression==true."""
struct StructuralShape <: CapturedAssertion
    idiom::Symbol
    key::Dict{String,Any}
    expr_source::String
    membership::Union{Nothing,NamedTuple{(:x, :collection, :orderable),Tuple{Any,Vector{Any},Bool}}}
end

"""Failing: condition returned false or threw — documents latent breakage (Q4)."""
struct FailingShape <: CapturedAssertion
    idiom::Symbol
    key::Dict{String,Any}
    expr_source::String
    reason::String                                # "returned-false" | "threw: <msg>"
    captured_operands::Dict{String,Any}
end

const CAPTURED = CapturedAssertion[]

# Dedup key set: "(idiom, key-fingerprint, values-fingerprint)".
# Same site + same values hit multiple times (e.g. in a loop) → one record.
# Different values for the same site → multiple records (Move 4 verifier
# needs the per-value-set coverage to catch iteration-order drift).
const CAPTURED_KEYS = Set{String}()

"""
Dedup fingerprint for a captured assertion. First component identifies
the site; second identifies the value-set. Identical fingerprints collapse
to one record.
"""
function _dedup_key(idiom::Symbol, key::Dict{String,Any}, values_repr::String)
    key_str = join(sort(["$(k)=$(v)" for (k, v) in key]), "|")
    return "$(idiom)::$(key_str)::$(values_repr)"
end

"""Try to push a captured assertion; skip if a dedup-identical one already exists."""
function _push_dedup!(a::CapturedAssertion, values_repr::String)
    dk = _dedup_key(a.idiom, a.key, values_repr)
    if dk in CAPTURED_KEYS
        return false
    end
    push!(CAPTURED_KEYS, dk)
    push!(CAPTURED, a)
    return true
end

# ═════════════════════════════════════════════════════════════════════════════
# AST classification — expression → (shape, operand-asts, tolerance-info)
# ═════════════════════════════════════════════════════════════════════════════

"""
Classify a boolean expression AST. Returns a NamedTuple with:
  - `shape`: :exact | :tolerance | :directional | :structural
  - `lhs`, `rhs`: the operand ASTs (or nothing for pure structural)
  - `op`: for directional, the comparison symbol; otherwise nothing
  - `atol`, `rtol`: for tolerance; default-isapprox resolves at capture time
  - `membership`: for Structural `x in S`, (x_ast, S_ast); otherwise nothing
"""
function classify(expr)
    # Tolerance: isapprox(lhs, rhs; atol=ε, rtol=ρ, ...) and kin.
    #
    # Julia's `isapprox` default-resolution rule (from Base):
    #   atol=0 (default), rtol=atol>0 ? 0 : sqrt(eps(T)) (default)
    # This means `isapprox(a, b, atol=0)` is NOT bit-exact — rtol defaults to
    # sqrt(eps) because atol is not > 0. Our classifier mirrors this by using
    # :_CONDITIONAL_DEFAULT for the non-specified slot when only one side is
    # provided, and the recorder resolves at runtime based on actual values.
    if @capture(expr, isapprox(lhs_, rhs_; atol=atol_, rtol=rtol_))
        return (shape=:tolerance, lhs=lhs, rhs=rhs, atol=atol, rtol=rtol, op=nothing, membership=nothing)
    end
    if @capture(expr, isapprox(lhs_, rhs_; atol=atol_))
        # atol given, rtol absent → rtol is conditional-default: 0 if atol>0, else sqrt(eps)
        return (shape=:tolerance, lhs=lhs, rhs=rhs, atol=atol, rtol=:_CONDITIONAL_DEFAULT, op=nothing, membership=nothing)
    end
    if @capture(expr, isapprox(lhs_, rhs_; rtol=rtol_))
        # rtol given, atol absent → atol=0 (Julia default when rtol specified)
        return (shape=:tolerance, lhs=lhs, rhs=rhs, atol=0.0, rtol=rtol, op=nothing, membership=nothing)
    end
    if @capture(expr, isapprox(lhs_, rhs_))
        # bare isapprox — atol=0, rtol=sqrt(eps) (Julia's default)
        return (shape=:tolerance, lhs=lhs, rhs=rhs, atol=0.0, rtol=:_DEFAULT, op=nothing, membership=nothing)
    end
    # abs(lhs - rhs) < ε  or  abs(lhs - rhs) <= ε
    if @capture(expr, abs(lhs_ - rhs_) < eps_) || @capture(expr, abs(lhs_ - rhs_) <= eps_)
        return (shape=:tolerance, lhs=lhs, rhs=rhs, atol=eps, rtol=0.0, op=nothing, membership=nothing)
    end
    # bare ≈ (Unicode, aliases isapprox with defaults)
    if @capture(expr, lhs_ ≈ rhs_)
        return (shape=:tolerance, lhs=lhs, rhs=rhs, atol=0.0, rtol=:_DEFAULT, op=nothing, membership=nothing)
    end
    # Exact: lhs == rhs
    if @capture(expr, lhs_ == rhs_)
        return (shape=:exact, lhs=lhs, rhs=rhs, atol=0.0, rtol=0.0, op=nothing, membership=nothing)
    end
    # Directional: bare <, <=, >, >=  (NOT wrapped in abs; not a tolerance form)
    for (opsym, opval) in [(:<, :<), (:<=, :<=), (:>, :>), (:>=, :>=)]
        if expr isa Expr && expr.head == :call && length(expr.args) == 3 && expr.args[1] === opval
            return (shape=:directional, lhs=expr.args[2], rhs=expr.args[3], atol=0.0, rtol=0.0, op=opsym, membership=nothing)
        end
    end
    # Membership: x in S
    if @capture(expr, x_ in S_)
        return (shape=:structural, lhs=nothing, rhs=nothing, atol=0.0, rtol=0.0, op=nothing,
                membership=(x_ast=x, S_ast=S))
    end
    # Fallback: Structural (isa, all, haskey, predicate-form expressions)
    return (shape=:structural, lhs=nothing, rhs=nothing, atol=0.0, rtol=0.0, op=nothing, membership=nothing)
end

"""Classify ε magnitude into stratum 1/2/3."""
function stratum_of(atol::Real, rtol::Real)
    ε = max(atol, rtol)  # effective tolerance magnitude
    ε == 0 && return 1
    ε <= 1e-14 && return 1
    ε <= 1e-12 && return 2
    ε <= 1e-10 && return 3
    return 3  # loose-by-intent, e.g. bare ≈ at ~1.5e-8
end

# ═════════════════════════════════════════════════════════════════════════════
# Runtime recorders — called from instrumented assertion sites
# ═════════════════════════════════════════════════════════════════════════════

"""
Record an `@assert expr` site. Called at runtime with pre-evaluated `cond`
and (optionally) pre-evaluated lhs/rhs. If `cond` is false or lhs/rhs
evaluation threw, record as Failing (Q4).
"""
function record_assert(key::Dict{String,Any}, expr_source::String, cond_thunk, lhs_thunk, rhs_thunk,
                       shape_info::NamedTuple)
    cond_result = nothing
    cond_error = nothing
    try
        cond_result = cond_thunk()
    catch e
        cond_error = sprint(showerror, e)
    end

    # Did the assertion pass?
    passed = cond_result === true

    if !passed
        operands = Dict{String,Any}()
        if lhs_thunk !== nothing
            try
                operands["lhs"] = lhs_thunk()
            catch e
                operands["lhs_error"] = sprint(showerror, e)
            end
        end
        if rhs_thunk !== nothing
            try
                operands["rhs"] = rhs_thunk()
            catch e
                operands["rhs_error"] = sprint(showerror, e)
            end
        end
        reason = cond_error !== nothing ? "threw: $cond_error" :
                 cond_result === false ? "returned-false" :
                 "returned-non-bool: $(cond_result)"
        _push_dedup!(FailingShape(:assert, key, expr_source, reason, operands),
                     "F|$reason|$(get(operands, "lhs", ""))|$(get(operands, "rhs", ""))")
        return nothing
    end

    # Passed — record by classified shape
    _record_passing(:assert, key, expr_source, lhs_thunk, rhs_thunk, shape_info)
    return nothing
end

"""Record a `check(name, cond, detail)` site — same semantics, name-keyed."""
function record_check(key::Dict{String,Any}, expr_source::String, cond_thunk, lhs_thunk, rhs_thunk,
                       shape_info::NamedTuple)
    cond_result = nothing
    cond_error = nothing
    try
        cond_result = cond_thunk()
    catch e
        cond_error = sprint(showerror, e)
    end

    passed = cond_result === true

    if !passed
        operands = Dict{String,Any}()
        lhs_thunk !== nothing && (try operands["lhs"] = lhs_thunk() catch e; operands["lhs_error"] = sprint(showerror, e) end)
        rhs_thunk !== nothing && (try operands["rhs"] = rhs_thunk() catch e; operands["rhs_error"] = sprint(showerror, e) end)
        reason = cond_error !== nothing ? "threw: $cond_error" :
                 cond_result === false ? "returned-false" :
                 "returned-non-bool: $(cond_result)"
        _push_dedup!(FailingShape(:check, key, expr_source, reason, operands),
                     "F|$reason|$(get(operands, "lhs", ""))|$(get(operands, "rhs", ""))")
        return nothing
    end

    _record_passing(:check, key, expr_source, lhs_thunk, rhs_thunk, shape_info)
    return nothing
end

"""Record a `@check(name, expr)` site — same semantics, name-keyed."""
function record_check_macro(key::Dict{String,Any}, expr_source::String, cond_thunk, lhs_thunk, rhs_thunk,
                            shape_info::NamedTuple)
    cond_result = nothing
    cond_error = nothing
    try
        cond_result = cond_thunk()
    catch e
        cond_error = sprint(showerror, e)
    end

    passed = cond_result === true

    if !passed
        operands = Dict{String,Any}()
        lhs_thunk !== nothing && (try operands["lhs"] = lhs_thunk() catch e; operands["lhs_error"] = sprint(showerror, e) end)
        rhs_thunk !== nothing && (try operands["rhs"] = rhs_thunk() catch e; operands["rhs_error"] = sprint(showerror, e) end)
        reason = cond_error !== nothing ? "threw: $cond_error" :
                 cond_result === false ? "returned-false" :
                 "returned-non-bool: $(cond_result)"
        _push_dedup!(FailingShape(:check_macro, key, expr_source, reason, operands),
                     "F|$reason|$(get(operands, "lhs", ""))|$(get(operands, "rhs", ""))")
        return nothing
    end

    _record_passing(:check_macro, key, expr_source, lhs_thunk, rhs_thunk, shape_info)
    return nothing
end

function _record_passing(idiom::Symbol, key::Dict{String,Any}, expr_source::String,
                         lhs_thunk, rhs_thunk, shape_info::NamedTuple)
    shape = shape_info.shape
    if shape === :exact
        lhs_val = lhs_thunk === nothing ? nothing : lhs_thunk()
        rhs_val = rhs_thunk === nothing ? nothing : rhs_thunk()
        _push_dedup!(ExactShape(idiom, key, expr_source, lhs_val, rhs_val),
                     "E|$(_values_repr(lhs_val))|$(_values_repr(rhs_val))")
    elseif shape === :tolerance
        lhs_raw = lhs_thunk === nothing ? NaN : lhs_thunk()
        rhs_raw = rhs_thunk === nothing ? NaN : rhs_thunk()
        lhs_val = _try_float64(lhs_raw)
        rhs_val = _try_float64(rhs_raw)
        # Non-scalar operands fall back to Structural (tolerance only applies to scalars)
        if lhs_val === nothing || rhs_val === nothing
            _push_dedup!(StructuralShape(idiom, key, expr_source, nothing),
                         "S|non-scalar-tolerance|$(_values_repr(lhs_raw))|$(_values_repr(rhs_raw))")
            return nothing
        end
        atol = shape_info.atol isa Real ? Float64(shape_info.atol) : 0.0
        if shape_info.rtol === :_DEFAULT
            # Bare isapprox / bare ≈ — rtol defaults to sqrt(eps)
            T = promote_type(typeof(lhs_val), typeof(rhs_val))
            rtol = Float64(Base.rtoldefault(T, T, atol))
        elseif shape_info.rtol === :_CONDITIONAL_DEFAULT
            # isapprox(a, b, atol=α) with rtol absent: Julia resolves rtol per
            # `atol > 0 ? 0 : sqrt(eps(T))`. Mirror that exactly.
            if atol > 0
                rtol = 0.0
            else
                T = promote_type(typeof(lhs_val), typeof(rhs_val))
                rtol = Float64(Base.rtoldefault(T, T, atol))
            end
        else
            rtol = shape_info.rtol isa Real ? Float64(shape_info.rtol) : 0.0
        end
        _push_dedup!(ToleranceShape(idiom, key, expr_source, lhs_val, rhs_val, atol, rtol, stratum_of(atol, rtol)),
                     "T|$(lhs_val)|$(rhs_val)|$(atol)|$(rtol)")
    elseif shape === :directional
        lhs_val = lhs_thunk === nothing ? nothing : lhs_thunk()
        rhs_val = rhs_thunk === nothing ? nothing : rhs_thunk()
        # Dedup on (site + operator) only — lhs/rhs values are forensic per §3
        # (the oracle is the inequality, not the values). Different iterations
        # of the same directional site collapse to one record regardless of
        # timing/state variance; the first hit's values go into the .jls for
        # drift diagnosis.
        _push_dedup!(DirectionalShape(idiom, key, expr_source, lhs_val, shape_info.op, rhs_val),
                     "D|$(shape_info.op)")
    elseif shape === :structural
        membership = nothing
        values_sig = "S|plain"
        if shape_info.membership !== nothing
            x_val = lhs_thunk === nothing ? nothing : lhs_thunk()
            S_val = rhs_thunk === nothing ? nothing : rhs_thunk()
            orderable = try; sort(collect(S_val)); true; catch; false; end
            S_canonical = try
                orderable ? sort(collect(S_val)) : collect(Set(S_val))
            catch
                Any[]  # uncollectable — record empty
            end
            membership = (x=x_val, collection=S_canonical, orderable=orderable)
            values_sig = "S|membership|$(_values_repr(x_val))|$(_values_repr(S_canonical))"
        end
        _push_dedup!(StructuralShape(idiom, key, expr_source, membership), values_sig)
    end
    return nothing
end

"""Safely convert to Float64; return nothing if the value isn't scalar-convertible."""
function _try_float64(v)
    try
        return Float64(v)
    catch
        return nothing
    end
end

"""Short repr of a value for dedup-key construction."""
_values_repr(v::Nothing) = "nothing"
_values_repr(v::AbstractFloat) = string(v)
_values_repr(v::Integer) = string(v)
_values_repr(v::Bool) = string(v)
_values_repr(v::AbstractString) = string(v)
_values_repr(v::Symbol) = string(v)
_values_repr(v::Tuple) = "(" * join(_values_repr.(v), ",") * ")"
_values_repr(v::AbstractVector) = "[" * join(_values_repr.(v), ",") * "]"
function _values_repr(v)
    # Fallback for complex objects: hash the stringified form.
    # Identical objects → identical hash; different objects of same type → different hash.
    try
        return "h:" * string(hash(string(v)))
    catch
        return "h:" * string(typeof(v))
    end
end

# ═════════════════════════════════════════════════════════════════════════════
# AST rewriting — inject capture calls into the parsed test-file expressions
# ═════════════════════════════════════════════════════════════════════════════

"""
Compute SHA-256 content hash of an expression AST, stringified.
Survives re-indentation but not variable renames.
"""
function expr_hash(expr)
    s = string(expr)
    return "sha256:" * bytes2hex(SHA.sha256(s))[1:16]
end

"""Source-text of an expression, best-effort."""
expr_str(e) = string(e)

"""
Build the thunk expressions the recorder needs: a no-arg closure that
re-evaluates the expression when called. Uses `()->(...)`.
"""
thunk(e) = e === nothing ? :nothing : :(() -> $e)

"""
Rewrite a single `@assert` call.
`@assert expr` → inject classification + recording.
"""
function rewrite_assert(expr, file::String, line::Int)
    info = classify(expr)
    shape_nt = :($(info))
    key = Dict{String,Any}("file" => file, "line" => line, "expr_hash" => expr_hash(expr))
    key_expr = :(Dict{String,Any}("file" => $file, "line" => $line, "expr_hash" => $(expr_hash(expr))))
    src = expr_str(expr)

    # For membership, lhs/rhs are the x/S ASTs; for others, use classified lhs/rhs
    lhs_ast = info.membership !== nothing ? info.membership.x_ast : info.lhs
    rhs_ast = info.membership !== nothing ? info.membership.S_ast : info.rhs

    quote
        $Main.CAPTURE_RUNTIME.record_assert(
            $key_expr,
            $src,
            $(thunk(expr)),
            $(thunk(lhs_ast)),
            $(thunk(rhs_ast)),
            $(shape_nt),
        )
    end
end

"""
Rewrite a `check(name, cond, detail)` call.
Match: `check(name, cond, detail)` or `check(name, cond)`
"""
function rewrite_check_call(call_expr, file::String, line::Int)
    # Extract name (args[2]) and cond (args[3]); optional detail (args[4])
    length(call_expr.args) < 3 && return call_expr
    name = call_expr.args[2]
    cond = call_expr.args[3]

    info = classify(cond)
    shape_nt = :($(info))
    name_str = name isa AbstractString ? string(name) : ""
    key_expr = :(Dict{String,Any}(
        "name" => $(name isa AbstractString ? name : :(string($name))),
        "file" => $file,
        "line" => $line,
    ))
    src = expr_str(cond)

    lhs_ast = info.membership !== nothing ? info.membership.x_ast : info.lhs
    rhs_ast = info.membership !== nothing ? info.membership.S_ast : info.rhs

    quote
        $Main.CAPTURE_RUNTIME.record_check(
            $key_expr,
            $src,
            $(thunk(cond)),
            $(thunk(lhs_ast)),
            $(thunk(rhs_ast)),
            $(shape_nt),
        )
    end
end

"""
Rewrite a `@check(name, expr)` macrocall.
"""
function rewrite_check_macro(macrocall, file::String, line::Int)
    # `@check "name" expr` parses as Expr(:macrocall, Symbol("@check"), LineNumberNode, "name", expr)
    args = filter(a -> !(a isa LineNumberNode), macrocall.args[2:end])
    length(args) < 2 && return macrocall
    name, expr = args[1], args[2]

    info = classify(expr)
    shape_nt = :($(info))
    key_expr = :(Dict{String,Any}(
        "name" => $(name isa AbstractString ? name : :(string($name))),
        "file" => $file,
        "line" => $line,
    ))
    src = expr_str(expr)

    lhs_ast = info.membership !== nothing ? info.membership.x_ast : info.lhs
    rhs_ast = info.membership !== nothing ? info.membership.S_ast : info.rhs

    quote
        $Main.CAPTURE_RUNTIME.record_check_macro(
            $key_expr,
            $src,
            $(thunk(expr)),
            $(thunk(lhs_ast)),
            $(thunk(rhs_ast)),
            $(shape_nt),
        )
    end
end

"""
Walk an AST recursively, rewriting:
  - `@assert expr` (Expr(:macrocall, Symbol("@assert"), ...))
  - `check(...)` calls (Expr(:call, :check, ...))
  - `@check name expr` (Expr(:macrocall, Symbol("@check"), ...))
into capture-recorder calls.

Preserves the original expression text for everything else.
"""
function walk_and_rewrite(expr, file::String, current_line::Ref{Int})
    if expr isa LineNumberNode
        current_line[] = expr.line
        return expr
    end
    if !(expr isa Expr)
        return expr
    end

    # Skip function / macro definition signatures — don't rewrite `check`
    # inside `function check(name, cond, detail="") ... end`.
    if expr.head === :function || expr.head === :macro
        # args = [signature, body...]
        sig = expr.args[1]
        body = [walk_and_rewrite(a, file, current_line) for a in expr.args[2:end]]
        return Expr(expr.head, sig, body...)
    end
    if expr.head === :(=) && length(expr.args) >= 2 && expr.args[1] isa Expr &&
       (expr.args[1].head === :call || expr.args[1].head === :where)
        # Short-form function definition: f(x) = body  OR  f(x) where {T} = body
        sig = expr.args[1]
        body = walk_and_rewrite(expr.args[2], file, current_line)
        return Expr(:(=), sig, body)
    end

    # @assert expr
    if expr.head == :macrocall && expr.args[1] isa Symbol && expr.args[1] === Symbol("@assert")
        # args: [Symbol("@assert"), LineNumberNode, actual_expr, optional_msg...]
        actual = filter(a -> !(a isa LineNumberNode) && !(a isa Symbol && a === Symbol("@assert")), expr.args)
        if !isempty(actual)
            ln = findfirst(a -> a isa LineNumberNode, expr.args)
            line = ln !== nothing ? expr.args[ln].line : current_line[]
            return rewrite_assert(actual[1], file, line)
        end
        return expr
    end
    # @check name expr
    if expr.head == :macrocall && expr.args[1] isa Symbol && expr.args[1] === Symbol("@check")
        ln = findfirst(a -> a isa LineNumberNode, expr.args)
        line = ln !== nothing ? expr.args[ln].line : current_line[]
        return rewrite_check_macro(expr, file, line)
    end
    # check(name, cond, detail)  — function call
    if expr.head == :call && length(expr.args) >= 3 && expr.args[1] === :check
        return rewrite_check_call(expr, file, current_line[])
    end

    # Recurse into children
    new_args = Any[]
    for a in expr.args
        push!(new_args, walk_and_rewrite(a, file, current_line))
    end
    return Expr(expr.head, new_args...)
end

# ═════════════════════════════════════════════════════════════════════════════
# Per-test-file capture orchestration
# ═════════════════════════════════════════════════════════════════════════════

"""
Capture one test file. Returns a per-file summary (assertion count, cascades).

The file is parsed (not executed) from disk. Assertion sites are rewritten
into capture-recorder calls. The rewritten AST is evaluated in a fresh
module, which provides the isolation needed to avoid cross-file state
pollution.
"""
function capture_one_file(path::String; recorder_module::Module)
    src = read(path, String)
    parsed = Meta.parseall(src, filename=path)  # file arg so @__DIR__ / @__FILE__ resolve to path's dir
    current_line = Ref(0)
    rewritten = walk_and_rewrite(parsed, path, current_line)

    # Capture runtime must be visible to the rewritten expressions via Main
    # (the rewriter injects $Main.CAPTURE_RUNTIME.record_*).
    before = length(CAPTURED)
    cascade_error = nothing
    try
        # Evaluate in a fresh module to isolate per-file state
        mod = Module(Symbol(replace(basename(path), ".jl" => "_captured_", "-" => "_")))
        Core.eval(mod, :(using Main: CAPTURE_RUNTIME))
        # Inject `include` so test files that do `include(joinpath(...))` work
        Core.eval(mod, :(
            include(path::AbstractString) = Base.include(@__MODULE__, path)
        ))
        # Seed the RNG at the start of each test file so that RNG-consuming
        # assertions (e.g. `draw()` inside a loop) produce the same captured
        # values across capture runs. This resolves open Q2's minimal-answer
        # trust-the-reseed case: captures are stable if each file starts with
        # the same RNG state. Files that call Random.seed! themselves override
        # this; files that don't inherit a pinned seed of 42.
        Random.seed!(42)
        Core.eval(mod, :(using Random; Random.seed!(42)))
        # Test files do `push!(LOAD_PATH, ...); using Credence` — that needs to work
        Core.eval(mod, rewritten)
    catch e
        cascade_error = sprint(showerror, e)
    end
    after = length(CAPTURED)

    return (
        file = path,
        captured_before = before,
        captured_after = after,
        new_captures = after - before,
        cascade = cascade_error,
    )
end

# ═════════════════════════════════════════════════════════════════════════════
# bad2_* corpus inventory — structural invariance channel
# ═════════════════════════════════════════════════════════════════════════════

"""
Walk `tools/credence-lint/corpus/<slug>/bad2_*.{jl,py}`, inventory each file
with its slug and expected detection. This is the structural-invariance
complement to the numerical/assertion capture.
"""
function inventory_bad2_corpus(corpus_root::String)
    entries = Dict{String,Any}[]
    isdir(corpus_root) || return entries

    for slug_dir in readdir(corpus_root; join=true)
        isdir(slug_dir) || continue
        slug = basename(slug_dir)
        for fname in readdir(slug_dir)
            if startswith(fname, "bad2_") && (endswith(fname, ".jl") || endswith(fname, ".py"))
                fpath = joinpath(slug_dir, fname)
                content = read(fpath, String)
                nlines = count(==('\n'), content) + 1
                push!(entries, Dict{String,Any}(
                    "file" => relpath(fpath, dirname(corpus_root) |> dirname |> dirname),  # relative to repo root
                    "slug" => slug,
                    "expected_detection" => "pass-two taint analysis",
                    "line_count" => nlines,
                    "content_hash" => "sha256:" * bytes2hex(SHA.sha256(content))[1:16],
                ))
            end
        end
    end
    return entries
end

# ═════════════════════════════════════════════════════════════════════════════
# Serialisation — .jls per shape + manifest.toml
# ═════════════════════════════════════════════════════════════════════════════

function serialise_outputs(output_dir::String, bad2_entries::Vector; repo_sha::String, julia_version::String)
    mkpath(output_dir)

    # Partition by shape
    exact_entries = filter(a -> a isa ExactShape, CAPTURED)
    tol1 = filter(a -> a isa ToleranceShape && a.stratum == 1, CAPTURED)
    tol2 = filter(a -> a isa ToleranceShape && a.stratum == 2, CAPTURED)
    tol3 = filter(a -> a isa ToleranceShape && a.stratum == 3, CAPTURED)
    dir_entries = filter(a -> a isa DirectionalShape, CAPTURED)
    struct_entries = filter(a -> a isa StructuralShape, CAPTURED)
    fail_entries = filter(a -> a isa FailingShape, CAPTURED)

    # Strata 1/2/3 .jls files bundle exact+tolerance by stratum
    strata_1 = vcat(exact_entries, tol1)
    strata_2 = tol2
    strata_3 = tol3

    Serialization.serialize(joinpath(output_dir, "strata-1.jls"), strata_1)
    Serialization.serialize(joinpath(output_dir, "strata-2.jls"), strata_2)
    Serialization.serialize(joinpath(output_dir, "strata-3.jls"), strata_3)
    Serialization.serialize(joinpath(output_dir, "directional.jls"), dir_entries)
    Serialization.serialize(joinpath(output_dir, "structural.jls"), struct_entries)
    Serialization.serialize(joinpath(output_dir, "failing.jls"), fail_entries)

    # Build the per-idiom sections, then sort for stable manifest output.
    # Keys used for sorting: (file, line, name, expr_source) — deterministic
    # across runs modulo value-drift in directional/tolerance shapes.
    function _stable_sort_key(row::Dict{String,Any})
        return (
            string(get(row, "file", "")),
            Int(get(row, "line", 0)),
            string(get(row, "name", "")),
            string(get(row, "expr_source", "")),
            string(get(row, "shape", "")),
        )
    end
    check_rows = sort([_manifest_row(a) for a in CAPTURED if a.idiom == :check]; by=_stable_sort_key)
    check_macro_rows = sort([_manifest_row(a) for a in CAPTURED if a.idiom == :check_macro]; by=_stable_sort_key)
    assert_rows = sort([_manifest_row(a) for a in CAPTURED if a.idiom == :assert]; by=_stable_sort_key)
    bad2_sorted = sort(bad2_entries; by=e -> (e["slug"], e["file"]))

    # Manifest — the TOML index of all captured entries
    manifest = Dict{String,Any}(
        "capture" => Dict{String,Any}(
            "repo_sha" => repo_sha,
            "julia_version" => julia_version,
            "cpu_string" => Sys.cpu_info()[1].model,
            "arch" => string(Sys.ARCH),
            "os" => string(Sys.KERNEL),
            "timestamp_utc" => string(Dates.now(Dates.UTC)),
            "counts" => Dict{String,Any}(
                "exact" => length(exact_entries),
                "tolerance_stratum_1" => length(tol1),
                "tolerance_stratum_2" => length(tol2),
                "tolerance_stratum_3" => length(tol3),
                "directional" => length(dir_entries),
                "structural" => length(struct_entries),
                "failing" => length(fail_entries),
                "total" => length(CAPTURED),
            ),
        ),
        "bad2_corpus" => Dict{String,Any}(
            "root" => "tools/credence-lint/corpus/",
            "entries" => bad2_sorted,
            "count" => length(bad2_sorted),
        ),
        # Detailed per-idiom listings (sorted for manifest stability)
        "check_assertions" => check_rows,
        "check_macro_assertions" => check_macro_rows,
        "assert_assertions" => assert_rows,
    )

    open(joinpath(output_dir, "manifest.toml"), "w") do io
        TOML.print(io, manifest)
    end
    return nothing
end

"""Extract a manifest row (TOML-safe) from a captured assertion."""
function _manifest_row(a::CapturedAssertion)
    base = Dict{String,Any}(
        "idiom" => string(a.idiom),
        "expr_source" => a.expr_source,
    )
    for (k, v) in a.key
        base[k] = v
    end
    if a isa ExactShape
        base["shape"] = "exact"
        base["captured_lhs"] = _toml_safe(a.captured_lhs)
        base["captured_rhs"] = _toml_safe(a.captured_rhs)
    elseif a isa ToleranceShape
        base["shape"] = "tolerance"
        base["captured_lhs"] = _toml_safe(a.captured_lhs)
        base["captured_rhs"] = _toml_safe(a.captured_rhs)
        base["atol"] = a.atol
        base["rtol"] = a.rtol
        base["stratum"] = a.stratum
    elseif a isa DirectionalShape
        base["shape"] = "directional"
        # Captured lhs/rhs are forensic per §3 — the oracle is the inequality,
        # not the values. Writing them to the manifest would destabilise the
        # double-run verify for timing-dependent captures (e.g. `@assert best_ms < 10.0`).
        # The .jls fixture retains the actual values for drift diagnosis.
        base["captured_lhs"] = "<forensic — see .jls>"
        base["captured_op"] = string(a.captured_op)
        base["captured_rhs"] = "<forensic — see .jls>"
    elseif a isa StructuralShape
        base["shape"] = "structural"
        if a.membership !== nothing
            base["membership_x"] = _toml_safe(a.membership.x)
            base["membership_S"] = _toml_safe.(a.membership.collection)
            base["membership_orderable"] = a.membership.orderable
        end
    elseif a isa FailingShape
        base["shape"] = "failing"
        base["reason"] = a.reason
        for (k, v) in a.captured_operands
            base["operand_$k"] = _toml_safe(v)
        end
    end
    return base
end

# (file-identifier extractor for partitioning; returns (key_kind, value))
function _key(a::CapturedAssertion)
    if haskey(a.key, "name")
        return ("name", a.key["name"])
    else
        return ("file", get(a.key, "file", ""))
    end
end

"""Convert arbitrary captured values to TOML-safe representation."""
_toml_safe(v::Nothing) = ""
_toml_safe(v::Bool) = v
_toml_safe(v::Int) = v
_toml_safe(v::Integer) = Int(v)
_toml_safe(v::AbstractFloat) = isfinite(v) ? Float64(v) : string(v)
_toml_safe(v::AbstractString) = String(v)
_toml_safe(v::Symbol) = string(v)
_toml_safe(v::AbstractVector) = [_toml_safe(x) for x in v]
_toml_safe(v::Tuple) = [_toml_safe(x) for x in v]
_toml_safe(v) = string(v)

# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

"""Run the full capture end-to-end and write to output_dir."""
function main(; output_dir::String, test_dir::String, corpus_root::String, repo_sha::String)
    # Create runtime module accessible as Main.CAPTURE_RUNTIME
    runtime_mod = Module(:CAPTURE_RUNTIME)
    Core.eval(runtime_mod, :(const record_assert = $record_assert))
    Core.eval(runtime_mod, :(const record_check = $record_check))
    Core.eval(runtime_mod, :(const record_check_macro = $record_check_macro))
    Core.eval(Main, :(const CAPTURE_RUNTIME = $runtime_mod))

    # Clear any prior state
    empty!(CAPTURED)
    empty!(CAPTURED_KEYS)

    # Walk test files
    test_files = sort(filter(f -> startswith(basename(f), "test_") && endswith(f, ".jl"),
                             readdir(test_dir; join=true)))
    println("Capturing from $(length(test_files)) test files...")
    for f in test_files
        summary = capture_one_file(f; recorder_module=runtime_mod)
        msg = "  $(basename(f)): captured $(summary.new_captures) sites"
        if summary.cascade !== nothing
            msg *= " | cascaded: $(first(summary.cascade, 80))..."
        end
        println(msg)
    end

    # bad2_* corpus inventory
    bad2 = inventory_bad2_corpus(corpus_root)
    println("bad2_* corpus: $(length(bad2)) files inventoried.")

    # Seed capture — record what the capture run saw in terms of seeds.
    # Per Q2 (conservative): test files each reseed Random as needed; we
    # trust their reseeds but record the seed state at capture-start.
    # (Per-site RNG-state capture is a Move-1-or-later concern if needed.)

    # Serialise
    serialise_outputs(output_dir, bad2; repo_sha=repo_sha, julia_version=string(VERSION))

    println("Done. Fixtures + manifest at $output_dir")
    summary = Dict(
        "total_captured" => length(CAPTURED),
        "exact" => count(a -> a isa ExactShape, CAPTURED),
        "tolerance" => count(a -> a isa ToleranceShape, CAPTURED),
        "directional" => count(a -> a isa DirectionalShape, CAPTURED),
        "structural" => count(a -> a isa StructuralShape, CAPTURED),
        "failing" => count(a -> a isa FailingShape, CAPTURED),
    )
    println("Summary: ", summary)
    return nothing
end

# ─── Command-line argument handling ──────────────────────────────────────

function parse_args(args::Vector{String})
    opts = Dict{String,Any}(
        "output" => "test/fixtures/posture-3-capture/",
        "test_dir" => "test",
        "corpus_root" => "tools/credence-lint/corpus",
        "verify" => false,
    )
    i = 1
    while i <= length(args)
        if args[i] == "--output"
            opts["output"] = args[i+1]; i += 2
        elseif args[i] == "--test-dir"
            opts["test_dir"] = args[i+1]; i += 2
        elseif args[i] == "--corpus-root"
            opts["corpus_root"] = args[i+1]; i += 2
        elseif args[i] == "--verify"
            opts["verify"] = true; i += 1
        else
            println("Unknown arg: $(args[i])")
            i += 1
        end
    end
    return opts
end

function repo_sha()
    # Pin to the branch-point, not the current HEAD. On the `de-finetti/complete`
    # branch, HEAD advances with each Move-N commit but the capture's
    # conceptual pin is the pre-branch master-tip (the SHA the test suite's
    # behaviour was captured against). `git merge-base HEAD master` yields
    # exactly that — and stays stable across commits on the same branch,
    # which means the fixture's recorded SHA doesn't churn when we amend.
    try
        mb = String(strip(read(`git merge-base HEAD master`, String)))
        isempty(mb) && return String(strip(read(`git rev-parse HEAD`, String)))
        return mb
    catch
        try
            return String(strip(read(`git rev-parse HEAD`, String)))
        catch
            return "unknown"
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_args(ARGS)
    # Resolve relative paths to absolute (so script works from any cwd)
    opts["output"] = abspath(opts["output"])
    opts["test_dir"] = abspath(opts["test_dir"])
    opts["corpus_root"] = abspath(opts["corpus_root"])
    if opts["verify"]
        println("Running double-verification...")
        mktempdir() do dir_a
            mktempdir() do dir_b
                println("=== run A ===")
                main(; output_dir=dir_a, test_dir=opts["test_dir"],
                     corpus_root=opts["corpus_root"], repo_sha=repo_sha())
                empty!(CAPTURED); empty!(CAPTURED_KEYS)
                println("=== run B ===")
                main(; output_dir=dir_b, test_dir=opts["test_dir"],
                     corpus_root=opts["corpus_root"], repo_sha=repo_sha())
                # Diff the two manifests (ignoring timestamp)
                a_toml = TOML.parsefile(joinpath(dir_a, "manifest.toml"))
                b_toml = TOML.parsefile(joinpath(dir_b, "manifest.toml"))
                delete!(a_toml["capture"], "timestamp_utc")
                delete!(b_toml["capture"], "timestamp_utc")
                if a_toml == b_toml
                    println("✓ Verified: manifests identical (modulo timestamp)")
                else
                    println("✗ DIVERGENT — manifests differ; protocol is under-specified")
                    println("--- counts A: ", a_toml["capture"]["counts"])
                    println("--- counts B: ", b_toml["capture"]["counts"])
                    # Per-section diff
                    for section in ("assert_assertions", "check_assertions", "check_macro_assertions")
                        as = haskey(a_toml, section) ? a_toml[section] : Dict[]
                        bs = haskey(b_toml, section) ? b_toml[section] : Dict[]
                        if length(as) != length(bs)
                            println("--- $section: A=$(length(as)), B=$(length(bs)) (delta $(length(bs) - length(as)))")
                        end
                    end
                    # Find entries present in one but not the other (by file+line+name)
                    a_keys = Set{String}()
                    b_keys = Set{String}()
                    for section in ("assert_assertions", "check_assertions", "check_macro_assertions")
                        as = haskey(a_toml, section) ? a_toml[section] : Dict[]
                        bs = haskey(b_toml, section) ? b_toml[section] : Dict[]
                        for e in as
                            push!(a_keys, "$(section)|$(get(e, "file", ""))|$(get(e, "line", ""))|$(get(e, "name", ""))|$(get(e, "shape", ""))|$(get(e, "captured_lhs", ""))|$(get(e, "captured_rhs", ""))")
                        end
                        for e in bs
                            push!(b_keys, "$(section)|$(get(e, "file", ""))|$(get(e, "line", ""))|$(get(e, "name", ""))|$(get(e, "shape", ""))|$(get(e, "captured_lhs", ""))|$(get(e, "captured_rhs", ""))")
                        end
                    end
                    a_only = setdiff(a_keys, b_keys)
                    b_only = setdiff(b_keys, a_keys)
                    println("--- In A but not B ($(length(a_only))):")
                    for k in first(collect(a_only), 5)
                        println("      $k")
                    end
                    println("--- In B but not A ($(length(b_only))):")
                    for k in first(collect(b_only), 5)
                        println("      $k")
                    end
                    exit(1)
                end
            end
        end
    else
        main(; output_dir=opts["output"], test_dir=opts["test_dir"],
             corpus_root=opts["corpus_root"], repo_sha=repo_sha())
    end
end

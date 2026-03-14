"""
    eval.jl — Evaluator for the Bayesian DSL.

Maps S-expressions to primitive operations.
The ONLY forms that exist:

    (belief h1 h2 h3 ...)          → Belief([h1, h2, h3, ...])
    (update <belief> <obs> <lik>)   → update(belief, obs, likelihood)
    (decide <belief> <actions> <u>) → decide(belief, actions, utility)
    (let <name> <expr> <body>)      → bind name to expr, evaluate body
    (define <name> <expr>)          → top-level binding (mutates env)
    (list e1 e2 ...)                → Vector of evaluated expressions
    (lambda (<params>) <body>)      → anonymous function
    (do e1 e2 ... en)               → evaluate all, return last
    (print <expr>)                  → print and return value
    (map fn lst)                    → apply fn to each element
    (fold fn lst)                   → reduce lst with fn (first elem is init)
    (first lst)                     → first element of a list
    (second lst)                    → second element of a list
    (nth lst idx)                   → 0-based index into a list
    (sample belief)                 → draw one hypothesis proportional to weights
    (weighted-sum belief fn)        → Σ_i w_i · fn(h_i)
    (max a b ...)                   → maximum of 2+ values

Arithmetic and comparison for utility/likelihood definitions:
    (+ a b ...) (- a b) (* a b ...) (/ a b)
    (> a b) (< a b) (= a b)
    (if <cond> <then> <else>)
    (log <x>) (exp <x>)

That's the whole language.
"""
module Eval

using ..Parse: SExpr, Atom, SList, parse_sexpr, parse_all
using ..Primitives: Belief, update, decide, weights, expected_utility, weighted_sum

export eval_dsl, run_dsl, load_dsl

# Environment: name → value bindings
const Env = Dict{Symbol, Any}

function default_env()
    Env(
        :pi => π,
        :e  => ℯ,
        # These exist alongside the special forms so that fold/map can
        # receive them as callable values: (fold + lst), (map f lst).
        # Special forms only trigger as head of a list expression;
        # as arguments, these resolve to Julia functions via the env.
        :+ => +,
        :* => *,
        :max => max,
        :min => min,
    )
end

# Evaluate a single S-expression in an environment
function eval_dsl(expr::Atom, env::Env)
    v = expr.value
    if v isa Symbol
        haskey(env, v) || error("unbound symbol: $v")
        return env[v]
    else
        return v  # number or bool literal
    end
end

function eval_dsl(expr::SList, env::Env)
    isempty(expr.items) && error("empty expression")

    head = expr.items[1]
    args = @view expr.items[2:end]

    # Special forms (don't evaluate all args immediately)
    if head isa Atom && head.value isa Symbol
        sym = head.value

        # ─── belief: the first primitive ───
        if sym == :belief
            hyps = [eval_dsl(a, env) for a in args]
            return Belief(hyps)
        end

        # ─── update: the second primitive ───
        if sym == :update
            length(args) == 3 || error("update requires exactly 3 arguments: belief, observation, likelihood")
            b = eval_dsl(args[1], env)
            obs = eval_dsl(args[2], env)
            lik = eval_dsl(args[3], env)
            return update(b, obs, lik)
        end

        # ─── decide: the third primitive ───
        if sym == :decide
            length(args) == 3 || error("decide requires exactly 3 arguments: belief, actions, utility")
            b = eval_dsl(args[1], env)
            acts = eval_dsl(args[2], env)
            u = eval_dsl(args[3], env)
            return decide(b, acts, u)
        end

        # ─── let: binding ───
        if sym == :let
            length(args) == 3 || error("let requires: name, value, body")
            name = args[1]
            name isa Atom && name.value isa Symbol || error("let name must be a symbol")
            val = eval_dsl(args[2], env)
            inner = copy(env)
            inner[name.value] = val
            return eval_dsl(args[3], inner)
        end

        # ─── define: top-level binding ───
        if sym == :define
            get(env, :__toplevel__, false) ||
                error("define is only permitted at top level. Use let for local bindings.")
            length(args) == 2 || error("define requires: name, value")
            name = args[1]
            name isa Atom && name.value isa Symbol || error("define name must be a symbol")
            val = eval_dsl(args[2], env)
            env[name.value] = val
            return val
        end

        # ─── lambda: function creation ───
        if sym == :lambda
            length(args) == 2 || error("lambda requires: (params) body")
            params_expr = args[1]
            params_expr isa SList || error("lambda params must be a list")
            param_names = Symbol[]
            for p in params_expr.items
                p isa Atom && p.value isa Symbol || error("lambda param must be a symbol")
                push!(param_names, p.value)
            end
            body = args[2]
            captured_env = copy(env)
            delete!(captured_env, :__toplevel__)
            return function(call_args...)
                length(call_args) == length(param_names) ||
                    error("expected $(length(param_names)) args, got $(length(call_args))")
                inner = copy(captured_env)
                for (name, val) in zip(param_names, call_args)
                    inner[name] = val
                end
                eval_dsl(body, inner)
            end
        end

        # ─── if: conditional ───
        if sym == :if
            length(args) == 3 || error("if requires: condition, then, else")
            cond = eval_dsl(args[1], env)
            return cond ? eval_dsl(args[2], env) : eval_dsl(args[3], env)
        end

        # ─── do: sequence ───
        if sym == :do
            result = nothing
            for a in args
                result = eval_dsl(a, env)
            end
            return result
        end

        # ─── list: construct a vector ───
        if sym == :list
            return [eval_dsl(a, env) for a in args]
        end

        # ─── print: inspect and return ───
        if sym == :print
            length(args) == 1 || error("print takes one argument")
            val = eval_dsl(args[1], env)
            println(val)
            return val
        end

        # ─── weights: inspect belief weights ───
        if sym == :weights
            length(args) == 1 || error("weights takes one argument")
            b = eval_dsl(args[1], env)
            return weights(b)
        end

        # ─── map: apply function to each element ───
        if sym == :map
            length(args) == 2 || error("map requires: function, list")
            f = eval_dsl(args[1], env)
            lst = eval_dsl(args[2], env)
            return [f(x) for x in lst]
        end

        # ─── fold: reduce list (first element is initial accumulator) ───
        if sym == :fold
            length(args) == 2 || error("fold requires: function, list")
            f = eval_dsl(args[1], env)
            lst = eval_dsl(args[2], env)
            length(lst) > 0 || error("fold requires non-empty list")
            acc = first(lst)
            for x in Iterators.drop(lst, 1)
                acc = f(acc, x)
            end
            return acc
        end

        # ─── first: first element of a list ───
        if sym == :first
            length(args) == 1 || error("first takes one argument")
            return first(eval_dsl(args[1], env))
        end

        # ─── second: second element of a list ───
        if sym == :second
            length(args) == 1 || error("second takes one argument")
            return eval_dsl(args[1], env)[2]
        end

        # ─── nth: 0-based index into a list ───
        if sym == :nth
            length(args) == 2 || error("nth requires: list, index")
            lst = eval_dsl(args[1], env)
            idx = eval_dsl(args[2], env)
            return lst[Int(idx) + 1]
        end

        # ─── sample: draw one hypothesis from a belief ───
        if sym == :sample
            length(args) == 1 || error("sample takes one argument")
            b = eval_dsl(args[1], env)
            w = weights(b)
            r = rand()
            cumw = 0.0
            for i in eachindex(w)
                cumw += w[i]
                if r < cumw
                    return b.hyps[i]
                end
            end
            return b.hyps[end]
        end

        # ─── max: variadic maximum ───
        if sym == :max
            length(args) >= 2 || error("max requires at least 2 arguments")
            return maximum(eval_dsl(a, env) for a in args)
        end

        # ─── weighted-sum: generic expectation under a belief ───
        if sym == Symbol("weighted-sum")
            length(args) == 2 || error("weighted-sum requires: belief, function")
            b = eval_dsl(args[1], env)
            f = eval_dsl(args[2], env)
            return weighted_sum(b, f)
        end

        # ─── arithmetic (+ and * are variadic, - and / remain binary) ───
        if sym == :+
            length(args) >= 2 || error("+ requires at least 2 arguments")
            return sum(eval_dsl(a, env) for a in args)
        end
        if sym == :*
            length(args) >= 2 || error("* requires at least 2 arguments")
            return prod(eval_dsl(a, env) for a in args)
        end
        if sym == :- return eval_dsl(args[1], env) - eval_dsl(args[2], env) end
        if sym == :/ return eval_dsl(args[1], env) / eval_dsl(args[2], env) end
        if sym == :log return log(eval_dsl(args[1], env)) end
        if sym == :exp return exp(eval_dsl(args[1], env)) end

        # ─── comparison ───
        if sym == :(=) return eval_dsl(args[1], env) == eval_dsl(args[2], env) end
        if sym == :> return eval_dsl(args[1], env) > eval_dsl(args[2], env) end
        if sym == :< return eval_dsl(args[1], env) < eval_dsl(args[2], env) end
    end

    # Function application: evaluate head, call with evaluated args
    func = eval_dsl(head, env)
    evaluated_args = [eval_dsl(a, env) for a in args]
    return func(evaluated_args...)
end

"""Run a DSL program (string of S-expressions)."""
function run_dsl(source::String; env::Env=default_env())
    env[:__toplevel__] = true
    # Load standard library
    stdlib_path = joinpath(@__DIR__, "stdlib.bdsl")
    if isfile(stdlib_path)
        for expr in parse_all(read(stdlib_path, String))
            eval_dsl(expr, env)
        end
    end
    # Run user code
    exprs = parse_all(source)
    result = nothing
    for expr in exprs
        result = eval_dsl(expr, env)
    end
    result
end

"""Load a DSL program and return the environment with all definitions."""
function load_dsl(source::String; env::Env=default_env())
    env[:__toplevel__] = true
    stdlib_path = joinpath(@__DIR__, "stdlib.bdsl")
    if isfile(stdlib_path)
        for expr in parse_all(read(stdlib_path, String))
            eval_dsl(expr, env)
        end
    end
    for expr in parse_all(source)
        eval_dsl(expr, env)
    end
    env
end

end # module Eval

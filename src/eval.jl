"""
    eval.jl — Evaluator for the Bayesian DSL.

Maps S-expressions to primitive operations.
The ONLY forms that exist:

    (belief h1 h2 h3 ...)          → Belief([h1, h2, h3, ...])
    (update <belief> <obs> <lik>)   → update(belief, obs, likelihood)
    (decide <belief> <actions> <u>) → decide(belief, actions, utility)
    (let <name> <expr> <body>)      → bind name to expr, evaluate body
    (list e1 e2 ...)                → Vector of evaluated expressions
    (lambda (<params>) <body>)      → anonymous function
    (do e1 e2 ... en)               → evaluate all, return last
    (print <expr>)                  → print and return value

Arithmetic and comparison for utility/likelihood definitions:
    (+ a b) (- a b) (* a b) (/ a b)
    (> a b) (< a b) (= a b)
    (if <cond> <then> <else>)
    (log <x>) (exp <x>)

That's the whole language.
"""
module Eval

using ..Parse: SExpr, Atom, SList, parse_sexpr, parse_all
using ..Primitives: Belief, update, decide, weights, expected_utility

export eval_dsl, run_dsl

# Environment: name → value bindings
const Env = Dict{Symbol, Any}

function default_env()
    Env(
        :pi => π,
        :e  => ℯ,
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

        # ─── eu: expected utility (derived) ───
        if sym == :eu
            length(args) == 3 || error("eu requires: belief, action, utility")
            b = eval_dsl(args[1], env)
            a = eval_dsl(args[2], env)
            u = eval_dsl(args[3], env)
            return expected_utility(b, a, u)
        end

        # ─── arithmetic ───
        if sym == :+ return eval_dsl(args[1], env) + eval_dsl(args[2], env) end
        if sym == :- return eval_dsl(args[1], env) - eval_dsl(args[2], env) end
        if sym == :* return eval_dsl(args[1], env) * eval_dsl(args[2], env) end
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
    exprs = parse_all(source)
    result = nothing
    for expr in exprs
        result = eval_dsl(expr, env)
    end
    result
end

end # module Eval

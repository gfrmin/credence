"""
    eval.jl — Evaluator for Credence

The frozen layer: three type constructors.
    (space ...)    → Space
    (measure ...)  → Measure
    (kernel ...)   → Kernel

Axiom-constrained functions exposed to the DSL environment:
    condition, expect, push, density

Everything else: lambda calculus + supporting forms.
"""
module Eval

using ..Parse: SExpr, Atom, SList, parse_sexpr, parse_all
using ..Ontology

export eval_dsl, run_dsl, load_dsl

const Env = Dict{Symbol, Any}

function default_env()
    Env(
        # Constants
        :pi => π,

        # Axiom-constrained functions, exposed as callable values.
        # The DSL calls these like any other function.
        # Their BEHAVIOUR is frozen (Bayesian inversion, integration,
        # composition, density). Their presence in the env is how the
        # DSL accesses them — not via special forms.
        :condition    => condition,
        Symbol("condition-on") => condition,   # Move 7 §2: event-form alias.
                                                # `(condition-on m e)` dispatches
                                                # to condition(m, e::Event); same
                                                # Julia generic, different DSL name
                                                # for consumers whose conditioning
                                                # object is an event rather than
                                                # a kernel-observation pair.
        :expect    => expect,
        :push      => push_measure,
        :density   => density,

        # Utility functions for fold/map
        :+ => +,
        :* => *,
        :- => -,
        :max => max,
        :min => min,
    )
end

# ── Atom evaluation ──

function eval_dsl(expr::Atom, env::Env)
    v = expr.value
    if v isa Symbol
        haskey(env, v) || error("unbound symbol: $v")
        return env[v]
    else
        return v
    end
end

# ── List evaluation ──

function eval_dsl(expr::SList, env::Env)
    isempty(expr.items) && error("empty expression")
    head = expr.items[1]
    args = @view expr.items[2:end]

    if head isa Atom && head.value isa Symbol
        sym = head.value

        # ═══════════════════════════════════════
        # FROZEN: Type constructors
        # ═══════════════════════════════════════

        if sym == :space
            return _eval_space(args, env)
        end

        if sym == :measure
            return _eval_measure(args, env)
        end

        if sym == :kernel
            return _eval_kernel(args, env)
        end

        # ═══════════════════════════════════════
        # Supporting forms (lambda calculus + sugar)
        # ═══════════════════════════════════════

        if sym == :let
            length(args) == 3 || error("let requires: name, value, body")
            name = args[1]
            name isa Atom && name.value isa Symbol || error("let name must be a symbol")
            val = eval_dsl(args[2], env)
            inner = copy(env)
            inner[name.value] = val
            return eval_dsl(args[3], inner)
        end

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
                for (n, v) in zip(param_names, call_args)
                    inner[n] = v
                end
                eval_dsl(body, inner)
            end
        end

        if sym == :if
            length(args) == 3 || error("if requires: condition, then, else")
            cond = eval_dsl(args[1], env)
            return cond ? eval_dsl(args[2], env) : eval_dsl(args[3], env)
        end

        if sym == :do
            result = nothing
            for a in args; result = eval_dsl(a, env); end
            return result
        end

        if sym == :list
            return [eval_dsl(a, env) for a in args]
        end

        if sym == :print
            val = eval_dsl(args[1], env)
            println(val)
            return val
        end

        # ── Collection operations ──

        if sym == :map
            f = eval_dsl(args[1], env)
            lst = eval_dsl(args[2], env)
            return [f(x) for x in lst]
        end

        if sym == :fold
            f = eval_dsl(args[1], env)
            lst = eval_dsl(args[2], env)
            length(lst) > 0 || error("fold requires non-empty list")
            acc = first(lst)
            for x in Iterators.drop(lst, 1); acc = f(acc, x); end
            return acc
        end

        if sym == Symbol("fold-init")
            f = eval_dsl(args[1], env)
            init = eval_dsl(args[2], env)
            lst = eval_dsl(args[3], env)
            acc = init
            for x in lst; acc = f(acc, x); end
            return acc
        end

        if sym == :first;  return first(eval_dsl(args[1], env)); end
        if sym == :second; return eval_dsl(args[1], env)[2]; end
        if sym == :nth
            lst = eval_dsl(args[1], env)
            idx = eval_dsl(args[2], env)
            return lst[Int(idx) + 1]
        end
        if sym == :length; return length(eval_dsl(args[1], env)); end
        if sym == :range
            n = Int(eval_dsl(args[1], env))
            return collect(0:n-1)
        end

        # ── Query operations on types ──

        if sym == :weights
            return weights(eval_dsl(args[1], env))
        end
        if sym == :mean
            return mean(eval_dsl(args[1], env))
        end
        if sym == :variance
            return variance(eval_dsl(args[1], env))
        end
        if sym == :support
            return support(eval_dsl(args[1], env))
        end
        if sym == Symbol("product-measure")
            # Single argument: may be a list of measures or a single measure.
            if length(args) == 1
                val = eval_dsl(args[1], env)
                if val isa AbstractVector
                    all(v -> v isa Measure, val) ||
                        error("(product-measure …): vector argument must contain only Measures, " *
                              "got element types $(unique(typeof.(val)))")
                    return ProductMeasure(Measure[v for v in val])
                elseif val isa Measure
                    return ProductMeasure(Measure[val])
                else
                    error("(product-measure …): single argument must be a Measure or " *
                          "Vector{Measure}, got $(typeof(val))")
                end
            end
            factors = Measure[eval_dsl(a, env) for a in args]
            return ProductMeasure(factors)
        end
        if sym == :factor
            m = eval_dsl(args[1], env)
            i = Int(eval_dsl(args[2], env))
            m isa ProductMeasure || error("factor requires a ProductMeasure, got $(typeof(m))")
            return factor(m, i + 1)  # 0-based DSL → 1-based Julia
        end
        if sym == Symbol("replace-factor")
            m = eval_dsl(args[1], env)
            i = Int(eval_dsl(args[2], env))
            new_factor = eval_dsl(args[3], env)
            m isa ProductMeasure || error("replace-factor requires a ProductMeasure, got $(typeof(m))")
            new_factor isa Measure || error("replace-factor requires a Measure as new factor, got $(typeof(new_factor))")
            return replace_factor(m, i + 1, new_factor)
        end
        if sym == Symbol("n-factors")
            m = eval_dsl(args[1], env)
            m isa ProductMeasure || error("n-factors requires a ProductMeasure, got $(typeof(m))")
            return length(m.factors)
        end
        if sym == Symbol("mixture-measure")
            components = Measure[]
            log_wts = Float64[]
            for a in args
                a isa SList || error("mixture-measure entries must be (weight measure)")
                length(a.items) == 2 || error("each mixture entry must be (weight measure)")
                w = Float64(eval_dsl(a.items[1], env))
                m = eval_dsl(a.items[2], env)
                m isa Measure || error("mixture-measure entries must be measures")
                push!(components, m)
                push!(log_wts, log(max(w, 1e-300)))
            end
            return MixtureMeasure(components[1].space, components, log_wts)
        end
        if sym == Symbol("kernel-target")
            return kernel_target(eval_dsl(args[1], env))
        end

        # ── Arithmetic ──

        if sym == :+; return sum(eval_dsl(a, env) for a in args); end
        if sym == :*; return prod(eval_dsl(a, env) for a in args); end
        if sym == :-; return eval_dsl(args[1], env) - eval_dsl(args[2], env); end
        if sym == :/; return eval_dsl(args[1], env) / eval_dsl(args[2], env); end
        if sym == :log; return log(eval_dsl(args[1], env)); end
        if sym == :exp; return exp(eval_dsl(args[1], env)); end
        if sym == :max; return maximum(eval_dsl(a, env) for a in args); end

        # ── Comparison ──

        if sym == :(=); return eval_dsl(args[1], env) == eval_dsl(args[2], env); end
        if sym == :>;   return eval_dsl(args[1], env) >  eval_dsl(args[2], env); end
        if sym == :<;   return eval_dsl(args[1], env) <  eval_dsl(args[2], env); end
    end

    # ── Function application ──
    func = eval_dsl(head, env)
    evaluated_args = [eval_dsl(a, env) for a in args]
    return func(evaluated_args...)
end

# ═══════════════════════════════════════
# Type constructors
# ═══════════════════════════════════════

function _eval_space(args, env)
    tag = args[1]
    tag isa Atom && tag.value isa Symbol || error("space type must be a symbol")

    if tag.value == :finite
        vals = Any[]
        for a in @view args[2:end]
            if a isa Atom
                push!(vals, a.value)
            else
                push!(vals, eval_dsl(a, env))
            end
        end
        return Finite(vals)

    elseif tag.value == :interval
        lo = eval_dsl(args[2], env)
        hi = eval_dsl(args[3], env)
        return Interval(Float64(lo), Float64(hi))

    elseif tag.value == :product
        factors = [eval_dsl(a, env) for a in @view args[2:end]]
        return ProductSpace(factors)

    elseif tag.value == :simplex
        k = Int(eval_dsl(args[2], env))
        return Simplex(k)

    elseif tag.value == :euclidean
        n = Int(eval_dsl(args[2], env))
        return Euclidean(n)

    elseif tag.value == Symbol("positive-reals")
        return PositiveReals()

    else
        error("unknown space type: $(tag.value)")
    end
end

function _eval_measure(args, env)
    space = eval_dsl(args[1], env)
    spec = args[2]
    spec isa Atom && spec.value isa Symbol || error("measure spec must be a symbol")
    return _make_measure(space, spec.value, @view(args[3:end]), env)
end

function _make_measure(space::Finite{T}, spec::Symbol, args, env) where T
    if spec == :uniform
        return CategoricalMeasure(space)
    elseif spec == :categorical
        ws = [Float64(eval_dsl(a, env)) for a in args]
        logw = [log(max(w, 1e-300)) for w in ws]
        return CategoricalMeasure(space, logw)
    else
        error("unknown measure for Finite space: $spec")
    end
end

function _make_measure(space::Interval, spec::Symbol, args, env)
    if spec == :beta
        α = Float64(eval_dsl(args[1], env))
        β = Float64(eval_dsl(args[2], env))
        return BetaMeasure(space, α, β)
    elseif spec == :uniform
        if space.lo == 0.0 && space.hi == 1.0
            return BetaMeasure(space, 1.0, 1.0)
        else
            error("uniform on Interval only defined for [0,1] (Beta(1,1)). Use Euclidean for Gaussian.")
        end
    else
        error("unknown measure for Interval space: $spec")
    end
end

function _make_measure(space::PositiveReals, spec::Symbol, args, env)
    if spec == :gamma
        α = Float64(eval_dsl(args[1], env))
        β = Float64(eval_dsl(args[2], env))
        return GammaMeasure(space, α, β)
    elseif spec == :exponential
        rate = Float64(eval_dsl(args[1], env))
        return ExponentialMeasure(rate)
    else
        error("unknown measure for PositiveReals space: $spec")
    end
end

function _make_measure(space::Euclidean, spec::Symbol, args, env)
    if spec == :gaussian
        μ = Float64(eval_dsl(args[1], env))
        σ = Float64(eval_dsl(args[2], env))
        return GaussianMeasure(space, μ, σ)
    elseif spec == :uniform
        error("uniform measure on Euclidean space is not well-defined (use :gaussian)")
    else
        error("unknown measure for Euclidean space: $spec")
    end
end

function _make_measure(space::Simplex, spec::Symbol, args, env)
    if spec == :dirichlet
        categories = eval_dsl(args[1], env)
        categories isa Finite || error(":dirichlet requires a Finite space for categories")
        alphas = [Float64(eval_dsl(a, env)) for a in @view(args[2:end])]
        return DirichletMeasure(space, categories, alphas)
    else
        error("unknown measure for Simplex space: $spec")
    end
end

function _eval_kernel(args, env)
    length(args) >= 3 || error("kernel requires: source-space, target-space, generator")
    source = eval_dsl(args[1], env)
    target = eval_dsl(args[2], env)
    gen = eval_dsl(args[3], env)

    # Parse optional trailing :family <value> pair. PushOnly by default — a
    # kernel used only for push is family-agnostic; one used to condition a
    # TaggedBetaMeasure must declare its leaf family at construction (see
    # src/ontology.jl LikelihoodFamily). Router families (FiringByTag,
    # DispatchByComponent) carry closures and stay Julia-only.
    family = PushOnly()
    family_set = false
    i = 4
    while i <= length(args)
        kw = args[i]
        (kw isa Atom && kw.value isa Symbol) ||
            error("kernel: expected keyword symbol at position $i, got $(args[i])")
        if kw.value == :family
            family_set && error("kernel: duplicate :family keyword")
            i + 1 <= length(args) ||
                error(":family requires a value (one of: bernoulli, flat)")
            fam_atom = args[i+1]
            (fam_atom isa Atom && fam_atom.value isa Symbol) ||
                error(":family value must be a symbol (one of: bernoulli, flat), got $(args[i+1])")
            family = _parse_family_keyword(fam_atom.value)
            family_set = true
            i += 2
        else
            error("kernel: unknown keyword :$(kw.value) (accepted: :family)")
        end
    end

    log_dens = _make_log_density(target, gen)
    Kernel(source, target, gen, log_dens; likelihood_family = family)
end

_parse_family_keyword(s::Symbol) =
    s === :bernoulli ? BetaBernoulli() :
    s === :flat      ? Flat() :
    error(":family value :$s unknown (one of: bernoulli, flat)")

function _make_log_density(target::Finite, gen::Function)
    # Generator returns a distribution spec: either a function (observation → log-probability)
    # or a Measure (the kernel maps h → Measure, and we derive log-density from it).
    function(h, o)
        dist_spec = gen(h)
        if dist_spec isa Function
            return dist_spec(o)
        elseif dist_spec isa Measure
            return log_density_at(dist_spec, o)
        else
            error("kernel generator must return a function or Measure for finite target spaces")
        end
    end
end

function _make_log_density(target::Interval, gen::Function)
    function(h, o)
        dist_spec = gen(h)
        if dist_spec isa Function
            return dist_spec(o)
        elseif dist_spec isa Measure
            return log_density_at(dist_spec, o)
        else
            error("kernel generator must return a function or Measure for interval target spaces")
        end
    end
end

function _make_log_density(target::Euclidean, gen::Function)
    function(h, o)
        dist_spec = gen(h)
        if dist_spec isa Function
            return dist_spec(o)
        elseif dist_spec isa Measure
            return log_density_at(dist_spec, o)
        else
            error("kernel generator must return a function or Measure for Euclidean target spaces")
        end
    end
end

# ═══════════════════════════════════════
# Entry points
# ═══════════════════════════════════════

function run_dsl(source::String; env::Env=default_env())
    env[:__toplevel__] = true
    stdlib_path = joinpath(@__DIR__, "stdlib.bdsl")
    if isfile(stdlib_path)
        for expr in parse_all(read(stdlib_path, String))
            eval_dsl(expr, env)
        end
    end
    exprs = parse_all(source)
    result = nothing
    for expr in exprs; result = eval_dsl(expr, env); end
    result
end

function load_dsl(source::String; env::Env=default_env())
    env[:__toplevel__] = true
    stdlib_path = joinpath(@__DIR__, "stdlib.bdsl")
    if isfile(stdlib_path)
        for expr in parse_all(read(stdlib_path, String))
            eval_dsl(expr, env)
        end
    end
    for expr in parse_all(source); eval_dsl(expr, env); end
    env
end

end # module Eval

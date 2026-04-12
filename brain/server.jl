#!/usr/bin/env julia
"""
    brain/server.jl — Credence brain process

JSON-RPC 2.0 over stdio. Reads newline-delimited JSON from stdin,
writes responses to stdout, logs to stderr.

The brain holds stateful objects (measures, agent states, DSL environments)
in a registry keyed by opaque string IDs. The host never sees the internal
representation — only the ID.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: expect, condition, push_measure, draw, density
using Credence: weights, mean, variance, prune, truncate, logsumexp
using Credence: CategoricalMeasure, BetaMeasure, TaggedBetaMeasure
using Credence: GaussianMeasure, GammaMeasure, DirichletMeasure
using Credence: NormalGammaMeasure, ProductMeasure, MixtureMeasure
using Credence: Finite, Interval, ProductSpace, Euclidean, PositiveReals, Simplex
using Credence: Kernel, FactorSelector, support
using Credence: Functional, Identity, Projection, NestedProjection, Tabular, LinearCombination, Composition, OpaqueClosure
using Credence: factor, replace_factor
using Credence: AgentState, sync_prune!, sync_truncate!
using Credence: Grammar, Program, CompiledKernel, ProductionRule
using Credence: enumerate_programs, compile_kernel
using Credence: analyse_posterior_subtrees, perturb_grammar
using Credence: aggregate_grammar_weights, top_k_grammar_ids, add_programs_to_state!
using Credence: next_grammar_id, reset_grammar_counter!
using Credence: show_expr
using Credence: log_density_at

using JSON3
using Serialization

# ═══════════════════════════════════════
# State registry
# ═══════════════════════════════════════

const STATE_REGISTRY = Dict{String, Any}()
const DSL_ENVS = Dict{String, Dict{Symbol, Any}}()
let counter = Ref(0)
    global function next_state_id()
        counter[] += 1
        "s_$(counter[])"
    end
end

function register_state(obj)
    id = next_state_id()
    STATE_REGISTRY[id] = obj
    id
end

function get_state(id::String)
    haskey(STATE_REGISTRY, id) || throw(StateNotFound(id))
    STATE_REGISTRY[id]
end

# ═══════════════════════════════════════
# Error types
# ═══════════════════════════════════════

struct StateNotFound <: Exception
    id::String
end

struct InferenceError <: Exception
    msg::String
end

struct DSLError <: Exception
    msg::String
end

# ═══════════════════════════════════════
# JSON-RPC helpers
# ═══════════════════════════════════════

function make_response(id, result)
    Dict("jsonrpc" => "2.0", "id" => id, "result" => result)
end

function make_error(id, code::Int, message::String)
    Dict("jsonrpc" => "2.0", "id" => id,
         "error" => Dict("code" => code, "message" => message))
end

function send_response(resp)
    println(stdout, JSON3.write(resp))
    flush(stdout)
end

function log_msg(msg)
    println(stderr, msg)
    flush(stderr)
end

# ═══════════════════════════════════════
# Space construction from spec
# ═══════════════════════════════════════

function build_space(spec)
    t = spec["type"]
    if t == "finite"
        vals = spec["values"]
        Finite(collect(Float64, vals))
    elseif t == "interval"
        Interval(Float64(spec["lo"]), Float64(spec["hi"]))
    elseif t == "product"
        ProductSpace(Space[build_space(f) for f in spec["factors"]])
    elseif t == "euclidean"
        Euclidean(Int(spec["dim"]))
    elseif t == "positive_reals"
        PositiveReals()
    elseif t == "simplex"
        Simplex(Int(spec["k"]))
    else
        error("unknown space type: $t")
    end
end

# ═══════════════════════════════════════
# Measure construction from spec
# ═══════════════════════════════════════

function build_measure(spec)
    t = spec["type"]
    if t == "categorical"
        space = build_space(spec["space"])
        if haskey(spec, "log_weights")
            CategoricalMeasure(space, collect(Float64, spec["log_weights"]))
        else
            CategoricalMeasure(space)
        end
    elseif t == "beta"
        BetaMeasure(Float64(spec["alpha"]), Float64(spec["beta"]))
    elseif t == "tagged_beta"
        TaggedBetaMeasure(Interval(0.0, 1.0), Int(spec["tag"]),
                          BetaMeasure(Float64(spec["alpha"]), Float64(spec["beta"])))
    elseif t == "gaussian"
        GaussianMeasure(Euclidean(1), Float64(spec["mu"]), Float64(spec["sigma"]))
    elseif t == "gamma"
        GammaMeasure(Float64(spec["alpha"]), Float64(spec["beta"]))
    elseif t == "dirichlet"
        alpha = collect(Float64, spec["alpha"])
        k = length(alpha)
        DirichletMeasure(Simplex(k), Finite(collect(Float64, 0:k-1)), alpha)
    elseif t == "normal_gamma"
        NormalGammaMeasure(Float64(spec["kappa"]), Float64(spec["mu"]),
                           Float64(spec["alpha"]), Float64(spec["beta"]))
    elseif t == "product"
        factors = Measure[build_measure(f) for f in spec["factors"]]
        ProductMeasure(factors)
    elseif t == "mixture"
        components = Measure[build_measure(c) for c in spec["components"]]
        lw = collect(Float64, spec["log_weights"])
        MixtureMeasure(components, lw)
    else
        error("unknown measure type: $t")
    end
end

# ═══════════════════════════════════════
# Kernel construction from spec
# ═══════════════════════════════════════

function build_kernel(spec, state_id::Union{String, Nothing}=nothing)
    t = spec["type"]
    if t == "bernoulli"
        # Beta-Bernoulli conjugate kernel
        source = Interval(0.0, 1.0)
        target = Finite([0.0, 1.0])
        Kernel(source, target,
            θ -> error("generate not used"),
            (θ, o) -> o == 1.0 ? log(max(θ, 1e-300)) : log(max(1.0 - θ, 1e-300)))

    elseif t == "gaussian_known_var"
        variance = Float64(spec["variance"])
        source = Euclidean(1)
        target = Euclidean(1)
        Kernel(source, target,
            μ -> error("generate not used"),
            (μ, o) -> -0.5 * (o - μ)^2 / variance,
            nothing,
            Dict{Symbol, Any}(:sigma_obs => sqrt(variance)))

    elseif t == "quality"
        # (θ, k) → Beta(θk, (1-θ)k) continuous quality observation
        source = ProductSpace(Space[Interval(0.0, 1.0), PositiveReals()])
        target = Interval(0.0, 1.0)
        Kernel(source, target,
            h -> error("generate not used"),
            (h, o) -> begin
                θ, k = h[1], h[2]
                a = max(θ * k, 1e-6)
                b = max((1.0 - θ) * k, 1e-6)
                (a - 1.0) * log(max(o, 1e-300)) + (b - 1.0) * log(max(1.0 - o, 1e-300))
            end)

    elseif t == "program_observation"
        # Per-component CompiledKernel dispatch
        state_id !== nothing || error("program_observation kernel requires a state_id context")
        state = get_state(state_id)::AgentState
        features = Dict{Symbol, Float64}(Symbol(k) => Float64(v)
                                          for (k, v) in spec["features"])
        true_label = Symbol(spec["true_label"])
        temporal_state = Dict{Symbol, Any}()

        recommendation_cache = Dict{Int, Symbol}()
        correct_cache = Dict{Int, Bool}()
        obs_space = Finite([0.0, 1.0])

        Kernel(Interval(0.0, 1.0), obs_space,
            _ -> error("generate not used"),
            (m_or_θ, obs) -> begin
                if m_or_θ isa TaggedBetaMeasure
                    tag = m_or_θ.tag
                    recommended = get!(recommendation_cache, tag) do
                        ck = state.compiled_kernels[tag]
                        ck.evaluate(features, temporal_state)
                    end
                    correct = recommended == true_label
                    correct_cache[tag] = correct
                    p = mean(m_or_θ.beta)
                    correct ? log(max(p, 1e-300)) : log(max(1.0 - p, 1e-300))
                else
                    obs == 1.0 ? log(max(m_or_θ, 1e-300)) : log(max(1.0 - m_or_θ, 1e-300))
                end
            end,
            nothing,
            Dict{Symbol, Any}(:correct_cache => correct_cache))

    elseif t == "tabular_log_density"
        source_vals = collect(Float64, spec["source_vals"])
        target_vals = collect(Float64, spec["target_vals"])
        densities = [collect(Float64, row) for row in spec["densities"]]
        source = Finite(source_vals)
        target = Finite(target_vals)
        Kernel(source, target,
            _ -> error("generate not used"),
            (h, o) -> begin
                si = findfirst(==(h), source_vals)
                ti = findfirst(==(o), target_vals)
                (si !== nothing && ti !== nothing) ? densities[si][ti] : -Inf
            end)

    elseif t == "dsl"
        env_id = spec["env_id"]
        fn_name = spec["fn_name"]
        args = get(spec, "args", [])
        env = get(DSL_ENVS, env_id, nothing)
        env !== nothing || error("DSL environment not found: $env_id")
        fn = env[Symbol(fn_name)]
        resolved_args = [resolve_arg(a) for a in args]
        fn(resolved_args...)

    else
        error("unknown kernel type: $t")
    end
end

# ═══════════════════════════════════════
# Function specs (for expect)
# ═══════════════════════════════════════

# Builds a Functional from a protocol function spec. Recursive: nested
# functionals (e.g. terms in a LinearCombination) are built by recursing
# into sub-specs. Type dispatch in ontology's expect() selects the
# computation based on the Functional subtype.
function build_function(spec)::Functional
    t = spec["type"]
    if t == "identity"
        Identity()
    elseif t == "projection" || t == "project"  # "project" kept for compat
        Projection(Int(spec["index"]) + 1)  # 0-based → 1-based
    elseif t == "nested_projection"
        NestedProjection(Int[Int(i) + 1 for i in spec["indices"]])
    elseif t == "tabular"
        Tabular(collect(Float64, spec["values"]))
    elseif t == "linear_combination"
        terms = Tuple{Float64, Functional}[
            (Float64(pair[1]), build_function(pair[2]))
            for pair in spec["terms"]
        ]
        offset = Float64(get(spec, "offset", 0.0))
        LinearCombination(terms, offset)
    elseif t == "opaque_bdsl" || t == "bdsl"
        env_id = spec["env_id"]
        env = get(DSL_ENVS, env_id, nothing)
        env !== nothing || error("DSL environment not found: $env_id")
        expr_str = spec["expr"]
        parsed = Credence.parse_all(expr_str)
        length(parsed) == 1 || error("expected single expression, got $(length(parsed))")
        f = Credence.Eval.eval_dsl(parsed[1], env)
        f isa Function || error("opaque_bdsl must evaluate to a function, got $(typeof(f))")
        OpaqueClosure(f)
    else
        error("unknown function type: $t")
    end
end

# ═══════════════════════════════════════
# Preference specs (for optimise/value)
# ═══════════════════════════════════════

function build_preference(spec)
    t = spec["type"]
    if t == "tabular_2d"
        matrix = [collect(Float64, row) for row in spec["matrix"]]
        (h, a) -> begin
            hi = round(Int, h) + 1
            ai = round(Int, a) + 1
            matrix[hi][ai]
        end
    elseif t == "bdsl"
        env_id = spec["env_id"]
        env = get(DSL_ENVS, env_id, nothing)
        env !== nothing || error("DSL environment not found: $env_id")
        expr_str = spec["expr"]
        parsed = Credence.parse_all(expr_str)
        length(parsed) == 1 || error("expected single expression")
        Credence.Eval.eval_dsl(parsed[1], env)
    else
        error("unknown preference type: $t")
    end
end

# ═══════════════════════════════════════
# Grammar construction from spec
# ═══════════════════════════════════════

function build_grammar(spec)
    features = Set{Symbol}(Symbol(f) for f in spec["features"])
    rules_spec = spec["rules"]
    rules = ProductionRule[]
    for (lhs, alternatives) in rules_spec
        for rhs in alternatives
            push!(rules, ProductionRule(string(lhs), string(rhs)))
        end
    end
    gid = next_grammar_id()
    Grammar(gid, rules, features)
end

# ═══════════════════════════════════════
# Argument resolution (for call_dsl)
# ═══════════════════════════════════════

function resolve_arg(arg)
    if arg isa Dict || arg isa JSON3.Object
        if haskey(arg, "ref")
            return get_state(string(arg["ref"]))
        end
        # Otherwise it's a nested dict — convert to Julia dict
        return Dict{String, Any}(string(k) => resolve_arg(v) for (k, v) in pairs(arg))
    elseif arg isa AbstractVector
        # Could be a plain list of numbers → Julia Float64 vector
        if all(x -> x isa Number, arg)
            return collect(Float64, arg)
        end
        return [resolve_arg(x) for x in arg]
    else
        return arg
    end
end

# ═══════════════════════════════════════
# Serialization helpers
# ═══════════════════════════════════════

function serialize_value(val)
    if val isa Number
        Dict("value" => val)
    elseif val isa Bool
        Dict("value" => val)
    elseif val isa Symbol
        Dict("value" => string(val))
    elseif val isa AbstractVector && all(x -> x isa Number, val)
        Dict("value" => collect(Float64, val))
    elseif val isa Tuple && all(x -> x isa Number, val)
        Dict("value" => collect(val))
    else
        # Opaque Julia object (including nested lists, measures, etc.)
        # Wrap as a single state — the host passes it back via {"ref": id}
        Dict("state_id" => register_state(val))
    end
end

# ═══════════════════════════════════════
# Method dispatch
# ═══════════════════════════════════════

function handle_request(method::String, params, id)
    if method == "initialize"
        handle_initialize(params)
    elseif method == "shutdown"
        Dict("result" => "ok")
    elseif method == "create_state"
        handle_create_state(params)
    elseif method == "destroy_state"
        handle_destroy_state(params)
    elseif method == "snapshot_state"
        handle_snapshot_state(params)
    elseif method == "restore_state"
        handle_restore_state(params)
    elseif method == "condition"
        handle_condition(params)
    elseif method == "weights"
        handle_weights(params)
    elseif method == "mean"
        handle_mean(params)
    elseif method == "expect"
        handle_expect(params)
    elseif method == "optimise"
        handle_optimise(params)
    elseif method == "value"
        handle_value(params)
    elseif method == "draw"
        handle_draw(params)
    elseif method == "enumerate"
        handle_enumerate(params)
    elseif method == "perturb_grammar"
        handle_perturb_grammar(params)
    elseif method == "add_programs"
        handle_add_programs(params)
    elseif method == "sync_prune"
        handle_sync_prune(params)
    elseif method == "sync_truncate"
        handle_sync_truncate(params)
    elseif method == "top_grammars"
        handle_top_grammars(params)
    elseif method == "belief_summary"
        handle_belief_summary(params)
    elseif method == "condition_and_prune"
        handle_condition_and_prune(params)
    elseif method == "eu_interact"
        handle_eu_interact(params)
    elseif method == "call_dsl"
        handle_call_dsl(params)
    elseif method == "factor"
        handle_factor(params)
    elseif method == "replace_factor"
        handle_replace_factor(params)
    elseif method == "n_factors"
        handle_n_factors(params)
    else
        throw(ErrorException("method not found: $method"))
    end
end

# ═══════════════════════════════════════
# Handler implementations
# ═══════════════════════════════════════

function handle_initialize(params)
    # Load DSL files
    dsl_files = get(params, "dsl_files", Dict())
    for (name, path) in dsl_files
        source = read(string(path), String)
        env = load_dsl(source)
        DSL_ENVS[string(name)] = env
        log_msg("Loaded DSL: $name from $path")
    end

    # Load plugins
    plugins = get(params, "plugins", [])
    for path in plugins
        include(string(path))
        log_msg("Loaded plugin: $path")
    end

    Dict("version" => "0.1.0")
end

function handle_create_state(params)
    t = params["type"]
    if t == "program_space"
        reset_grammar_counter!()
        grammars_spec = params["grammars"]
        max_depth = Int(get(params, "max_depth", 3))
        action_space = Symbol[Symbol(a) for a in params["action_space"]]

        # Build grammars and enumerate programs
        grammar_pool = Grammar[]
        for gs in grammars_spec
            g = build_grammar(gs)
            push!(grammar_pool, g)
        end

        # Initialize empty AgentState
        belief = MixtureMeasure(Interval(0.0, 1.0),
            Measure[TaggedBetaMeasure(Interval(0.0, 1.0), 1, BetaMeasure())],
            [0.0])
        state = AgentState(belief,
            Tuple{Int,Int}[(0, 0)],
            CompiledKernel[],
            Program[],
            Dict{Int, Grammar}(),
            max_depth)

        # Enumerate from each grammar
        # First, fix the placeholder: remove the dummy component
        state.belief = MixtureMeasure(Interval(0.0, 1.0),
            Measure[], Float64[])

        # Re-initialize properly after enumeration
        total_added = 0
        for g in grammar_pool
            state.grammars[g.id] = g
            # Can't use add_programs_to_state! on empty state, so build manually
            programs = enumerate_programs(g, max_depth;
                action_space=action_space, min_log_prior=-20.0)
            for (pi, p) in enumerate(programs)
                tag = length(state.compiled_kernels) + 1
                push!(state.belief.components,
                    TaggedBetaMeasure(Interval(0.0, 1.0), tag, BetaMeasure()))
                lw = -g.complexity * log(2) - p.complexity * log(2)
                push!(state.belief.log_weights, lw)
                push!(state.metadata, (g.id, pi))
                push!(state.compiled_kernels, compile_kernel(p, g, pi))
                push!(state.all_programs, p)
                total_added += 1
            end
        end

        # Normalize weights
        if !isempty(state.belief.log_weights)
            state.belief = MixtureMeasure(Interval(0.0, 1.0),
                state.belief.components, state.belief.log_weights)
        end

        id = register_state(state)
        Dict("state_id" => id, "n_components" => length(state.belief.components))
    else
        measure = build_measure(params)
        id = register_state(measure)
        result = Dict{String, Any}("state_id" => id)
        if measure isa MixtureMeasure
            result["n_components"] = length(measure.components)
        end
        result
    end
end

function handle_destroy_state(params)
    id = string(params["state_id"])
    delete!(STATE_REGISTRY, id)
    "ok"
end

function handle_snapshot_state(params)
    id = string(params["state_id"])
    state = get_state(id)
    io = IOBuffer()
    serialize(io, state)
    data = base64encode(take!(io))
    Dict("data" => data)
end

function handle_restore_state(params)
    data = base64decode(string(params["data"]))
    io = IOBuffer(data)
    state = deserialize(io)
    id = register_state(state)
    Dict("state_id" => id)
end

function handle_factor(params)
    id = string(params["state_id"])
    state = get_state(id)
    state isa ProductMeasure || error("factor requires a ProductMeasure, got $(typeof(state))")
    idx = Int(params["index"]) + 1  # 0-based → 1-based
    1 <= idx <= length(state.factors) || error("factor index out of range: $(idx-1)")
    new_id = register_state(factor(state, idx))
    Dict("state_id" => new_id)
end

function handle_replace_factor(params)
    id = string(params["state_id"])
    state = get_state(id)
    state isa ProductMeasure || error("replace_factor requires a ProductMeasure, got $(typeof(state))")
    idx = Int(params["index"]) + 1
    1 <= idx <= length(state.factors) || error("replace_factor index out of range: $(idx-1)")
    new_factor_id = string(params["new_factor_id"])
    new_factor = get_state(new_factor_id)
    new_factor isa Measure || error("replace_factor new factor must be a Measure, got $(typeof(new_factor))")
    new_state = replace_factor(state, idx, new_factor)
    new_id = register_state(new_state)
    Dict("state_id" => new_id)
end

function handle_n_factors(params)
    id = string(params["state_id"])
    state = get_state(id)
    state isa ProductMeasure || error("n_factors requires a ProductMeasure, got $(typeof(state))")
    Dict("n_factors" => length(state.factors))
end

function handle_condition(params)
    id = string(params["state_id"])
    state = get_state(id)
    kernel = build_kernel(params["kernel"], id)
    obs = params["observation"]
    if obs isa AbstractString
        obs = Symbol(obs)
    elseif obs isa Number
        obs = Float64(obs)
    end

    new_state = condition(state, kernel, obs)
    STATE_REGISTRY[id] = new_state

    # Compute log marginal if possible
    log_marg = try
        log_predictive(state, kernel, obs)
    catch
        nothing
    end

    result = Dict{String, Any}("state_id" => id)
    log_marg !== nothing && (result["log_marginal"] = log_marg)
    result
end

function handle_weights(params)
    id = string(params["state_id"])
    state = get_state(id)
    Dict("weights" => collect(Float64, weights(state)))
end

function handle_mean(params)
    id = string(params["state_id"])
    state = get_state(id)
    Dict("mean" => Float64(mean(state)))
end

function handle_expect(params)
    id = string(params["state_id"])
    state = get_state(id)
    f = build_function(params["function"])
    Dict("value" => Float64(expect(state, f)))
end

function handle_optimise(params)
    id = string(params["state_id"])
    state = get_state(id)
    pref_spec = params["preference"]

    # functional_per_action: explicit Functional per action, dispatched through
    # expect's type table. Enables closed-form EU via Projection/LinearCombination.
    if pref_spec["type"] == "functional_per_action"
        best_action = nothing
        best_eu = -Inf
        for (action_str, fn_spec) in pref_spec["actions"]
            φ = build_function(fn_spec)
            eu = Float64(expect(state, φ))
            if eu > best_eu
                best_eu = eu
                best_action = tryparse(Int, string(action_str))
                best_action === nothing && (best_action = string(action_str))
            end
        end
        return Dict("action" => best_action, "eu" => best_eu)
    end

    # Legacy path for tabular_2d / bdsl preferences
    actions = build_space(params["actions"])
    pref = build_preference(pref_spec)
    best_action = nothing
    best_eu = -Inf
    for a in support(actions)
        eu = expect(state, h -> pref(h, a))
        if eu > best_eu
            best_eu = eu
            best_action = a
        end
    end
    Dict("action" => best_action, "eu" => best_eu)
end

function handle_value(params)
    id = string(params["state_id"])
    state = get_state(id)
    pref_spec = params["preference"]

    if pref_spec["type"] == "functional_per_action"
        best_eu = -Inf
        for (_, fn_spec) in pref_spec["actions"]
            φ = build_function(fn_spec)
            eu = Float64(expect(state, φ))
            best_eu = max(best_eu, eu)
        end
        return Dict("value" => best_eu)
    end

    actions = build_space(params["actions"])
    pref = build_preference(pref_spec)
    best_eu = -Inf
    for a in support(actions)
        eu = expect(state, h -> pref(h, a))
        best_eu = max(best_eu, eu)
    end
    Dict("value" => best_eu)
end

function handle_draw(params)
    id = string(params["state_id"])
    state = get_state(id)
    val = draw(state)
    Dict("value" => val)
end

# ── Tier 2: Program-space operations ──

function handle_enumerate(params)
    id = string(params["state_id"])
    state = get_state(id)::AgentState
    grammar = build_grammar(params["grammar"])
    max_depth = Int(get(params, "max_depth", state.current_max_depth))
    action_space = Symbol[Symbol(a) for a in params["action_space"]]

    state.grammars[grammar.id] = grammar
    n_added = add_programs_to_state!(state, grammar, max_depth;
        action_space=action_space)

    Dict("n_added" => n_added,
         "grammar_id" => grammar.id,
         "n_components" => length(state.belief.components))
end

function handle_perturb_grammar(params)
    id = string(params["state_id"])
    state = get_state(id)::AgentState
    grammar_id = Int(params["grammar_id"])
    all_features = Set{Symbol}(Symbol(f) for f in params["all_features"])

    haskey(state.grammars, grammar_id) || error("grammar not found: $grammar_id")
    grammar = state.grammars[grammar_id]

    w = weights(state.belief)
    freq_table = analyse_posterior_subtrees(state.all_programs, w;
        min_frequency=2, min_complexity=2)

    new_grammar = perturb_grammar(grammar, freq_table, all_features)
    state.grammars[new_grammar.id] = new_grammar

    Dict("new_grammar_id" => new_grammar.id)
end

function handle_add_programs(params)
    id = string(params["state_id"])
    state = get_state(id)::AgentState
    grammar_id = Int(params["grammar_id"])
    max_depth = Int(params["max_depth"])

    haskey(state.grammars, grammar_id) || error("grammar not found: $grammar_id")
    grammar = state.grammars[grammar_id]

    n_added = add_programs_to_state!(state, grammar, max_depth)
    Dict("n_added" => n_added,
         "n_components" => length(state.belief.components))
end

function handle_sync_prune(params)
    id = string(params["state_id"])
    state = get_state(id)::AgentState
    threshold = Float64(get(params, "threshold", -30.0))
    sync_prune!(state; threshold=threshold)
    Dict("n_remaining" => length(state.belief.components))
end

function handle_sync_truncate(params)
    id = string(params["state_id"])
    state = get_state(id)::AgentState
    max_components = Int(get(params, "max_components", 2000))
    sync_truncate!(state; max_components=max_components)
    Dict("n_remaining" => length(state.belief.components))
end

function handle_top_grammars(params)
    id = string(params["state_id"])
    state = get_state(id)::AgentState
    k = Int(get(params, "k", 3))

    w = weights(state.belief)
    gw = aggregate_grammar_weights(w, state.metadata)
    top_ids = top_k_grammar_ids(state, k)

    grammars = [Dict("grammar_id" => gi, "weight" => get(gw, gi, 0.0))
                for gi in top_ids]
    Dict("grammars" => grammars)
end

# ── Batch operations ──

function handle_belief_summary(params)
    id = string(params["state_id"])
    state = get_state(id)::AgentState
    n = length(state.belief.components)
    w = weights(state.belief)
    means = [mean(state.belief.components[i]) for i in 1:n]
    gw = aggregate_grammar_weights(w, state.metadata)

    # Top programs (up to 10)
    perm = sortperm(w, rev=true)
    top_n = min(10, n)
    top_progs = []
    for i in 1:top_n
        idx = perm[i]
        gi, pi = state.metadata[idx]
        expr_str = try show_expr(state.all_programs[idx].expr) catch; "?" end
        push!(top_progs, Dict(
            "index" => idx - 1,  # 0-based for host
            "weight" => w[idx],
            "grammar_id" => gi,
            "program_id" => pi,
            "mean" => means[idx],
            "expr" => expr_str))
    end

    Dict("n_components" => n,
         "weights" => collect(Float64, w),
         "means" => collect(Float64, means),
         "grammar_weights" => Dict(string(k) => v for (k, v) in gw),
         "top_programs" => top_progs)
end

function handle_condition_and_prune(params)
    id = string(params["state_id"])
    state = get_state(id)::AgentState
    kernel = build_kernel(params["kernel"], id)
    obs = Float64(params["observation"])
    threshold = Float64(get(params, "prune_threshold", -30.0))
    max_comp = Int(get(params, "max_components", 2000))

    log_marg = try log_predictive(state.belief, kernel, obs) catch; nothing end

    state.belief = condition(state.belief, kernel, obs)
    sync_prune!(state; threshold=threshold)
    sync_truncate!(state; max_components=max_comp)

    result = Dict{String, Any}("n_remaining" => length(state.belief.components))
    log_marg !== nothing && (result["log_marginal"] = log_marg)
    result
end

function handle_eu_interact(params)
    id = string(params["state_id"])
    state = get_state(id)::AgentState
    features = Dict{Symbol, Float64}(Symbol(k) => Float64(v)
                                      for (k, v) in params["features"])
    rewards = Dict{Symbol, Float64}(Symbol(k) => Float64(v)
                                     for (k, v) in params["rewards"])
    temporal_state = Dict{Symbol, Any}()

    w = weights(state.belief)
    label_probs = Dict{Symbol, Float64}()

    for (j, comp) in enumerate(state.belief.components)
        ck = state.compiled_kernels[j]
        rec = ck.evaluate(features, temporal_state)
        mean_j = mean(comp)
        # p(rec is correct) = mean_j, p(rec is wrong) = 1 - mean_j
        for (label, _) in rewards
            p_label = if rec == label
                w[j] * mean_j
            else
                # If there are only 2 labels, the other gets 1-mean_j
                w[j] * (1.0 - mean_j) / max(length(rewards) - 1, 1)
            end
            label_probs[label] = get(label_probs, label, 0.0) + p_label
        end
    end

    eu = sum(label_probs[l] * rewards[l] for l in keys(rewards))
    Dict("eu" => eu, "p_labels" => Dict(string(l) => p for (l, p) in label_probs))
end

# ── DSL operations ──

function handle_call_dsl(params)
    env_id = string(params["env_id"])
    fn_name = string(params["function"])
    args = get(params, "args", [])

    env = get(DSL_ENVS, env_id, nothing)
    env !== nothing || error("DSL environment not found: $env_id")

    fn_sym = Symbol(fn_name)
    haskey(env, fn_sym) || error("function not found in DSL env '$env_id': $fn_name")
    fn = env[fn_sym]

    resolved_args = [resolve_arg(a) for a in args]
    result = fn(resolved_args...)

    serialize_value(result)
end

# ═══════════════════════════════════════
# Main loop
# ═══════════════════════════════════════

function main()
    log_msg("Credence brain server starting...")

    for line in eachline(stdin)
        isempty(strip(line)) && continue

        local req
        try
            req = JSON3.read(line)
        catch e
            send_response(make_error(nothing, -32700, "Parse error: $(sprint(showerror, e))"))
            continue
        end

        id = get(req, :id, get(req, "id", nothing))
        method = get(req, :method, get(req, "method", nothing))
        params = get(req, :params, get(req, "params", Dict()))

        if method === nothing
            send_response(make_error(id, -32600, "Invalid request: missing method"))
            continue
        end

        try
            result = handle_request(string(method), params, id)

            if method == "shutdown"
                send_response(make_response(id, result))
                return
            end

            send_response(make_response(id, result))
        catch e
            code = if e isa StateNotFound
                -32000
            elseif e isa InferenceError
                -32001
            elseif e isa DSLError
                -32002
            elseif occursin("method not found", sprint(showerror, e))
                -32601
            else
                -32602
            end
            msg = sprint(showerror, e)
            log_msg("Error handling $method: $msg")
            send_response(make_error(id, code, msg))
        end
    end
end

# Base64 encoding (Julia stdlib)
using Base64

main()

#!/usr/bin/env julia
# Role: skin
"""
    apps/skin/server.jl — Credence skin process

JSON-RPC 2.0 over stdio. Reads newline-delimited JSON from stdin,
writes responses to stdout, logs to stderr.

The skin holds stateful objects (measures, agent states, DSL environments)
in a registry keyed by opaque string IDs. The host never sees the internal
representation — only the ID.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Credence: expect, condition, push_measure, draw, density
using Credence: Event, TagSet, FeatureEquals, FeatureInterval, Conjunction, Disjunction, Complement
using Credence: _dispatch_path
using Credence: weights, mean, variance, prune, truncate, logsumexp
using Credence: CategoricalMeasure, ProductMeasure, MixtureMeasure
using Credence: Prevision, Measure, params
using Credence: BetaPrevision, TaggedBetaPrevision, GaussianPrevision
using Credence: MvGaussianPrevision, MvGaussianMeasure, LinearGaussian, GaussianMeasure
using Credence: GammaPrevision, DirichletPrevision, NormalGammaPrevision
using Credence: ProductPrevision, MixturePrevision, CategoricalPrevision
using Credence: Finite, Interval, ProductSpace, Euclidean, PositiveReals, Simplex
using Credence: Kernel, FactorSelector, support
using Credence: Functional, Identity, Projection, NestedProjection, Tabular, LinearCombination, OpaqueClosure
using Credence: CenteredPower, CenteredSquare, marginalise
using Credence: factor, replace_factor
using Credence: AgentState, sync_prune!, sync_truncate!
using Credence: Grammar, Program, CompiledKernel, ProductionRule
using Credence: enumerate_programs, compile_kernel
using Credence: analyse_posterior_subtrees, perturb_grammar
using Credence: aggregate_grammar_weights, top_k_grammar_ids, add_programs_to_state!
using Credence: next_grammar_id, reset_grammar_counter!
using Credence: show_expr
using Credence: log_density_at
using Credence: StructureBMA, build_structure_model, build_structure_prior, structure_observe,
                belief_at_context, context_from_features, structure_decision_kernel
using Credence.Ontology: decide_with_voi
using Credence: RoutingState, EmissionBelief, route_decide, escalate_decide, route_outcome!,
    routing_belief_readout, reconstruct_routing_tops_from_data, reconstruct_latency_from_data

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

# Model registry: structural-analysis DESCRIPTORS (e.g. a `StructureBMA`) held under a
# SEPARATE `m_*` handle from the `s_*` belief states (Invariant 3 / decouple Move-3 Q1 —
# a descriptor carries no beliefs; conflating the two would let one mutate the other).
const MODEL_REGISTRY = Dict{String, Any}()
let counter = Ref(0)
    global function next_model_id()
        counter[] += 1
        "m_$(counter[])"
    end
end

function register_model(obj)
    id = next_model_id()
    MODEL_REGISTRY[id] = obj
    id
end

function get_model(id::String)
    haskey(MODEL_REGISTRY, id) || throw(StateNotFound(id))
    MODEL_REGISTRY[id]
end

# Routing state (decouple Move 4) — a MUTABLE, growing bundle: route_outcome! mutates `tops`
# + emission cells in place, and `extra_tops`/ρ/σ cells grow with unseen models/contexts. So
# it gets its OWN `rt_*` registry, NOT MODEL_REGISTRY (which holds IMMUTABLE descriptors —
# putting a mutated, growing bundle there would void that invariant) and NOT STATE_REGISTRY
# (so the generic Prevision verbs give a clean StateNotFound on a routing handle).
const ROUTING_REGISTRY = Dict{String, Any}()
let counter = Ref(0)
    global function next_routing_id()
        counter[] += 1
        "rt_$(counter[])"
    end
end

function register_routing(rt)
    id = next_routing_id()
    ROUTING_REGISTRY[id] = rt
    id
end

function get_routing(id::String)
    haskey(ROUTING_REGISTRY, id) || throw(StateNotFound(id))
    ROUTING_REGISTRY[id]
end

# ═══════════════════════════════════════
# Error types
# ═══════════════════════════════════════

struct StateNotFound <: Exception
    id::String
end

struct MethodNotFound <: Exception
    method::String
end

# Raised by `initialize` when the client's pinned protocol major does not
# match the server's. Surfaces as JSON-RPC error -32010 so a client compiled
# against an incompatible contract refuses to proceed rather than silently
# misinterpreting later responses. See docs/decouple/move-0-design.md §4.
struct ProtocolMismatch <: Exception
    server::String
    client::String
end
Base.showerror(io::IO, e::ProtocolMismatch) =
    print(io, "protocol major mismatch: server ", e.server, ", client ", e.client)

# Wrapping exceptions for narrower client-facing error codes. Thrown at
# handler boundaries around build_kernel / build_function / condition /
# expect call sites so that protocol errors (-32001) vs inference errors
# (-32002) surface distinctly instead of collapsing to -32603.
struct DSLError <: Exception
    msg::String
    cause::Exception
end
Base.showerror(io::IO, e::DSLError) =
    print(io, "DSLError: ", e.msg, "\n  cause: ", sprint(showerror, e.cause))

struct InferenceError <: Exception
    msg::String
    cause::Exception
end
Base.showerror(io::IO, e::InferenceError) =
    print(io, "InferenceError: ", e.msg, "\n  cause: ", sprint(showerror, e.cause))

_wrap_dsl(msg) = e -> e isa StateNotFound || e isa MethodNotFound ? rethrow() :
                      throw(DSLError(msg, e))
_wrap_inf(msg) = e -> e isa StateNotFound || e isa MethodNotFound ? rethrow() :
                      throw(InferenceError(msg, e))

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
# State construction from spec
# ═══════════════════════════════════════

# Rebuild a dense covariance Matrix from the wire's vector-of-rows (the shape
# `params(::MvGaussianPrevision)` emits — a bare matrix flattens on JSON).
_cov_from_rows(rows) = mapreduce(r -> permutedims(collect(Float64, r)), vcat, rows)

# Discretise a continuous prior onto a label grid (the labelled_mixture latent prior): the
# unnormalised log-density at each label; MixturePrevision normalises. Keeps a ρ~Beta carried
# EXACTLY in shape with no host density arithmetic (mirrors discretised_gaussian's grid prior).
# Labels must be interior to the support (β-kernel diverges at {0,1} for α<1 or β<1).
function _discretise_label_prior(spec, labels::Vector{Float64})
    t = spec["type"]
    if t == "beta"
        a = Float64(spec["alpha"]); b = Float64(spec["beta"])
        [(a - 1.0) * log(ℓ) + (b - 1.0) * log(1.0 - ℓ) for ℓ in labels]
    elseif t == "gaussian"
        mu = Float64(spec["mu"]); sigma = Float64(spec["sigma"])
        [-0.5 * ((ℓ - mu) / sigma)^2 for ℓ in labels]
    else
        error("unknown label_prior type: $t (expected beta or gaussian)")
    end
end

function build_prevision(spec)
    t = spec["type"]
    if t == "categorical"
        space = build_space(spec["space"])
        if haskey(spec, "log_weights")
            CategoricalMeasure(space, collect(Float64, spec["log_weights"]))
        else
            CategoricalMeasure(space)
        end
    elseif t == "beta"
        BetaPrevision(Float64(spec["alpha"]), Float64(spec["beta"]))
    elseif t == "tagged_beta"
        TaggedBetaPrevision(Int(spec["tag"]),
                            BetaPrevision(Float64(spec["alpha"]), Float64(spec["beta"])))
    elseif t == "gaussian"
        GaussianPrevision(Float64(spec["mu"]), Float64(spec["sigma"]))
    elseif t == "mv_gaussian"
        MvGaussianPrevision(collect(Float64, spec["mu"]), _cov_from_rows(spec["sigma"]))
    elseif t == "gamma"
        GammaPrevision(Float64(spec["alpha"]), Float64(spec["beta"]))
    elseif t == "dirichlet"
        DirichletPrevision(collect(Float64, spec["alpha"]))
    elseif t == "normal_gamma"
        NormalGammaPrevision(Float64(spec["kappa"]), Float64(spec["mu"]),
                             Float64(spec["alpha"]), Float64(spec["beta"]))
    elseif t == "product"
        factors = [build_measure(f) for f in spec["factors"]]
        ProductMeasure(factors)
    elseif t == "mixture"
        components = [build_prevision(c) for c in spec["components"]]
        lw = collect(Float64, spec["log_weights"])
        MixturePrevision(components, lw)
    elseif t == "labelled_mixture"
        # A carried, shared discrete latent (decouple Move 1): a mixture over `labels`
        # (e.g. a ρ-grid), each component a LabelledCategoricalPrevision sharing one
        # categorical V-prior (`component_log_weights`). A per-component-routing kernel
        # (group_noisy_channel) reads each label at condition time, so conditioning learns
        # the latent jointly with V and shares it across observations. The latent prior over
        # `labels` is `label_prior` (a belief-spec the engine discretises onto the grid —
        # e.g. `{type:beta, alpha, beta}` for a ρ~Beta carried exactly, no host density
        # arithmetic) OR explicit `label_log_weights`; default uniform.
        labels = collect(Float64, spec["labels"])
        comp_lw = collect(Float64, spec["component_log_weights"])
        label_lw = if haskey(spec, "label_prior")
            _discretise_label_prior(spec["label_prior"], labels)
        elseif haskey(spec, "label_log_weights")
            collect(Float64, spec["label_log_weights"])
        else
            zeros(Float64, length(labels))
        end
        comps = LabelledCategoricalPrevision[
            LabelledCategoricalPrevision(ℓ, CategoricalPrevision(copy(comp_lw))) for ℓ in labels]
        MixturePrevision(comps, label_lw)
    elseif t == "discretised_gaussian"
        # A Gaussian discretised onto a grid (a stated prior shape, decouple Move 1) — the
        # engine builds the grid prior the body assembled host-side (utility.py gaussian_weights).
        # Log-weights are the unnormalised Gaussian; CategoricalMeasure normalises (≡ exp/Σ up to
        # ULP). The grid is a stated truncation; widen it, never renormalise (utility.py §4.4).
        grid = collect(Float64, spec["grid"])
        mu = Float64(spec["mu"]); sigma = Float64(spec["sigma"])
        CategoricalMeasure(Finite(grid), [-0.5 * ((x - mu) / sigma)^2 for x in grid])
    else
        error("unknown type: $t")
    end
end

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
    elseif t == "mv_gaussian"
        mu = collect(Float64, spec["mu"])
        MvGaussianMeasure(Euclidean(length(mu)),
                          MvGaussianPrevision(mu, _cov_from_rows(spec["sigma"])))
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
            (θ, o) -> o == 1.0 ? log(max(θ, 1e-300)) : log(max(1.0 - θ, 1e-300));
            likelihood_family = BetaBernoulli())

    elseif t == "gaussian_known_var"
        variance = Float64(spec["variance"])
        source = Euclidean(1)
        target = Euclidean(1)
        Kernel(source, target,
            μ -> error("generate not used"),
            (μ, o) -> -0.5 * (o - μ)^2 / variance;
            params = Dict{Symbol, Any}(:sigma_obs => sqrt(variance)),
            likelihood_family = PushOnly())

    elseif t == "linear_gaussian"
        # y ~ N(aᵀw, σ²): joint linear-Gaussian observation of a multivariate
        # Gaussian state. The conjugate update (exact Kalman) is keyed off the
        # LinearGaussian family; log_density is the predictive at a concrete w.
        coeffs = collect(Float64, spec["coeffs"])
        variance = Float64(spec["variance"])
        d = length(coeffs)
        Kernel(Euclidean(d), Euclidean(1),
            w -> error("generate not used"),
            (w, o) -> -0.5 * (o - sum(coeffs[i] * w[i] for i in 1:d))^2 / variance;
            likelihood_family = LinearGaussian(coeffs, sqrt(variance)))

    elseif t == "group_noisy_channel"
        # A document of correlated chunk-extractions, read as evidence about a categorical
        # V carried in a `labelled_mixture` (label = ρ). DispatchByComponent reads each
        # component's label to set r_d = ρ·covariate; the chunk reports (1-based candidate
        # positions) are the `condition` observation (a JSON array). Routing is per
        # component, so log_density is never called — the labelled categorical reaches the
        # resolved family through `categorical_logdensity`.
        cov = Float64(spec["covariate"])
        A = Int(spec["n_alternatives"])
        Kernel(Finite([0.0]), Finite([0.0]),
            h -> error("generate not used"),
            (h, o) -> error("group_noisy_channel routes via categorical_logdensity, not log_density");
            likelihood_family = DispatchByComponent(c -> GroupNoisyChannel(cov, c.label, A)))

    elseif t == "logistic_reaction"
        # A binary reaction to a latent x (e.g. a utility on a grid), under the choice model
        # marginalised over a declared τ-prior. Conditions a categorical over the x-grid;
        # the body ships (sign, threshold, τ-grid, τ-weights) as declared data.
        fam = LogisticReaction(Float64(spec["sign"]), Float64(spec["threshold"]),
                               collect(Float64, spec["tau_values"]),
                               collect(Float64, spec["tau_weights"]))
        Kernel(Euclidean(1), Finite([0.0, 1.0]),
            x -> error("generate not used"),
            (x, o) -> logistic_reaction_logdensity(fam, x, o);
            likelihood_family = fam)

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
            end;
            likelihood_family = PushOnly())

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
                if m_or_θ isa TaggedBetaPrevision
                    tag = m_or_θ.tag
                    recommended = get!(recommendation_cache, tag) do
                        ck = state.compiled_kernels[tag]
                        ck.evaluate(features, temporal_state)
                    end
                    correct = recommended == true_label
                    correct_cache[tag] = correct
                    p = mean(m_or_θ.beta)
                    correct ? log(max(p, 1e-300)) : log(max(1.0 - p, 1e-300))  # credence-lint: allow — precedent:declarative-construction — Kernel log-density closure: Bernoulli likelihood from Beta mean
                else
                    obs == 1.0 ? log(max(m_or_θ, 1e-300)) : log(max(1.0 - m_or_θ, 1e-300))
                end
            end;
            params = Dict{Symbol, Any}(:correct_cache => correct_cache),
            likelihood_family = BetaBernoulli())

    elseif t == "tabular_log_density"
        source_vals = collect(Float64, spec["source_vals"])
        target_vals = collect(Float64, spec["target_vals"])
        densities = [collect(Float64, row) for row in spec["densities"]]
        length(densities) == length(source_vals) ||
            error("tabular kernel: densities has $(length(densities)) rows but source_vals has $(length(source_vals)) entries")
        for (i, row) in enumerate(densities)
            length(row) == length(target_vals) ||
                error("tabular kernel: densities row $i has $(length(row)) entries but target_vals has $(length(target_vals)) entries")
        end
        source = Finite(source_vals)
        target = Finite(target_vals)
        Kernel(source, target,
            _ -> error("generate not used"),
            (h, o) -> begin
                si = findfirst(==(h), source_vals)
                si !== nothing ||
                    error("tabular kernel: hypothesis $h not in source_vals $source_vals — this should have been caught at construction")
                ti = findfirst(==(o), target_vals)
                ti !== nothing ||
                    error("tabular kernel: observation $o not in target_vals $target_vals — this should have been caught at construction")
                densities[si][ti]
            end;
            likelihood_family = PushOnly())

    elseif t == "dsl"
        env_id = spec["env_id"]
        fn_name = spec["fn_name"]
        args = get(spec, "args", [])
        env = get(DSL_ENVS, env_id, nothing)
        env !== nothing || error("DSL environment not found: $env_id")
        fn = env[Symbol(fn_name)]
        resolved_args = [resolve_arg(a) for a in args]
        fn(resolved_args...)

    elseif t == "flat"
        # Flat likelihood: no-op conjugate for BetaMeasure. Any source/target
        # shape with a declared likelihood_family = Flat() is recognised by
        # maybe_conjugate(::BetaPrevision, ...). Declared here over
        # Interval(0,1) → Finite(0,1) for parity with the bernoulli kernel.
        source = Interval(0.0, 1.0)
        target = Finite([0.0, 1.0])
        Kernel(source, target,
            θ -> error("generate not used"),
            (θ, o) -> 0.0;
            likelihood_family = Flat())

    elseif t == "categorical"
        # Dirichlet-Categorical conjugate kernel. Categories are Float64
        # labels [0.0, 1.0, …, k-1] to match the skin's create_state for
        # DirichletMeasure. The Categorical likelihood marker carries the
        # category list as a Finite so maybe_conjugate(::DirichletPrevision,
        # ...) can look up obs→idx at update time.
        cats = Finite(collect(Float64, spec["categories"]))
        k = length(cats.values)
        source = Simplex(k)
        target = cats
        Kernel(source, target,
            _ -> error("generate not used"),
            (θ, o) -> begin
                idx = findfirst(==(o), cats.values)
                idx === nothing ? -Inf : log(max(θ[idx], 1e-300))
            end;
            likelihood_family = Categorical(cats))

    elseif t == "normal_gamma"
        # Normal-Gamma conjugate kernel. The hypothesis h = (μ, σ²) is
        # drawn from NormalGammaMeasure (see `draw(::NormalGammaMeasure)`);
        # the likelihood is N(o; μ, σ²). Density is used by
        # log_predictive's generic fallback even though the posterior
        # goes through the conjugate registry.
        source = Euclidean(1)
        target = Euclidean(1)
        Kernel(source, target,
            _ -> error("generate not used"),
            (h, o) -> begin
                μ, σ² = h[1], h[2]
                -0.5 * log(2π * σ²) - 0.5 * (o - μ)^2 / σ²
            end;
            likelihood_family = NormalGammaLikelihood())

    elseif t == "exponential"
        # Gamma-Exponential conjugate kernel. Rate λ ∈ ℝ₊; observation
        # o ∈ ℝ₊. Log-density log(λ) - λo.
        source = PositiveReals()
        target = PositiveReals()
        Kernel(source, target,
            λ -> error("generate not used"),
            (λ, o) -> log(max(λ, 1e-300)) - λ * o;
            likelihood_family = Exponential())

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
    elseif t == "projection" || t == "project"
        Projection(Int(spec["index"]) + 1)  # 0-based → 1-based
    elseif t == "nested_projection"
        NestedProjection(Int[Int(i) + 1 for i in spec["indices"]])
    elseif t == "tabular"
        Tabular(collect(Float64, spec["values"]))
    elseif t == "centered_power" || t == "centered_square"
        # E[(θ−μ)^n] over a scalar belief; μ=0 (default) is the raw moment E[θ^n].
        # The integrated claim-EU ships LinearCombination([(u_c−u_w, centered_power n=2),
        # (u_w, identity)], offset=−κ) — exact via the closed-form Beta moment.
        deg = t == "centered_square" ? 2 : Int(spec["n"])
        CenteredPower{deg}(Float64(get(spec, "mu", 0.0)))
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
        elseif haskey(arg, "type")
            # A declarative belief-spec (same shape as create_state) — reconstruct
            # the belief WITHOUT registering it. This is how a belief crosses the
            # wire INTO a pure call_dsl function: build, never hold. See
            # docs/decouple/move-2-design.md.
            return build_prevision(arg)
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

# Serialize a belief (Prevision or Measure facade) to a declarative belief-spec
# dict via the `params` protocol — the same shape `create_state`/`build_prevision`
# consume, so it round-trips. The `:type` tag becomes a string. PURE: emits the
# belief OUT of a call_dsl function without registering any state.
function _belief_spec(belief)
    nt = params(belief)
    Dict{String, Any}(string(k) => (v isa Symbol ? string(v) : v) for (k, v) in pairs(nt))
end

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
    elseif val isa Prevision || val isa Measure
        # A belief returned from a pure call_dsl function → inline belief-spec,
        # no state registry. See docs/decouple/move-2-design.md.
        _belief_spec(val)
    elseif val isa AbstractVector && all(x -> x isa Prevision || x isa Measure, val)
        # A positional list of beliefs (e.g. a MAUT weight vector) → list of specs.
        [_belief_spec(x) for x in val]
    else
        # Opaque Julia object (nested lists, agent states, etc.)
        # Wrap as a single state — the host passes it back via {"ref": id}
        Dict("state_id" => register_state(val))
    end
end

# ═══════════════════════════════════════
# Method dispatch
# ═══════════════════════════════════════

# Wire protocol version, versioned independently of the engine semver (which
# rides the `credence-skin` image tag). MAJOR bumps on a breaking protocol
# change; MINOR on additive. Apps pin the major in code and `initialize`
# rejects a mismatching major with -32010. See docs/decouple/master-plan.md.
const PROTOCOL_VERSION = "1.8"
protocol_major(v) = String(first(split(String(v), ".")))

# Advertised method set, returned by `initialize` for client capability
# discovery. Must stay in sync with the `handle_request` dispatch below
# (the `_dispatch_path` diagnostic verb is intentionally omitted).
const SKIN_METHODS = [
    "initialize", "shutdown", "create_state", "destroy_state",
    "snapshot_state", "restore_state", "condition", "condition_on_event",
    "weights", "mean", "expect", "optimise", "value", "marginalise", "draw", "enumerate",
    "perturb_grammar", "add_programs", "sync_prune", "sync_truncate",
    "top_grammars", "belief_summary", "condition_and_prune", "eu_interact",
    "call_dsl", "factor", "replace_factor", "n_factors",
    "structure_bma", "structure_observe", "structure_decide",
    "routing_init", "routing_decide", "routing_escalate", "routing_outcome",
    "routing_belief", "destroy_routing",
]

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
    elseif method == "condition_on_event"
        handle_condition_on_event(params)
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
    elseif method == "marginalise"
        handle_marginalise(params)
    elseif method == "structure_bma"
        handle_structure_bma(params)
    elseif method == "structure_observe"
        handle_structure_observe(params)
    elseif method == "structure_decide"
        handle_structure_decide(params)
    elseif method == "routing_init"
        handle_routing_init(params)
    elseif method == "routing_decide"
        handle_routing_decide(params)
    elseif method == "routing_escalate"
        handle_routing_escalate(params)
    elseif method == "routing_outcome"
        handle_routing_outcome(params)
    elseif method == "routing_belief"
        handle_routing_belief(params)
    elseif method == "destroy_routing"
        handle_destroy_routing(params)
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
    elseif method == "_dispatch_path"
        handle_dispatch_path(params)
    else
        throw(MethodNotFound(string(method)))
    end
end

# ═══════════════════════════════════════
# Handler implementations
# ═══════════════════════════════════════

function handle_initialize(params)
    # Protocol handshake: if the client pins a protocol major, reject a
    # mismatch up front rather than misinterpreting later responses. Omitted
    # `protocol` means a legacy client — proceed (backward compatible).
    client_proto = get(params, "protocol", nothing)
    if client_proto !== nothing
        cmaj = protocol_major(client_proto)
        smaj = protocol_major(PROTOCOL_VERSION)
        cmaj == smaj || throw(ProtocolMismatch(smaj, cmaj))
    end

    # Inline DSL sources (the external surface): name -> BDSL source string,
    # passed over the wire so the engine never reaches into the host
    # filesystem. This is how a containerised consumer declares its domain.
    dsl_sources = get(params, "dsl_sources", Dict())
    for (name, source) in dsl_sources
        env = load_dsl(string(source))
        DSL_ENVS[string(name)] = env
        log_msg("Loaded DSL: $name (inline source)")
    end

    # Path-based DSL files: co-released-image-only (the engine and the .bdsl
    # share a filesystem inside an engine-repo image, e.g. credence-proxy
    # loading examples/router.bdsl). Not part of the external wire contract.
    dsl_files = get(params, "dsl_files", Dict())
    for (name, path) in dsl_files
        source = read(string(path), String)
        env = load_dsl(source)
        DSL_ENVS[string(name)] = env
        log_msg("Loaded DSL: $name from $path")
    end

    # Julia plugin injection: co-released-image-only. Loading host .jl into the
    # engine process is a brain-injection vector and is NOT part of the external
    # wire contract; external consumers declare domains as data (dsl_sources),
    # never as code. See docs/decouple/master-plan.md.
    plugins = get(params, "plugins", [])
    for path in plugins
        log_msg("WARNING: plugin injection is co-released-image-only, not a wire-contract surface: $path")
        include(string(path))
        log_msg("Loaded plugin: $path")
    end

    Dict("version" => "0.1.0", "protocol" => PROTOCOL_VERSION, "methods" => SKIN_METHODS)
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

        # Build components from grammars
        all_components = TaggedBetaPrevision[]
        all_lw = Float64[]
        all_meta = Tuple{Int,Int}[]
        all_ck = CompiledKernel[]
        all_progs = Program[]

        for g in grammar_pool
            programs = enumerate_programs(g, max_depth;
                action_space=action_space, min_log_prior=-20.0)
            for (pi, p) in enumerate(programs)
                tag = length(all_ck) + 1
                push!(all_components, TaggedBetaPrevision(tag, BetaPrevision(1.0, 1.0)))
                push!(all_lw, -g.complexity * log(2) - p.complexity * log(2))
                push!(all_meta, (g.id, pi))
                push!(all_ck, compile_kernel(p, g, pi))
                push!(all_progs, p)
            end
        end

        belief = MixturePrevision(all_components, all_lw)
        grammar_dict = Dict{Int, Grammar}(g.id => g for g in grammar_pool)
        state = AgentState(belief, all_meta, all_ck, all_progs, grammar_dict, max_depth)

        id = register_state(state)
        Dict("state_id" => id, "n_components" => length(state.belief.components))
    else
        state = build_prevision(params)
        id = register_state(state)
        result = Dict{String, Any}("state_id" => id)
        if state isa MixturePrevision || state isa MixtureMeasure
            result["n_components"] = length(state.components)
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
    state isa ProductMeasure || state isa ProductPrevision ||
        error("factor requires a ProductMeasure or ProductPrevision, got $(typeof(state))")
    idx = Int(params["index"]) + 1  # 0-based → 1-based
    1 <= idx <= length(state.factors) || error("factor index out of range: $(idx-1)")
    new_id = register_state(factor(state, idx))
    Dict("state_id" => new_id)
end

function handle_replace_factor(params)
    id = string(params["state_id"])
    state = get_state(id)
    state isa ProductMeasure || state isa ProductPrevision ||
        error("replace_factor requires a ProductMeasure or ProductPrevision, got $(typeof(state))")
    idx = Int(params["index"]) + 1
    1 <= idx <= length(state.factors) || error("replace_factor index out of range: $(idx-1)")
    new_factor_id = string(params["new_factor_id"])
    new_factor = get_state(new_factor_id)
    if state isa ProductMeasure && new_factor isa Prevision
        new_factor = wrap_in_measure(new_factor)
    end
    new_state = replace_factor(state, idx, new_factor)
    new_id = register_state(new_state)
    Dict("state_id" => new_id)
end

function handle_n_factors(params)
    id = string(params["state_id"])
    state = get_state(id)
    state isa ProductMeasure || state isa ProductPrevision ||
        error("n_factors requires a ProductMeasure or ProductPrevision, got $(typeof(state))")
    Dict("n_factors" => length(state.factors))
end

# ── Move 4: _dispatch_path observability hook (underscore-prefixed, test-only) ──
# Returns "conjugate" if the (state, kernel) pair matches a registered
# conjugate entry in the ConjugatePrevision registry; "particle" otherwise.
# Does NOT mutate state or execute an update. Used by Stratum-2 tests to
# assert registry-dispatch decisions explicitly — silent registry misses
# would fall through to particle and produce correct values for the wrong
# reason without this hook.
#
# Accepts a Prevision or Measure state; forwards to the prevision-level
# `_dispatch_path` declared in src/prevision.jl.

function handle_dispatch_path(params)
    id = string(params["state_id"])
    state = get_state(id)
    kernel = build_kernel(params["kernel"])
    p = state isa Prevision ? state : state.prevision
    path = _dispatch_path(p, kernel)
    Dict("path" => string(path))
end

# Move 7 Phase 5: event-spec builder for condition_on_event RPC.
# Mirrors build_kernel's dispatch-on-type shape. Posture 2's Event
# hierarchy is Boolean-composable; this builder handles the structural
# types + binary compositions.
function build_event(spec)
    t = spec["type"]
    if t == "tag_set"
        tags = Set{Int}(Int(x) for x in spec["tags"])
        # TagSet requires a Space; the skin's conventional space for
        # tagged mixtures is Interval(0,1). Callers may specify
        # "space" explicitly if a different Space is needed.
        space = haskey(spec, "space") ? build_space(spec["space"]) : Interval(0.0, 1.0)
        TagSet(space, tags)
    elseif t == "feature_equals"
        space = haskey(spec, "space") ? build_space(spec["space"]) : Interval(0.0, 1.0)
        FeatureEquals(space, Symbol(spec["feature"]), spec["value"])
    elseif t == "feature_interval"
        space = haskey(spec, "space") ? build_space(spec["space"]) : Interval(0.0, 1.0)
        FeatureInterval(space, Symbol(spec["feature"]),
                        Float64(spec["lo"]), Float64(spec["hi"]))
    elseif t == "conjunction"
        Conjunction(build_event(spec["left"]), build_event(spec["right"]))
    elseif t == "disjunction"
        Disjunction(build_event(spec["left"]), build_event(spec["right"]))
    elseif t == "complement"
        Complement(build_event(spec["inner"]))
    else
        error("unknown event type: $t")
    end
end

function handle_condition_on_event(params)
    id = string(params["state_id"])
    state = get_state(id)
    event = try
        build_event(params["event"])
    catch e
        _wrap_dsl("build_event failed")(e)
    end

    new_state = try
        condition(state, event)
    catch e
        _wrap_inf("condition_on_event failed")(e)
    end
    STATE_REGISTRY[id] = new_state

    Dict{String, Any}("state_id" => id)
end

function handle_condition(params)
    id = string(params["state_id"])
    state = get_state(id)
    kernel = try
        build_kernel(params["kernel"], id)
    catch e
        _wrap_dsl("build_kernel failed")(e)
    end
    obs = params["observation"]
    if obs isa AbstractString
        obs = Symbol(obs)
    elseif obs isa Number
        obs = Float64(obs)
    end

    log_marg = try
        log_predictive(state, kernel, obs)
    catch e
        _wrap_inf("log_predictive failed")(e)
    end

    new_state = try
        condition(state, kernel, obs)
    catch e
        _wrap_inf("condition failed")(e)
    end
    STATE_REGISTRY[id] = new_state

    Dict{String, Any}("state_id" => id, "log_marginal" => log_marg)
end

function handle_weights(params)
    id = string(params["state_id"])
    state = get_state(id)
    Dict("weights" => collect(Float64, weights(state)))
end

# marginalise: marginal of a flat product-grid categorical along `axis` (0-based).
# A terminal readout (the consumer never re-conditions the marginal) — the engine
# sums out the other axes so the consumer ships only `{shape, axis}` data, doing no
# belief arithmetic of its own (Invariant 1). `shape` is the per-axis grid sizes in
# row-major order (last axis fastest, matching the consumer's product enumeration).
function handle_marginalise(params)
    id = string(params["state_id"])
    state = get_state(id)
    shape = Int[Int(x) for x in params["shape"]]
    axis = Int(params["axis"])
    Dict("weights" => collect(Float64, marginalise(state, shape, axis)))
end

function handle_mean(params)
    id = string(params["state_id"])
    state = get_state(id)
    Dict("mean" => Float64(mean(state)))
end

function handle_expect(params)
    id = string(params["state_id"])
    state = get_state(id)
    f = try
        build_function(params["function"])
    catch e
        _wrap_dsl("build_function failed")(e)
    end
    v = try
        Float64(expect(state, f))
    catch e
        _wrap_inf("expect failed")(e)
    end
    Dict("value" => v)
end

# Deterministic action iteration: sort keys of functional_per_action's
# actions dict. JSON3 dicts don't guarantee iteration order, and strict
# `>` tie-breaking would otherwise make the winning action host-dependent.
# Returns keys sorted: numeric keys first (by parsed value), then
# non-numeric keys lexicographically. Tuples share element types to keep
# Julia's isless dispatch happy.
function _sorted_action_keys(actions)
    ks = collect(keys(actions))
    sort(ks; by = k -> begin
        s = string(k)
        parsed = tryparse(Int, s)
        parsed === nothing ? (1, 0, s) : (0, parsed, "")
    end)
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
        for action_str in _sorted_action_keys(pref_spec["actions"])
            fn_spec = pref_spec["actions"][action_str]
            φ = try
                build_function(fn_spec)
            catch e
                _wrap_dsl("build_function failed for action $action_str")(e)
            end
            eu = try
                Float64(expect(state, φ))
            catch e
                _wrap_inf("expect failed for action $action_str")(e)
            end
            if eu > best_eu
                best_eu = eu
                best_action = tryparse(Int, string(action_str))
                best_action === nothing && (best_action = string(action_str))
            end
        end
        return Dict("action" => best_action, "eu" => best_eu)
    end

    # tabular_2d / bdsl preferences — no Functional spec, loop explicitly.
    actions = build_space(params["actions"])
    pref = build_preference(pref_spec)
    best_action = nothing
    best_eu = -Inf
    for a in support(actions)
        eu = try
            expect(state, h -> pref(h, a))
        catch e
            _wrap_inf("expect failed for action $a")(e)
        end
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
        for action_str in _sorted_action_keys(pref_spec["actions"])
            fn_spec = pref_spec["actions"][action_str]
            φ = try
                build_function(fn_spec)
            catch e
                _wrap_dsl("build_function failed for action $action_str")(e)
            end
            eu = try
                Float64(expect(state, φ))
            catch e
                _wrap_inf("expect failed for action $action_str")(e)
            end
            best_eu = max(best_eu, eu)
        end
        return Dict("value" => best_eu)
    end

    actions = build_space(params["actions"])
    pref = build_preference(pref_spec)
    best_eu = -Inf
    for a in support(actions)
        eu = try
            expect(state, h -> pref(h, a))
        catch e
            _wrap_inf("expect failed for action $a")(e)
        end
        best_eu = max(best_eu, eu)
    end
    Dict("value" => best_eu)
end

# ═══════════════════════════════════════
# Structure-BMA verbs (decouple Move 3)
#
# A non-embedding consumer drives structure-BMA inference over the wire: build the
# 2ⁿ-edge model + prior (`structure_bma`), learn per (context, response)
# (`structure_observe`), and take the proceed/block/ask EU decision (`structure_decide`).
# The descriptor gets a SEPARATE `model_id` handle from the belief `state_id` (Move-3 Q1).
# ALL arithmetic — the chain-rule reweight, the EU coefficient assembly — is engine-side
# (src/structure_bma.jl, src/stdlib.jl `decide_with_voi`); the verbs only marshal JSON,
# so Invariant 1 holds across the wire by construction. `belief_at_context` is deliberately
# NOT a verb: no consumer reads the per-context belief across the wire (Move-3 Q2).
# ═══════════════════════════════════════

# A features dict {name: bucket} → X (positional, validated against the model's declared
# buckets; vocabulary drift fails loud rather than mis-routing).
_structure_X(model, features) =
    context_from_features(model, Dict(string(k) => string(v) for (k, v) in pairs(features)))

function handle_structure_bma(params)
    names = String[string(n) for n in params["feature_names"]]
    values = Vector{String}[String[string(v) for v in vs] for vs in params["feature_values"]]
    model = try
        build_structure_model(names, values;
                              alpha0 = Float64(get(params, "alpha0", 2.0)),
                              beta0 = Float64(get(params, "beta0", 2.0)),
                              p_edge = Float64(get(params, "p_edge", 0.5)))
    catch e
        _wrap_dsl("build_structure_model failed")(e)
    end
    # Optional inline warm_counts ({contexts:[{ctx,n1,n0}]}) reconstructs the warm posterior
    # server-side in one call (symmetric with routing_init) — else the cold prior. Lets a wire
    # consumer warm-seed governance without N structure_observe round-trips.
    prior = try
        reconstruct_structure_prior_from_data(model, get(params, "warm_counts", nothing))
    catch e
        _wrap_inf("structure warm reconstruction failed")(e)
    end
    Dict{String, Any}("model_id" => register_model(model),
                      "state_id" => register_state(prior))
end

function handle_structure_observe(params)
    model = get_model(string(params["model_id"]))
    sid = string(params["state_id"])
    top = get_state(sid)
    X = try
        _structure_X(model, params["features"])
    catch e
        _wrap_dsl("structure context build failed")(e)
    end
    obs = Int(params["observation"])
    new_top = try
        structure_observe(model, top, X, obs)
    catch e
        _wrap_inf("structure_observe failed")(e)
    end
    STATE_REGISTRY[sid] = new_top
    Dict{String, Any}("state_id" => sid)
end

function handle_structure_decide(params)
    model = get_model(string(params["model_id"]))
    top = get_state(string(params["state_id"]))
    # Belief view + optional harm coordinate (a second model+belief+context for the unsafe
    # outcome). Context build is a protocol concern (vocabulary), so wrap as DSLError.
    bx, harm_belief, harm_cost = try
        bx0 = belief_at_context(model, top, _structure_X(model, params["features"]))
        hb, hc = nothing, 0.0
        if get(params, "harm", nothing) !== nothing
            h = params["harm"]
            hmodel = get_model(string(h["model_id"]))
            htop = get_state(string(h["state_id"]))
            hb = belief_at_context(hmodel, htop, _structure_X(hmodel, h["features"]))
            hc = Float64(get(h, "harm_cost", 0.0))
        end
        (bx0, hb, hc)
    catch e
        _wrap_dsl("structure decision context build failed")(e)
    end
    action = try
        decide_with_voi(bx, structure_decision_kernel();
                        cost = Float64(params["cost"]),
                        aversion = Float64(params["aversion"]),
                        interrupt_cost = Float64(params["interrupt_cost"]),
                        expected_repeats = Float64(get(params, "expected_repeats", 0.0)),
                        w_time = Float64(get(params, "w_time", 0.0)),
                        exp_time = Float64(get(params, "exp_time", 0.0)),
                        harm_belief = harm_belief, harm_cost = harm_cost)
    catch e
        _wrap_inf("decide_with_voi failed")(e)
    end
    Dict{String, Any}("action" => string(action))
end

# ═══════════════════════════════════════
# Routing verbs (decouple Move 4)
#
# A non-embedding consumer drives EU-max model routing over the wire: build the per-model
# StructureBMA belief + emission/latency latents server-side (`routing_init`, from declared
# DATA — roster, costs, reward, and the warm/latency counts INLINE), choose the EU-max model
# (`routing_decide`) or the next escalation rung (`routing_escalate`), learn from a turn's
# exec signal (`routing_outcome` — the coupled-EM confound step), and read the learned belief
# (`routing_belief`, telemetry). The whole RoutingState is ONE opaque `rt_*` handle; ALL
# arithmetic (the ProductMeasure join, the optimise argmax, the decode + EM) is engine-side
# (src/routing.jl), so Invariant 1 holds across the wire by construction.
# ═══════════════════════════════════════

# Wire dicts use string-or-symbol keys depending on the JSON reader; normalise to a plain
# String=>String features dict (as the structure verbs do) and a String=>Any profile dict.
_features(f) = Dict(string(k) => string(v) for (k, v) in pairs(f))
_profile(p) = p === nothing ? nothing : Dict(string(k) => v for (k, v) in pairs(p))

function handle_routing_init(params)
    names = String[string(n) for n in params["feature_names"]]
    values = Vector{String}[String[string(v) for v in vs] for vs in params["feature_values"]]
    roster = params["roster"]                       # [[name, provider, model_id, cost], …]
    rnames = String[]; rproviders = String[]; rids = String[]; rcosts = Float64[]
    for m in roster
        push!(rnames, string(m[1])); push!(rproviders, string(m[2]))
        push!(rids, string(m[3])); push!(rcosts, Float64(m[4]))
    end
    K = length(rids)
    rt = try
        model = build_structure_model(names, values;
                                      alpha0 = Float64(get(params, "alpha0", 2.0)),
                                      beta0 = Float64(get(params, "beta0", 2.0)),
                                      p_edge = Float64(get(params, "p_edge", 0.5)))
        tops = reconstruct_routing_tops_from_data(model, K, get(params, "warm_counts", nothing))
        emp = get(params, "emission_prior", nothing)
        emission = (emp !== nothing && length(emp) == 4) ?
            EmissionBelief((Float64(emp[1]), Float64(emp[2]), Float64(emp[3]), Float64(emp[4]))) :
            EmissionBelief()
        lat = get(params, "latency_counts", nothing)
        latency = lat === nothing ? nothing : reconstruct_latency_from_data(lat)
        RoutingState(model, tops, Dict{String, MixturePrevision}(), emission,
                     rnames, rproviders, rids, rcosts,
                     Float64(get(params, "reward", 0.02)), Float64(get(params, "w_time", 0.0)),
                     latency)
    catch e
        _wrap_dsl("routing_init failed")(e)
    end
    Dict{String, Any}("routing_state_id" => register_routing(rt), "n_models" => K)
end

function handle_routing_decide(params)
    rt = get_routing(string(params["routing_state_id"]))
    res = try
        route_decide(rt, _features(params["features"]),
                     get(params, "roster", nothing), _profile(get(params, "profile", nothing)))
    catch e
        _wrap_inf("routing_decide failed")(e)
    end
    res === nothing ? Dict{String, Any}("model" => nothing) : res   # <2 candidates ⇒ inert
end

function handle_routing_escalate(params)
    rt = get_routing(string(params["routing_state_id"]))
    rew = get(params, "reward", nothing)
    res = try
        escalate_decide(rt, _features(params["features"]), get(params, "roster", nothing),
                        [Int(t) for t in get(params, "tried", Int[])],
                        rew === nothing ? rt.reward : Float64(rew),
                        _profile(get(params, "profile", nothing)))
    catch e
        _wrap_inf("routing_escalate failed")(e)
    end
    res === nothing ? Dict{String, Any}("model" => nothing) : res   # no positive-EU rung ⇒ STOP
end

function handle_routing_outcome(params)
    id = string(params["routing_state_id"])
    rt = get_routing(id)
    human = get(params, "human", nothing)
    try
        route_outcome!(rt, string(params["model_id"]), _features(params["features"]),
                       Bool(params["success"]); human = human === nothing ? nothing : Bool(human))
    catch e
        _wrap_inf("routing_outcome failed")(e)
    end
    Dict{String, Any}("routing_state_id" => id)              # mutated in place; no log_marginal
end

# Telemetry read-back (the routing analogue of belief_at_context-as-verb): the EM-learned
# θ/ρ̄/σ̄ at a (model, context), for calibration/shadow-mode + the wire parity test. The hot
# routing_decide builds the per-context view server-side and never round-trips this. The
# actual reads live in the engine (`routing_belief_readout`); the verb only marshals.
function handle_routing_belief(params)
    rt = get_routing(string(params["routing_state_id"]))
    try
        routing_belief_readout(rt, string(params["model_id"]), _features(params["features"]))
    catch e
        _wrap_inf("routing_belief failed")(e)
    end
end

function handle_destroy_routing(params)
    delete!(ROUTING_REGISTRY, string(params["routing_state_id"]))
    Dict{String, Any}("ok" => true)
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
        # show_expr is verified total over the AST type hierarchy
        # (src/program_space/types.jl — a method per *Expr subtype); any
        # failure is a genuine bug and should surface with full context,
        # not collapse to "?". Do not re-add a defensive catch here.
        expr_str = show_expr(state.all_programs[idx].expr)
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
    kernel = try
        build_kernel(params["kernel"], id)
    catch e
        _wrap_dsl("build_kernel failed")(e)
    end
    obs = Float64(params["observation"])
    threshold = Float64(get(params, "prune_threshold", -30.0))
    max_comp = Int(get(params, "max_components", 2000))

    log_marg = try
        log_predictive(state.belief, kernel, obs)
    catch e
        _wrap_inf("log_predictive failed (state=$id, obs=$obs)")(e)
    end

    state.belief = try
        condition(state.belief, kernel, obs)
    catch e
        _wrap_inf("condition failed (state=$id, obs=$obs)")(e)
    end
    sync_prune!(state; threshold=threshold)
    sync_truncate!(state; max_components=max_comp)

    Dict{String, Any}("n_remaining" => length(state.belief.components),
                      "log_marginal" => log_marg)
end

function handle_eu_interact(params)
    id = string(params["state_id"])
    state = get_state(id)::AgentState
    features = Dict{Symbol, Float64}(Symbol(k) => Float64(v)
                                      for (k, v) in params["features"])
    rewards = Dict{Symbol, Float64}(Symbol(k) => Float64(v)
                                     for (k, v) in params["rewards"])
    # The mean_j / (1 - mean_j) split below assumes exactly two outcome
    # labels (correct-recommendation vs the single complement). Multi-
    # label support requires per-component categorical beliefs, not a
    # scalar Beta mean.
    length(rewards) == 2 || error(
        "handle_eu_interact currently supports only binary outcomes; " *
        "got $(length(rewards)) labels")
    temporal_state = Dict{Symbol, Any}()

    # Per-component recommendation (a forward program evaluation, not weight
    # arithmetic), then one FiringChoice expect per outcome label:
    #   P(label) = Σ_j w_j·(rec_j == label ? θ_j : (1-θ_j)/(n-1))
    # The per-component dispatch and the weighted mixture sum live in `expect`.
    recs = [state.compiled_kernels[j].evaluate(features, temporal_state)
            for j in eachindex(state.compiled_kernels)]
    inv = 1.0 / max(length(rewards) - 1, 1)
    when_not = LinearCombination(Tuple{Float64, TestFunction}[(-inv, Identity())], inv)  # (1-θ)/(n-1)
    label_probs = Dict{Symbol, Float64}()
    for label in keys(rewards)
        label_probs[label] = expect(state.belief,
            FiringChoice(Bool[r == label for r in recs], Identity(), when_not))
    end

    # EU as a single canalised expectation. Binary outcome (asserted above): per
    # component EU is affine in θ_j — a program recommending L1 scores r1·θ + r2·(1-θ),
    # one recommending L2 scores r1·(1-θ) + r2·θ. FiringChoice picks the branch and
    # `expect` does the weighted mixture sum, so no arithmetic on probabilities here.
    labels = collect(keys(rewards)); L1, L2 = labels[1], labels[2]
    r1, r2 = rewards[L1], rewards[L2]
    eu = expect(state.belief, FiringChoice(Bool[r == L1 for r in recs],
        LinearCombination(Tuple{Float64, TestFunction}[(r1 - r2, Identity())], r2),
        LinearCombination(Tuple{Float64, TestFunction}[(r2 - r1, Identity())], r1)))
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
    log_msg("Credence skin server starting...")

    # Startup-complete sentinel (issue #22). All `using Credence` / module
    # loading above has completed; emit a JSON line on stdout so the
    # Python client can detect readiness with `process.poll()` safeguard
    # instead of relying on a fixed timeout that misfires under cold-
    # compile + loaded-runner variance. The matching read lives in
    # `apps/skin/client.py::SkinClient._wait_for_ready()`.
    println(stdout, JSON3.write(Dict("status" => "ready")))
    flush(stdout)

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
            elseif e isa MethodNotFound
                -32601
            elseif e isa ProtocolMismatch
                -32010
            elseif e isa DSLError
                -32001
            elseif e isa InferenceError
                -32002
            else
                -32603
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

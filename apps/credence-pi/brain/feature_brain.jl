# Role: brain (thin shim)
"""
    feature_brain.jl — credence-pi's thin domain shim over the engine's structure-BMA stdlib.

After decouple Move 3 the structure-BMA builder + observe + readout live in the engine
(`src/structure_bma.jl`: `build_structure_model` / `build_structure_prior` /
`structure_observe` / `belief_at_context` / …). This module re-exports them under the
names credence-pi's daemon and tests use, and keeps the genuinely credence-pi-specific
wiring: reading the declared feature env, the harm/tail/latency opt-in plumbing, the
per-context EU decision (`decide`/`decide_multi`, which select belief views and delegate
the EU math to the engine's `decide_with_voi`), the counts-JSON reconstruction, the
effector policy, and `wire_brain!`. It reimplements NO axiom op and assembles NO EU
coefficients — every belief change is `condition` (inside the lifted `structure_observe`);
every decision delegates to the engine's `decide_with_voi` (which maximises through the
canonical typed `optimise`). The `.bdsl` files keep the DECLARED DATA (feature spaces,
capabilities, utility constants).
"""
module FeatureBrain

using Main.Credence: StructureBMA, build_structure_model, build_structure_prior,
    build_structure_prior_dense, structure_observe, structure_observe_soft,
    structure_firing_tags, belief_at_context, context_from_features, structure_decision_kernel,
    MixturePrevision, GammaPrevision, Identity,
    GeometricTail, FeatureDecl, Finite, expect
# The EU-max template lives in the engine stdlib (decouple Move 3): the app ships utility
# scalars and SELECTS belief views, it does not assemble coefficients. No `optimise`/
# `net_voi`/`LinearCombination`/`Projection`/`ProductMeasure` import remains — a grep of
# this shim for those primitives is now empty, the leak-closed witness.
import Main.Credence.Ontology: decide_with_voi
using JSON3

export StructureBMA, build_model, build_model_from_env, build_model_from_decls,
       build_prior, build_prior_dense, wire_brain!, context_from_features, firing_tags,
       belief_at_context, observe, observe_soft, decide, decide_multi,
       reconstruct_posterior, reconstruct_harm_posterior

# ── The structure-BMA core now lives in the engine; alias the names this app's daemon
#    and tests use to the lifted `Credence` functions (decouple Move 3). ──
const build_model       = build_structure_model
const build_prior       = build_structure_prior
const build_prior_dense = build_structure_prior_dense
const observe           = structure_observe
const observe_soft      = structure_observe_soft
const firing_tags       = structure_firing_tags

# ── Declared-feature → model construction (env/eval-layer plumbing — stays consumer-side:
#    `FeatureDecl` is an eval-layer type included after the engine's Ontology module, and
#    reading a BDSL env is body wiring, not engine machinery). ──

"""
    build_model_from_env(env; ...) -> StructureBMA

Read the declared feature spaces from `env[:features]` (a `Vector{FeatureDecl}`
populated by features.bdsl) and build the typed structure family.
"""
function build_model_from_env(env; alpha0::Float64 = 2.0, beta0::Float64 = 2.0,
                              p_edge::Float64 = 0.5)
    decls = get(env, Symbol("features"), nothing)
    decls isa AbstractVector || error("feature brain: env[:features] must be a list of FeatureDecl")
    build_model_from_decls(decls; alpha0 = alpha0, beta0 = beta0, p_edge = p_edge)
end

"""
    build_model_from_decls(decls; ...) -> StructureBMA

Build a model from a `Vector{FeatureDecl}` (declared DATA). Shared by the waste model
(`env[:features]`) and the harm model (`env[:safety-features]`).
"""
function build_model_from_decls(decls; alpha0::Float64 = 2.0, beta0::Float64 = 2.0,
                                p_edge::Float64 = 0.5)
    decls isa AbstractVector || error("feature brain: expected a list of FeatureDecl")
    names = String[]
    vals = Vector{String}[]
    for d in decls
        d isa FeatureDecl || error("feature brain: expected FeatureDecl, got $(typeof(d))")
        d.space isa Finite ||
            error("feature brain: feature '$(d.name)' must have a Finite space, got $(typeof(d.space))")
        push!(names, string(d.name))
        push!(vals, String[string(v) for v in d.space.values])
    end
    build_model(names, vals; alpha0 = alpha0, beta0 = beta0, p_edge = p_edge)
end

# Are all of a model's declared features present in a body-sent feature dict? (Used to
# decide whether the harm posterior can be consulted for this event; if the body has not
# yet been upgraded to emit safety features, harm governance stays off — backward-compat.)
function _has_features(model::StructureBMA, features)
    fd = Dict{String, String}(string(k) => string(v) for (k, v) in features)
    all(haskey(fd, n) for n in model.feature_names)
end

# ── Decision: per-context EU-max via the engine's `decide_with_voi` template ──
#
# The proceed/block/ask EU math — tail-aware linear-cost waste utility, optional harm
# coordinate, VOI ask-gate — lives in the engine stdlib (`decide_with_voi`, src/stdlib.jl)
# so it is byte-identical in-process and over the skin wire. The app's job here is only to
# SELECT the per-context belief view(s) and pass the utility scalars; it assembles no
# coefficients and calls no `optimise`/`net_voi` itself (the decouple Move-3 leak-close).
# θ = P(approve|X); c = per-call cost; λ = false-block aversion; q = interrupt cost;
# m = expected_repeats (the multi-turn look-ahead, `expect(continuation, GeometricTail())`,
# m=0 ⇒ myopic). The coefficient algebra and its block-cutoff proof now live with the math.

# Per-request profile override (the user's utility weights, shipped by the body PER REQUEST so a
# user switches their trade-off with no daemon restart): the value for `key` if present, else the
# wired default. Preference DATA from the body; the brain still does all the EU-max.
_pget(profile, key::AbstractString, default::Float64)::Float64 =
    (profile isa AbstractDict && haskey(profile, key) && profile[key] !== nothing) ?
        Float64(profile[key]) : default

"""
    decide(model, top, X, cost; aversion, interrupt_cost, expected_repeats=0.0,
           w_time=0.0, exp_time=0.0) -> Symbol

Return `:proceed`, `:block`, or `:ask`. Selects the structure-BMA per-context belief
view and delegates the EU-max to the engine's single-outcome `decide_with_voi` (no harm
coordinate). `cost` is the per-call USD estimate; `aversion` is λ; `interrupt_cost` is q;
`expected_repeats` is m — the tail a block prevents (0 ⇒ myopic; supplied by the tail
brain as `expect(continuation_belief, GeometricTail())`).
"""
function decide(model::StructureBMA, top::MixturePrevision, X::AbstractVector, cost::Float64;
                aversion::Float64, interrupt_cost::Float64, expected_repeats::Float64 = 0.0,
                w_time::Float64 = 0.0, exp_time::Float64 = 0.0)
    bx = belief_at_context(model, top, X)
    decide_with_voi(bx, structure_decision_kernel(); cost = cost, aversion = aversion,
                    interrupt_cost = interrupt_cost, expected_repeats = expected_repeats,
                    w_time = w_time, exp_time = exp_time)
end

# ── Multi-outcome decision: one EU integrating waste AND harm in one currency ──
#
# Two posteriors at decision time: θ_a = P(approve|Xw) (waste brain, live-learned) and
# θ_u = P(unsafe|Xh) (harm brain, warm-frozen). A block avoids the waste TAIL ((1+m) calls)
# and the harm (H·θ_u, one-shot). The waste cutoff θ_a < (1+m)/((1+m)+λ) + H·θ_u/(c·[(1+m)+λ])
# SLIDES with BOTH the harm belief and the detected tail — no OR-of-thresholds can express it
# (eval/multi_outcome.jl + REGEX_IMPOSSIBLE.md). The joint-belief EU assembly lives in the
# engine's `decide_with_voi` (harm coordinate); here we only select the two views and pass H.
"""
    decide_multi(waste_model, top_waste, harm_model, harm_top, Xw, Xh, cost;
                 aversion, interrupt_cost, harm_cost, expected_repeats=0.0,
                 w_time=0.0, exp_time=0.0) -> Symbol

Multi-outcome EU decision over the joint of the approve-belief and the unsafe-belief.
Delegates to the engine's `decide_with_voi` with the harm coordinate (`harm_belief`,
`harm_cost = H`). `harm_cost = 0` AND `expected_repeats = 0` reduce it to the
single-outcome `decide`. `expected_repeats` (m) scales only the waste tail.
"""
function decide_multi(waste_model::StructureBMA, top_waste::MixturePrevision,
                      harm_model::StructureBMA, harm_top::MixturePrevision,
                      Xw::AbstractVector, Xh::AbstractVector, cost::Float64;
                      aversion::Float64, interrupt_cost::Float64, harm_cost::Float64,
                      expected_repeats::Float64 = 0.0,
                      w_time::Float64 = 0.0, exp_time::Float64 = 0.0)
    bxa = belief_at_context(waste_model, top_waste, Xw)   # P(approve|Xw)
    bxu = belief_at_context(harm_model, harm_top, Xh)     # P(unsafe|Xh)
    decide_with_voi(bxa, structure_decision_kernel(); cost = cost, aversion = aversion,
                    interrupt_cost = interrupt_cost, expected_repeats = expected_repeats,
                    w_time = w_time, exp_time = exp_time, harm_belief = bxu, harm_cost = harm_cost)
end

# ── harm posterior persistence: version-stable per-context COUNTS ──
#
# Bayesian updating is order-independent, so the final posterior depends only on the per-
# context counts (n1 = harm-labelled, n0 = safe-labelled). We ship those as JSON and
# reconstruct by replaying `observe` — robust across Julia versions (unlike Serialization).
# The JSON shape: { "contexts": [ { "ctx": ["external-send","tainted-external-target","yes","no"],
#                                   "n1": 12, "n0": 1 }, ... ], ...metadata... }.
"""
    reconstruct_posterior(model, counts_path) -> MixturePrevision

Rebuild a frozen posterior from shipped per-context counts by replaying `observe`.
Bayesian updating is order-independent, so the posterior depends only on each
context's (n1, n0) counts — making this artifact version-stable (JSON), unlike a
`Serialization` blob (fragile across Julia versions; CI/image pin 1.11). Generic
over the StructureBMA: the harm posterior (P(unsafe|safety-features)) and the warm
WASTE posterior (P(approve|waste-features)) both reconstruct through this one path.
"""
function reconstruct_posterior(model::StructureBMA, counts_path::AbstractString)
    data = JSON3.read(read(counts_path, String))
    entries = data.contexts
    top = build_prior(model)
    for e in entries
        ctx = String[String(v) for v in e.ctx]
        for _ in 1:Int(e.n1); top = observe(model, top, ctx, 1); end
        for _ in 1:Int(e.n0); top = observe(model, top, ctx, 0); end
    end
    top
end

# Back-compat alias: the harm posterior was the first consumer of this path.
const reconstruct_harm_posterior = reconstruct_posterior

# Pass-1 followup logic, kept (yes→proceed, no→block); deterministic in Move 3.
function followup_after_response(event)
    resp = string(get(event, "response", ""))
    resp == "yes" ? :proceed : resp == "no" ? :block : :nothing
end

# ── Wiring: inject the brain closures into the BDSL env ──
#
# The daemon's call-sites are unchanged in SHAPE — it still does
# `env[:make-prior]()`, `env[:decide-action](...)`, `env[:observe-response](...)`,
# `env[:followup-after-response](event)` — but those symbols now resolve to Julia
# brain functions. Utility constants are declared DATA read from the env
# (utility.bdsl); defaults keep the brain runnable without that file.

function wire_brain!(env)
    p_edge   = Float64(get(env, Symbol("edge-inclusion-prior"), 0.5))
    λ        = Float64(get(env, Symbol("false-block-aversion"), 1.0))
    q        = Float64(get(env, Symbol("interrupt-cost"), 0.02))
    fallback = Float64(get(env, Symbol("fallback-call-cost"), 0.5))
    a0       = Float64(get(env, Symbol("cell-prior-alpha"), 2.0))
    b0       = Float64(get(env, Symbol("cell-prior-beta"), 2.0))

    H        = Float64(get(env, Symbol("harm-cost"), 0.0))
    hresp    = Symbol(get(env, Symbol("harm-response"), "ask"))  # :ask (confirm) | :block (enforce)
    wtime    = Float64(get(env, Symbol("w-time"), 0.0))          # TIME coordinate: $/sec of wall-clock

    model = build_model_from_env(env; alpha0 = a0, beta0 = b0, p_edge = p_edge)

    # Optional harm posterior (multi-outcome governance). ACTIVE only when the operator
    # declared `safety-features`, shipped a trained harm posterior (`harm-brain-path`),
    # AND set `harm-cost > 0`. Any missing piece (or a load failure) leaves decide-action
    # as exactly the single-outcome waste path — backward-compatible, fail-loud-then-off.
    harm_model = nothing
    harm_top = nothing
    sfeat = get(env, Symbol("safety-features"), nothing)
    # Default to the harm posterior shipped next to this brain file; an operator may
    # override with a `harm-brain-path` define. INERT until harm-cost > 0. The artifact is
    # version-stable per-context COUNTS (JSON), reconstructed via `observe` — NOT a Julia
    # Serialization blob, which is fragile across Julia versions (CI/image pin 1.11).
    hpath = get(env, Symbol("harm-brain-path"), joinpath(@__DIR__, "harm_brain.counts.json"))
    if H > 0.0 && sfeat isa AbstractVector && hpath !== nothing && !isempty(string(hpath))
        if isfile(string(hpath))
            try
                harm_model = build_model_from_decls(sfeat; alpha0 = a0, beta0 = b0, p_edge = p_edge)
                harm_top = reconstruct_posterior(harm_model, string(hpath))
                @info "credence-pi: harm posterior reconstructed; multi-outcome governance ON" path=string(hpath)
            catch e
                @warn "credence-pi: harm posterior failed to load; harm governance OFF" error=e
                harm_model = nothing
                harm_top = nothing
            end
        else
            @warn "credence-pi: harm-brain-path set but file missing; harm governance OFF" path=string(hpath)
        end
    end

    # Optional tail belief (multi-turn look-ahead) — OPT-IN via (define tail-aware "on").
    # The continuation posterior P(another identical call follows | features) uses the SAME
    # feature model as the waste brain, trained on loop continue/stop events
    # (eval/train_tail_brain.jl). When on, the per-call expected remaining repeats
    # m = expect(belief, GeometricTail()) scales the waste tail in decide/decide_multi (block
    # a long loop earlier; ask about an uncertain one); off ⇒ m=0, the myopic per-call EU.
    # OFF by default: it preserves the calibration-friendly cold-start (at θ≈0.5 we ASK to
    # learn, not pre-emptively block on a continuation prior), and on the WARM brain it
    # changes 0 ClawsBench decisions (high-m contexts already have low θ) — it bites in the
    # uncertain-persistent regime (test_feature_brain.jl §8; eval/tail_lookahead.jl). Version-
    # stable COUNTS JSON, reconstructed via `observe` like the warm/harm posteriors.
    tail_top = nothing
    if string(get(env, Symbol("tail-aware"), "off")) == "on"
        tpath = get(env, Symbol("tail-brain-path"), joinpath(@__DIR__, "tail_brain.counts.json"))
        if tpath !== nothing && !isempty(string(tpath)) && isfile(string(tpath))
            try
                tail_top = reconstruct_posterior(model, string(tpath))
                @info "credence-pi: tail (continuation) posterior ON; multi-turn look-ahead governance" path=string(tpath)
            catch e
                @warn "credence-pi: tail posterior failed to load; look-ahead OFF (myopic m=0)" error=e
                tail_top = nothing
            end
        else
            @warn "credence-pi: tail-aware on but tail brain missing; look-ahead OFF (myopic m=0)" path=string(tpath)
        end
    end

    # Expected remaining identical repeats a block prevents, per context: the closed-form
    # geometric-tail mean of the continuation posterior (m=0 with no tail brain ⇒ myopic EU).
    # This is the multi-turn look-ahead, computed by `expect` of a declared Functional.
    m_at(X) = tail_top === nothing ? 0.0 :
              expect(belief_at_context(model, tail_top, X), GeometricTail())

    # Optional governance LATENCY belief (TIME coordinate) — OPT-IN by file presence, like the
    # tail belief. Per-context Poisson-Gamma E[turns|X] (read by `expect`=α/β) × measured s̄/turn
    # ⇒ E[time|X] seconds a block would save. Absent ⇒ exp_time=0 ⇒ time-blind governance
    # (bit-identical). Version-stable counts-JSON (sufficient statistic), reconstructed exactly.
    gov_latency = nothing
    lpath = get(env, Symbol("governance-latency-path"), joinpath(@__DIR__, "governance_latency.counts.json"))
    if lpath !== nothing && !isempty(string(lpath)) && isfile(string(lpath))
        try
            data = JSON3.read(read(string(lpath), String))
            α0, β0 = Float64(data["turns_prior"][1]), Float64(data["turns_prior"][2])
            rate = Float64(data["rate_s"])
            tm = Dict{String, Float64}()
            for ctx in data["contexts"]
                g = GammaPrevision(α0 + Float64(ctx["sum_turns"]), β0 + Float64(ctx["n_obs"]))
                tm[join(string.(ctx["ctx"]), "|")] = Float64(expect(g, Identity())) * rate
            end
            gov_latency = tm
            @info "credence-pi: governance latency belief ON (time coordinate)" path=string(lpath)
        catch e
            @warn "credence-pi: governance latency failed to load; time-blind" error=e
            gov_latency = nothing
        end
    end
    time_at(X) = gov_latency === nothing ? 0.0 : get(gov_latency, join(string.(X), "|"), 0.0)

    env[Symbol("make-prior")] = () -> build_prior(model)

    env[Symbol("decide-action")] = (top, features, cost, profile = nothing) -> begin
        c = cost === nothing ? fallback : Float64(cost)
        c = c <= 0.0 ? fallback : c
        # Per-request profile override (the user's utility weights, no daemon restart): λ/q/H/w_time
        # for THIS decision only; absent ⇒ the wired defaults. Beliefs untouched.
        λc = _pget(profile, "lambda", λ); qc = _pget(profile, "q", q)
        Hc = _pget(profile, "harm", H); wt = _pget(profile, "w_time", wtime)
        # Multi-outcome only when harm is active AND the body sent the safety features for
        # this event; otherwise the single-outcome waste decision (unchanged).
        if harm_model !== nothing && harm_top !== nothing && _has_features(harm_model, features)
            Xw = context_from_features(model, features)
            Xh = context_from_features(harm_model, features)
            m = m_at(Xw)
            d = decide_multi(model, top, harm_model, harm_top, Xw, Xh, c;
                             aversion = λc, interrupt_cost = qc, harm_cost = Hc, expected_repeats = m,
                             w_time = wt, exp_time = time_at(Xw))
            # Research-stage effector policy (harm-response = :ask): a harm-DRIVEN stop is a
            # CONFIRMATION, not a refusal — the harm belief is benchmark-seeded, not yet
            # user-calibrated, so asking has value and the response is the calibration
            # signal we learn from. Waste-driven blocks are unchanged (waste is proven).
            # Like shadowMode, this is an effector policy, not a change to the EU reasoning:
            # the harm term decided "do not proceed"; :ask realises that as "confirm".
            if d === :block && hresp === :ask
                d_waste = decide(model, top, Xw, c; aversion = λc, interrupt_cost = qc,
                                 expected_repeats = m, w_time = wt, exp_time = time_at(Xw))
                d = d_waste === :block ? :block : :ask   # harm was the driver ⇒ confirm
            end
            d
        else
            X = context_from_features(model, features)
            decide(model, top, X, c; aversion = λc, interrupt_cost = qc, expected_repeats = m_at(X),
                   w_time = wt, exp_time = time_at(X))
        end
    end

    env[Symbol("observe-response")] = (top, features, obs) -> begin
        X = context_from_features(model, features)
        observe(model, top, X, obs)
    end

    env[Symbol("followup-after-response")] = followup_after_response

    model
end

end # module FeatureBrain

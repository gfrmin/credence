# Role: brain-side application
"""
    host.jl — Host driver for RSS article ranking via program-space inference.

    Observation model: Plackett-Luce. Each reading event is a categorical
    observation — "user chose article A from all unread articles S."
    Dismiss events are negative Bernoulli observations.

    The ranking is the prediction. The utility is the Plackett-Luce
    log-likelihood — a proper scoring rule. Conditioning maximises it.

    Tier 3: RSS-domain-specific. Uses Tier 1 (Credence DSL) and Tier 2
    (ProgramSpace) for domain-independent inference machinery.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
using Credence
using Credence: expect, condition, draw, weights, mean, logsumexp
using Credence: CategoricalMeasure, BetaPrevision, TaggedBetaMeasure, MixtureMeasure
using Credence: Finite, Interval, Kernel, Measure
using Credence: FiringByTag, BetaBernoulli, Flat
using Credence: AgentState, sync_prune!, sync_truncate!
using Credence: Grammar, Program, CompiledKernel, ProductionRule
using Credence: enumerate_programs, compile_kernel
using Credence: aggregate_grammar_weights, top_k_grammar_ids, add_programs_to_state!
using Credence: next_grammar_id

include("features.jl")
include("terminals.jl")
include("preferences.jl")

using Random
using Dates

# ═══════════════════════════════════════
# Initialization
# ═══════════════════════════════════════

"""
    init_rss_agent(feeds, categories, known_tags; ...) → (AgentState, FeatureRegistry)

Build the initial agent state: feature registry, seed grammars, enumerated
programs, flat MixtureMeasure of TaggedBetaMeasures.
"""
function init_rss_agent(;
    feeds::Vector{Tuple{Int, String}},
    categories::Vector{String},
    known_tags::Vector{String},
    program_max_depth::Int = 3,
    min_log_prior::Float64 = -20.0,
    verbose::Bool = true,
)
    reg = build_feature_registry(feeds, categories, known_tags)
    grammar_pool = generate_rss_seed_grammars(reg)

    verbose && println("Generated $(length(grammar_pool)) seed grammars, $(length(reg.feature_set)) features")

    components = Measure[]
    log_prior_weights = Float64[]
    metadata = Tuple{Int, Int}[]
    compiled_kernels = CompiledKernel[]
    all_programs = Program[]

    idx = 0
    for g in grammar_pool
        programs = enumerate_programs(g, program_max_depth;
                                       action_space = RSS_ACTION_SPACE,
                                       min_log_prior = min_log_prior)
        for (pi, p) in enumerate(programs)
            idx += 1
            push!(components, TaggedBetaMeasure(Interval(0.0, 1.0), idx, BetaPrevision(1.0, 1.0)))
            lw = -g.complexity * log(2) - p.complexity * log(2)
            push!(log_prior_weights, lw)
            push!(metadata, (g.id, pi))
            push!(compiled_kernels, compile_kernel(p, g, pi))
            push!(all_programs, p)
        end
    end

    verbose && println("Total programs: $(length(components))")

    belief = MixtureMeasure(Interval(0.0, 1.0), components, log_prior_weights)
    grammar_dict = Dict{Int, Grammar}(g.id => g for g in grammar_pool)
    state = AgentState(belief, metadata, compiled_kernels, all_programs,
                       grammar_dict, program_max_depth)

    (state, reg)
end

# ═══════════════════════════════════════
# Kernel construction
# ═══════════════════════════════════════

"""
    build_read_kernel(compiled_kernels, chosen_features, all_unread_features, temporal_state) → Kernel

Plackett-Luce kernel for a reading event. The user chose one article from
the full unread set. Programs that fire on the chosen article (and few others)
get the highest likelihood.

Always condition with obs=1.0.
"""
function build_read_kernel(
    compiled_kernels::Vector{CompiledKernel},
    chosen_features::Dict{Symbol, Float64},
    all_unread_features::Vector{Dict{Symbol, Float64}},
    temporal_state::Dict{Symbol, Any},
)
    obs_space = Finite([0.0, 1.0])
    n_articles = length(all_unread_features)

    # Eagerly compute firing set on the chosen article: programs whose predicate
    # matches here are the ones whose θ is evidenced by this observation. Used
    # both for the FiringByTag likelihood-family dispatch and for the
    # Plackett-Luce numerator check in the generator.
    fires_chosen_set = Set{Int}(
        i for (i, ck) in enumerate(compiled_kernels)
        if ck.evaluate(chosen_features, temporal_state) == :match
    )
    fire_counts_cache = Dict{Int, Int}()

    Kernel(Interval(0.0, 1.0), obs_space,
        _ -> error("generate not used"),
        (m_or_θ, obs) -> begin
            if m_or_θ isa TaggedBetaMeasure
                tag = m_or_θ.tag
                p = mean(m_or_θ.beta)
                ck = compiled_kernels[tag]

                fires_chosen = tag in fires_chosen_set

                n_fires = get!(fire_counts_cache, tag) do
                    count(f -> ck.evaluate(f, temporal_state) == :match, all_unread_features)
                end

                log_score_chosen = fires_chosen ? log(max(p, 1e-300)) : log(0.5)
                log_denom = logsumexp([
                    n_fires > 0 ? log(max(p, 1e-300)) + log(n_fires) : -Inf,
                    (n_articles - n_fires) > 0 ? log(0.5) + log(n_articles - n_fires) : -Inf,
                ])

                log_score_chosen - log_denom
            else
                log(max(m_or_θ, 1e-300)) - log(max(n_articles * 0.5, 1e-300))
            end
        end;
        likelihood_family = FiringByTag(fires_chosen_set, BetaBernoulli(), Flat()))
end

"""
    build_dismiss_kernel(compiled_kernels, dismissed_features, temporal_state) → Kernel

Kernel for a dismiss event (user marked article as read without opening).
Programs that fire on the dismissed article lose weight.

Always condition with obs=0.0.
"""
function build_dismiss_kernel(
    compiled_kernels::Vector{CompiledKernel},
    dismissed_features::Dict{Symbol, Float64},
    temporal_state::Dict{Symbol, Any},
)
    obs_space = Finite([0.0, 1.0])

    fires_set = Set{Int}(
        i for (i, ck) in enumerate(compiled_kernels)
        if ck.evaluate(dismissed_features, temporal_state) == :match
    )

    Kernel(Interval(0.0, 1.0), obs_space,
        _ -> error("generate not used"),
        (m_or_θ, obs) -> begin
            if m_or_θ isa TaggedBetaMeasure
                tag = m_or_θ.tag
                fires = tag in fires_set
                p = mean(m_or_θ.beta)
                fires ? log(max(1.0 - p, 1e-300)) : log(0.5)  # credence-lint: allow — precedent:declarative-construction — Kernel log-density closure: FiringByTag likelihood from Beta mean
            else
                log(max(1.0 - m_or_θ, 1e-300))
            end
        end;
        likelihood_family = FiringByTag(fires_set, BetaBernoulli(), Flat()))
end

# ═══════════════════════════════════════
# Session processing
# ═══════════════════════════════════════

struct ReadEvent
    entry_id::Int
    read_at::DateTime
end

struct DismissEvent
    entry_id::Int
end

"""
    process_session!(state, read_events, dismiss_events, article_features, temporal_state)

Process a batch of reading and dismiss events. Read events are processed
in chronological order (reading order IS the signal). Each read shrinks
the unread set for subsequent conditioning steps.
"""
function process_session!(
    state::AgentState,
    read_events::Vector{ReadEvent},
    dismiss_events::Vector{DismissEvent},
    article_features::Dict{Int, Dict{Symbol, Float64}},
    temporal_state::Dict{Symbol, Any} = Dict{Symbol, Any}();
    verbose::Bool = false,
)
    # Sort reads chronologically — reading order is the signal
    sorted_reads = sort(read_events, by = e -> e.read_at)

    # Track which articles are still in the unread set
    unread_ids = Set(keys(article_features))

    # Process reads sequentially (Plackett-Luce decomposition)
    for (i, event) in enumerate(sorted_reads)
        event.entry_id in unread_ids || continue

        chosen_features = article_features[event.entry_id]
        unread_feature_list = [article_features[id] for id in unread_ids]

        k = build_read_kernel(state.compiled_kernels, chosen_features,
                               unread_feature_list, temporal_state)
        state.belief = condition(state.belief, k, 1.0)

        delete!(unread_ids, event.entry_id)
        sync_prune!(state; threshold = -30.0)
        sync_truncate!(state; max_components = 2000)

        verbose && println("  Read $i/$(length(sorted_reads)): entry $(event.entry_id), $(length(state.belief.components)) components")
    end

    # Process dismissals (order doesn't matter)
    for event in dismiss_events
        haskey(article_features, event.entry_id) || continue

        k = build_dismiss_kernel(state.compiled_kernels,
                                  article_features[event.entry_id], temporal_state)
        state.belief = condition(state.belief, k, 0.0)

        sync_prune!(state; threshold = -30.0)
        sync_truncate!(state; max_components = 2000)

        verbose && println("  Dismiss: entry $(event.entry_id), $(length(state.belief.components)) components")
    end

    state
end

# ═══════════════════════════════════════
# Ranking
# ═══════════════════════════════════════

"""
    rank_articles(state, article_features, temporal_state; method) → Vector{Tuple{Int, Float64}}

Rank articles by posterior expected interestingness. Returns
[(entry_id, score), ...] sorted descending.
"""
function rank_articles(
    state::AgentState,
    article_features::Dict{Int, Dict{Symbol, Float64}},
    temporal_state::Dict{Symbol, Any} = Dict{Symbol, Any}();
    method::Symbol = :expected_score,
)
    if method == :thompson
        return _rank_thompson(state, article_features, temporal_state)
    end

    w = weights(state.belief)
    scores = Dict{Int, Float64}()

    for (entry_id, features) in article_features
        score = 0.0
        for (j, comp) in enumerate(state.belief.components)
            tbm = comp::TaggedBetaMeasure
            fires = state.compiled_kernels[j].evaluate(features, temporal_state) == :match
            p = mean(tbm.beta)
            score += w[j] * (fires ? p : 0.5)  # credence-lint: allow — precedent:posterior-iteration — mixture EU requires per-component compiled-kernel dispatch
        end
        scores[entry_id] = score
    end

    sort(collect(scores), by = x -> -x[2])
end

function _rank_thompson(
    state::AgentState,
    article_features::Dict{Int, Dict{Symbol, Float64}},
    temporal_state::Dict{Symbol, Any},
)
    # Draw a program from the posterior
    drawn = draw(state.belief)
    tag = drawn.tag
    ck = state.compiled_kernels[tag]
    drawn_p = draw(drawn.beta)

    scores = Dict{Int, Float64}()
    for (entry_id, features) in article_features
        fires = ck.evaluate(features, temporal_state) == :match
        scores[entry_id] = fires ? drawn_p : 0.5
    end

    sort(collect(scores), by = x -> -x[2])
end

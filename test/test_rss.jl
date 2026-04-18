#!/usr/bin/env julia
"""
    test_rss.jl — Tests for RSS article ranking domain.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "apps", "julia", "rss", "host.jl"))

using Dates
using Random

passed = 0
failed = 0
macro check(name, expr)
    quote
        try
            if $(esc(expr))
                global passed += 1
            else
                global failed += 1
                println("FAILED: ", $(esc(name)))
            end
        catch e
            global failed += 1
            println("ERROR: ", $(esc(name)), " — ", e)
        end
    end
end

println("=" ^ 60)
println("TEST 1: Feature extraction")
println("=" ^ 60)

feeds = [(1, "Hacker News"), (2, "Ars Technica"), (3, "Low Quality Blog")]
cats = ["tech", "news"]
tags = ["programming", "ai", "linux"]

reg = build_feature_registry(feeds, cats, tags)
@check "feature set includes fixed features" :feed_priority in reg.feature_set
@check "feature set includes feed features" :feed_is_hacker_news in reg.feature_set
@check "feature set includes tag features" :tag_programming in reg.feature_set
@check "feature set includes category features" :cat_is_tech in reg.feature_set

article = Article(1, 1, "Hacker News", "tech", "Test Article", "https://example.com",
                  DateTime(2026, 4, 1, 14, 30, 0), 500, true, false, false, 3,
                  ["programming", "ai"], 1)

now_time = DateTime(2026, 4, 2, 10, 0, 0)
f = extract_features(article, reg; now=now_time)

@check "all features in [0,1]" all(v -> 0.0 <= v <= 1.0, values(f))
@check "feed_priority=1 → 1.0" f[:feed_priority] == 1.0
@check "has_code=true → 1.0" f[:has_code] == 1.0
@check "has_images=false → 0.0" f[:has_images] == 0.0
@check "feed one-hot correct" f[:feed_is_hacker_news] == 1.0
@check "feed one-hot other=0" f[:feed_is_ars_technica] == 0.0
@check "tag present → 1.0" f[:tag_programming] == 1.0
@check "tag absent → 0.0" f[:tag_linux] == 0.0
@check "category one-hot" f[:cat_is_tech] == 1.0

println("  Features extracted: $(length(f))")
println()

println("=" ^ 60)
println("TEST 2: Seed grammar enumeration")
println("=" ^ 60)

grammars = generate_rss_seed_grammars(reg)
@check "grammars generated" length(grammars) >= 8

global total_programs = 0
for g in grammars
    programs = enumerate_programs(g, 3; action_space=RSS_ACTION_SPACE, min_log_prior=-20.0)
    for (pi, p) in enumerate(programs)
        ck = compile_kernel(p, g, pi)
        result = ck.evaluate(f, Dict{Symbol, Any}())
        @check "program returns :match or :no_match" result in (:match, :no_match)
        global total_programs += 1
    end
end
println("  Total programs across all grammars: $total_programs")
@check "substantial program count" total_programs >= 50
println()

println("=" ^ 60)
println("TEST 3: Plackett-Luce kernel correctness")
println("=" ^ 60)

# Set up: 3 articles. A fires on our test program, B and C don't.
article_a = Article(10, 1, "Hacker News", "tech", "AI on HN", "https://a.com",
                    DateTime(2026, 4, 1, 10, 0, 0), 300, true, false, false, 2,
                    ["programming"], 1)
article_b = Article(20, 3, "Low Quality Blog", "news", "Random", "https://b.com",
                    DateTime(2026, 4, 1, 10, 0, 0), 100, false, false, false, 0,
                    ["gaming"], 3)
article_c = Article(30, 2, "Ars Technica", "news", "Another", "https://c.com",
                    DateTime(2026, 4, 1, 10, 0, 0), 200, false, true, false, 1,
                    [], 2)

fa = extract_features(article_a, reg; now=now_time)
fb = extract_features(article_b, reg; now=now_time)
fc = extract_features(article_c, reg; now=now_time)

# Initialize agent (depth 2 for faster kernel tests)
state, _ = init_rss_agent(feeds=feeds, categories=cats, known_tags=tags,
                           program_max_depth=2, verbose=false)

# Build read kernel: user chose article A from {A, B, C}
all_features = [fa, fb, fc]
k_read = build_read_kernel(state.compiled_kernels, fa, all_features, Dict{Symbol, Any}())

# Find programs that fire on A but not B, and vice versa
global discriminating_idx = nothing
global wrong_idx = nothing
ts = Dict{Symbol, Any}()
for (j, ck) in enumerate(state.compiled_kernels)
    fires_a = ck.evaluate(fa, ts) == :match
    fires_b = ck.evaluate(fb, ts) == :match
    if fires_a && !fires_b && discriminating_idx === nothing
        global discriminating_idx = j
    end
    if !fires_a && fires_b && wrong_idx === nothing
        global wrong_idx = j
    end
    discriminating_idx !== nothing && wrong_idx !== nothing && break
end

# With Beta(1,1), mean=0.5=base_rate → no discrimination. Use a reliable Beta
# to test that Plackett-Luce correctly favours programs that fire on the chosen article.
if discriminating_idx !== nothing && wrong_idx !== nothing
    reliable_disc = TaggedBetaMeasure(Interval(0.0, 1.0), discriminating_idx, BetaMeasure(5.0, 1.0))
    reliable_wrong = TaggedBetaMeasure(Interval(0.0, 1.0), wrong_idx, BetaMeasure(5.0, 1.0))

    ll_disc = k_read.log_density(reliable_disc, 1.0)
    ll_wrong = k_read.log_density(reliable_wrong, 1.0)

    @check "log-density is finite and negative" isfinite(ll_disc) && ll_disc < 0
    @check "discriminating program has higher ll than wrong program" ll_disc > ll_wrong
    println("  Discriminating (fires on A) ll: $(round(ll_disc, digits=4))")
    println("  Wrong (fires on B not A) ll: $(round(ll_wrong, digits=4))")
else
    println("  WARNING: need both discriminating and wrong programs — test inconclusive")
end
println()

println("=" ^ 60)
println("TEST 4: Dismiss kernel correctness")
println("=" ^ 60)

k_dismiss = build_dismiss_kernel(state.compiled_kernels, fa, ts)

# With reliable Beta, program that fires on dismissed article should be penalized more
if discriminating_idx !== nothing
    # discriminating_idx fires on A — if A is dismissed, this is BAD
    reliable_fires = TaggedBetaMeasure(Interval(0.0, 1.0), discriminating_idx, BetaMeasure(5.0, 1.0))
    ll_dismiss_fires = k_dismiss.log_density(reliable_fires, 0.0)

    # Find a program that doesn't fire on A
    global neutral_idx = nothing
    for (j, ck) in enumerate(state.compiled_kernels)
        if ck.evaluate(fa, ts) != :match
            global neutral_idx = j
            break
        end
    end

    if neutral_idx !== nothing
        reliable_neutral = TaggedBetaMeasure(Interval(0.0, 1.0), neutral_idx, BetaMeasure(5.0, 1.0))
        ll_dismiss_neutral = k_dismiss.log_density(reliable_neutral, 0.0)
        @check "firing program penalized more than neutral" ll_dismiss_fires < ll_dismiss_neutral
        println("  Firing program dismiss ll: $(round(ll_dismiss_fires, digits=4))")
        println("  Neutral program dismiss ll: $(round(ll_dismiss_neutral, digits=4))")
    end
end
println()

println("=" ^ 60)
println("TEST 5: Integration — learning from synthetic data")
println("=" ^ 60)

# Generate corpus and simulate a user who prefers programming articles from HN
Random.seed!(42)
corpus = generate_synthetic_corpus(50; rng_seed=42)

state2, reg2 = init_rss_agent(feeds=feeds, categories=cats, known_tags=tags,
                                program_max_depth=3, verbose=false)

# User preference: reads articles tagged "programming" from feed 1 (HN)
function user_wants_to_read(a::Article)
    "programming" in a.tags && a.feed_id == 1
end

# Simulate 5 sessions of reading
for session in 1:5
    session_articles = corpus[((session-1)*10+1):min(session*10, length(corpus))]
    article_feats = Dict{Int, Dict{Symbol, Float64}}()
    for a in session_articles
        article_feats[a.id] = extract_features(a, reg2; now=now_time)
    end

    # User reads the ones they like first, in order
    liked = filter(user_wants_to_read, session_articles)
    reads = [ReadEvent(a.id, now_time + Dates.Minute(i)) for (i, a) in enumerate(liked)]

    # Dismiss obviously uninteresting (low priority, no programming tag)
    disliked = filter(a -> a.feed_priority == 3 && !("programming" in a.tags), session_articles)
    dismissals = [DismissEvent(a.id) for a in disliked]

    process_session!(state2, reads, dismissals, article_feats; verbose=false)
end

# Now rank a new batch of articles
test_articles = generate_synthetic_corpus(20; rng_seed=99)
test_feats = Dict{Int, Dict{Symbol, Float64}}()
for a in test_articles
    test_feats[a.id] = extract_features(a, reg2; now=now_time)
end

ranking = rank_articles(state2, test_feats)

# Check: preferred articles (programming + HN) should rank higher on average
preferred_ids = Set(a.id for a in test_articles if user_wants_to_read(a))
non_preferred_ids = Set(a.id for a in test_articles if !user_wants_to_read(a))

if !isempty(preferred_ids) && !isempty(non_preferred_ids)
    preferred_scores = [s for (id, s) in ranking if id in preferred_ids]
    non_preferred_scores = [s for (id, s) in ranking if id in non_preferred_ids]

    mean_preferred = isempty(preferred_scores) ? 0.0 : sum(preferred_scores) / length(preferred_scores)
    mean_non_preferred = isempty(non_preferred_scores) ? 0.0 : sum(non_preferred_scores) / length(non_preferred_scores)

    @check "preferred articles score higher than non-preferred" mean_preferred > mean_non_preferred
    println("  Mean preferred score: $(round(mean_preferred, digits=4))")
    println("  Mean non-preferred score: $(round(mean_non_preferred, digits=4))")
    println("  $(length(preferred_ids)) preferred, $(length(non_preferred_ids)) non-preferred in test set")
else
    println("  WARNING: no preferred/non-preferred split in test set")
end
println()

println("=" ^ 60)
println("TEST 6: Session processing shrinks unread set")
println("=" ^ 60)

state3, reg3 = init_rss_agent(feeds=feeds, categories=cats, known_tags=tags,
                                program_max_depth=2, verbose=false)
initial_components = length(state3.belief.components)

small_articles = generate_synthetic_corpus(10; rng_seed=77)
small_feats = Dict{Int, Dict{Symbol, Float64}}()
for a in small_articles
    small_feats[a.id] = extract_features(a, reg3; now=now_time)
end

reads = [ReadEvent(small_articles[i].id, now_time + Dates.Minute(i)) for i in 1:3]
process_session!(state3, reads, DismissEvent[], small_feats; verbose=false)

@check "belief still has components after conditioning" length(state3.belief.components) > 0
@check "components may have been pruned" length(state3.belief.components) <= initial_components
println("  Components: $initial_components → $(length(state3.belief.components))")
println()

println("=" ^ 60)
println("TEST 7.5: Non-firing programs retain α=β=1 exactly after conditioning (FiringByTag)")
println("=" ^ 60)

# Fresh state: condition once on a read of article A. Any program that does not
# fire on A's features must have α=β=1 unchanged — FiringByTag routes its branch
# to Flat (no Beta update). Integer accumulation; exact equality is the bar.
state_ft, _ = init_rss_agent(feeds=feeds, categories=cats, known_tags=tags,
                              program_max_depth=2, verbose=false)

# Find a program that does NOT fire on fa; record its pre-condition α/β.
non_firing_idx = nothing
for (j, ck) in enumerate(state_ft.compiled_kernels)
    if ck.evaluate(fa, ts) != :match
        global non_firing_idx = j
        break
    end
end

if non_firing_idx !== nothing
    pre_component = state_ft.belief.components[non_firing_idx]
    pre_alpha = pre_component.beta.alpha
    pre_beta = pre_component.beta.beta

    k_read_ft = build_read_kernel(state_ft.compiled_kernels, fa, [fa, fb, fc], ts)
    state_ft.belief = condition(state_ft.belief, k_read_ft, 1.0)

    post_component = state_ft.belief.components[non_firing_idx]
    @check "non-firing program α unchanged (==)" post_component.beta.alpha == pre_alpha
    @check "non-firing program β unchanged (==)" post_component.beta.beta == pre_beta
    @check "non-firing program tag preserved" post_component.tag == non_firing_idx
    println("  Non-firing program α: $pre_alpha → $(post_component.beta.alpha) (exact)")
    println("  Non-firing program β: $pre_beta → $(post_component.beta.beta) (exact)")
else
    println("  WARNING: no non-firing program found — test inconclusive")
end
println()

println("=" ^ 60)
println("TEST 7: Posterior concentration")
println("=" ^ 60)

w = weights(state2.belief)
top_weight = maximum(w)
@check "top program has above-average weight" top_weight > 1.0 / length(w)
@check "weights sum to ~1" abs(sum(w) - 1.0) < 1e-10

# Check that top grammars are identifiable
gw = aggregate_grammar_weights(weights(state2.belief), state2.metadata)
top_grammar_id = first(sort(collect(gw), by=x->-x[2]))[1]
println("  Top grammar: $top_grammar_id with weight $(round(gw[top_grammar_id], digits=4))")
println("  Top program weight: $(round(top_weight, digits=6))")
println("  Total components: $(length(state2.belief.components))")
println()

# ═══════════════════════════════════════
# Summary
# ═══════════════════════════════════════
println("=" ^ 60)
if failed == 0
    println("ALL TESTS PASSED ($passed checks)")
else
    println("$passed passed, $failed FAILED")
end
println("=" ^ 60)

failed > 0 && exit(1)

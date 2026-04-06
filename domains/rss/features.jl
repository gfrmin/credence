"""
    features.jl — RSS article features for preference learning.

    All features are raw observables in [0, 1]. No pre-baked judgments.
    Dynamic features (per-feed, per-category, per-tag) are built at
    init time from the user's actual feed list.
"""

using Dates

# ── Article struct ──

struct Article
    id::Int                     # Miniflux entry_id
    feed_id::Int
    feed_name::String
    category_name::String
    title::String
    url::String
    published_at::DateTime
    word_count::Int
    has_code::Bool
    has_images::Bool
    has_audio::Bool
    heading_count::Int
    tags::Vector{String}        # LLM-generated topic tags
    feed_priority::Int          # 1=must read, 2=normal, 3=low
end

# ── Feature registry (built once at init) ──

struct FeatureRegistry
    feature_set::Set{Symbol}
    feed_map::Dict{Int, Symbol}        # feed_id → :feed_is_xxx
    cat_map::Dict{String, Symbol}      # category_name → :cat_is_xxx
    tag_map::Dict{String, Symbol}      # tag_name → :tag_xxx
end

const FIXED_FEATURES = Set{Symbol}([
    :feed_priority, :log_word_count, :has_code, :has_images, :has_audio,
    :heading_count_norm, :hour_sin, :hour_cos, :dow_sin, :dow_cos,
    :is_weekend, :log_age_hours,
])

function sanitize_name(s::String)::Symbol
    cleaned = lowercase(strip(s))
    cleaned = replace(cleaned, r"[^a-z0-9]+" => "_")
    cleaned = strip(cleaned, '_')
    isempty(cleaned) && (cleaned = "unknown")
    Symbol(cleaned)
end

function build_feature_registry(
    feeds::Vector{Tuple{Int, String}},          # [(feed_id, feed_name), ...]
    categories::Vector{String},
    known_tags::Vector{String},
)::FeatureRegistry
    feed_map = Dict{Int, Symbol}()
    for (fid, fname) in feeds
        feed_map[fid] = Symbol("feed_is_", sanitize_name(fname))
    end

    cat_map = Dict{String, Symbol}()
    for cname in categories
        cat_map[cname] = Symbol("cat_is_", sanitize_name(cname))
    end

    tag_map = Dict{String, Symbol}()
    for tname in known_tags
        tag_map[tname] = Symbol("tag_", sanitize_name(tname))
    end

    feature_set = union(
        FIXED_FEATURES,
        Set(values(feed_map)),
        Set(values(cat_map)),
        Set(values(tag_map)),
    )

    FeatureRegistry(feature_set, feed_map, cat_map, tag_map)
end

# ── Feature extraction ──

function extract_features(
    article::Article,
    reg::FeatureRegistry;
    now::DateTime = Dates.now(Dates.UTC),
)::Dict{Symbol, Float64}
    f = Dict{Symbol, Float64}()

    # Feed priority: 1→1.0, 2→0.5, 3→0.0
    f[:feed_priority] = article.feed_priority == 1 ? 1.0 :
                         article.feed_priority == 2 ? 0.5 : 0.0

    # Content structure
    f[:log_word_count] = clamp(log(max(article.word_count, 1)) / log(10000), 0.0, 1.0)
    f[:has_code] = article.has_code ? 1.0 : 0.0
    f[:has_images] = article.has_images ? 1.0 : 0.0
    f[:has_audio] = article.has_audio ? 1.0 : 0.0
    f[:heading_count_norm] = clamp(article.heading_count / 20.0, 0.0, 1.0)

    # Temporal (cyclical encoding)
    hour = Dates.hour(article.published_at)
    f[:hour_sin] = (sin(2π * hour / 24) + 1) / 2
    f[:hour_cos] = (cos(2π * hour / 24) + 1) / 2
    dow = Dates.dayofweek(article.published_at)  # 1=Monday, 7=Sunday
    f[:dow_sin] = (sin(2π * dow / 7) + 1) / 2
    f[:dow_cos] = (cos(2π * dow / 7) + 1) / 2
    f[:is_weekend] = dow >= 6 ? 1.0 : 0.0

    # Age
    age_hours = max(Dates.value(now - article.published_at) / (1000 * 3600), 0.1)
    f[:log_age_hours] = clamp(log(age_hours) / log(720), 0.0, 1.0)  # 720h = 30 days

    # Feed one-hot
    for (fid, sym) in reg.feed_map
        f[sym] = fid == article.feed_id ? 1.0 : 0.0
    end

    # Category one-hot
    for (cname, sym) in reg.cat_map
        f[sym] = cname == article.category_name ? 1.0 : 0.0
    end

    # Tag binary
    for (tname, sym) in reg.tag_map
        f[sym] = tname in article.tags ? 1.0 : 0.0
    end

    f
end

# ── Synthetic data for testing ──

function generate_synthetic_corpus(
    n::Int;
    rng_seed::Int = 42,
    feeds = [(1, "Hacker News"), (2, "Ars Technica"), (3, "Low Quality Blog")],
    categories = ["tech", "news"],
    tags = ["programming", "ai", "linux", "security", "gaming"],
)::Vector{Article}
    rng = Random.MersenneTwister(rng_seed)
    articles = Article[]
    base_time = DateTime(2026, 4, 1, 9, 0, 0)

    for i in 1:n
        fid, fname = feeds[rand(rng, 1:length(feeds))]
        cat = categories[rand(rng, 1:length(categories))]
        n_tags = rand(rng, 1:3)
        atags = String[tags[j] for j in Random.randperm(rng, length(tags))[1:n_tags]]
        wc = round(Int, exp(randn(rng) * 0.8 + 5.5))  # median ~250
        pub = base_time - Dates.Hour(rand(rng, 0:168))  # within last week
        priority = fid == 1 ? 1 : fid == 2 ? 2 : 3

        push!(articles, Article(
            i, fid, fname, cat, "Article $i", "https://example.com/$i",
            pub, wc,
            rand(rng) < 0.3,   # has_code
            rand(rng) < 0.5,   # has_images
            rand(rng) < 0.05,  # has_audio
            rand(rng, 0:8),    # heading_count
            atags, priority,
        ))
    end
    articles
end

using Random

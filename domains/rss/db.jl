"""
    db.jl — PostgreSQL queries for the RSS domain.

    Reads from the shared Miniflux + sidecar database.
    All queries are read-only. Uses LibPQ.jl.
"""

using LibPQ
using Dates

# ── Connection ──

function connect_db(database_url::String = ENV["DATABASE_URL"])
    LibPQ.Connection(database_url)
end

# ── Data loading (called at init) ──

function fetch_feeds(conn)::Vector{Tuple{Int, String}}
    result = execute(conn, """
        SELECT f.id, f.title
        FROM feeds f
        ORDER BY f.id
    """)
    [(parse(Int, row[1]), row[2]) for row in result]
end

function fetch_categories(conn)::Vector{String}
    result = execute(conn, "SELECT title FROM categories ORDER BY title")
    [row[1] for row in result]
end

function fetch_known_tags(conn)::Vector{String}
    result = execute(conn, "SELECT DISTINCT tag FROM article_tags ORDER BY tag")
    [row[1] for row in result]
end

function fetch_feed_priorities(conn)::Dict{Int, Int}
    result = execute(conn, "SELECT feed_id, COALESCE(priority, 2) FROM feed_config")
    Dict(parse(Int, row[1]) => parse(Int, row[2]) for row in result)
end

# ── Article loading ──

"""
    fetch_unread_articles(conn, priorities) → Vector{Article}

Load all unread entries with their tags and content metadata.
Joins across Miniflux entries, sidecar article_tags, and article_snapshots.
"""
function fetch_unread_articles(conn, priorities::Dict{Int, Int})::Vector{Article}
    result = execute(conn, """
        SELECT
            e.id,
            e.feed_id,
            f.title AS feed_title,
            COALESCE(c.title, '') AS category_title,
            e.title,
            e.url,
            e.published_at,
            COALESCE(s.content_text, e.content, '') AS content_text,
            COALESCE(s.content_html, '', '') AS content_html
        FROM entries e
        JOIN feeds f ON f.id = e.feed_id
        LEFT JOIN categories c ON c.id = f.category_id
        LEFT JOIN LATERAL (
            SELECT content_text, content_html
            FROM article_snapshots
            WHERE entry_id = e.id
            ORDER BY version DESC
            LIMIT 1
        ) s ON true
        WHERE e.status = 'unread'
        ORDER BY e.published_at DESC
    """)

    # Batch-load tags for all unread entries
    entry_ids = [parse(Int, row[1]) for row in result]
    tags_by_entry = _fetch_tags_batch(conn, entry_ids)

    articles = Article[]
    for row in result
        eid = parse(Int, row[1])
        fid = parse(Int, row[2])
        content_text = something(row[8], "")
        content_html = something(row[9], "")

        push!(articles, Article(
            eid, fid,
            something(row[3], ""),        # feed_title
            something(row[4], ""),        # category_title
            something(row[5], ""),        # title
            something(row[6], ""),        # url
            _parse_pg_timestamp(row[7]),  # published_at
            length(split(content_text)),  # word_count
            _has_code(content_text, content_html),
            _has_images(content_html),
            _has_audio_enclosure(conn, eid),
            _count_headings(content_html),
            get(tags_by_entry, eid, String[]),
            get(priorities, fid, 2),
        ))
    end
    articles
end

function _fetch_tags_batch(conn, entry_ids::Vector{Int})::Dict{Int, Vector{String}}
    isempty(entry_ids) && return Dict{Int, Vector{String}}()
    ids_str = join(entry_ids, ",")
    result = execute(conn, "SELECT entry_id, tag FROM article_tags WHERE entry_id IN ($ids_str)")
    tags = Dict{Int, Vector{String}}()
    for row in result
        eid = parse(Int, row[1])
        push!(get!(tags, eid, String[]), row[2])
    end
    tags
end

function _has_audio_enclosure(conn, entry_id::Int)::Bool
    result = execute(conn, """
        SELECT 1 FROM enclosures
        WHERE entry_id = $entry_id AND mime_type LIKE 'audio/%'
        LIMIT 1
    """)
    !isempty(result)
end

# ── Content feature helpers ──

_has_code(text, html) = occursin(r"```|<code|<pre", text * html)
_has_images(html) = occursin("<img", html)
_count_headings(html) = count(r"<h[1-6]"i, html)

function _parse_pg_timestamp(s)::DateTime
    s === nothing && return DateTime(2020, 1, 1)
    # PostgreSQL timestamps: "2026-04-01 14:30:00+00" or ISO 8601
    cleaned = replace(string(s), r"\+\d{2}$" => "")
    cleaned = replace(cleaned, "T" => " ")
    cleaned = replace(cleaned, r"\.\d+$" => "")
    try
        DateTime(cleaned, dateformat"yyyy-mm-dd HH:MM:SS")
    catch
        DateTime(2020, 1, 1)
    end
end

# ── Event loading (for conditioning) ──

"""
    fetch_read_events_since(conn, since) → Vector{ReadEvent}

Fetch read events (user opened article detail) since a given time.
"""
function fetch_read_events_since(conn, since::DateTime)::Vector{ReadEvent}
    result = execute(conn, """
        SELECT entry_id, read_at
        FROM read_events
        WHERE read_at > '$(Dates.format(since, dateformat"yyyy-mm-dd HH:MM:SS"))'
        ORDER BY read_at ASC
    """)
    [ReadEvent(parse(Int, row[1]), _parse_pg_timestamp(row[2])) for row in result]
end

"""
    fetch_dismiss_events_since(conn, since) → Vector{DismissEvent}

Dismiss = entry marked as read (status='read') but never opened (no read_event).
Uses Miniflux's changed_at to detect when status changed.
"""
function fetch_dismiss_events_since(conn, since::DateTime)::Vector{DismissEvent}
    since_str = Dates.format(since, dateformat"yyyy-mm-dd HH:MM:SS")
    result = execute(conn, """
        SELECT e.id
        FROM entries e
        WHERE e.status = 'read'
          AND e.changed_at > '$since_str'
          AND e.id NOT IN (
              SELECT entry_id FROM read_events WHERE read_at > '$since_str'
          )
    """)
    [DismissEvent(parse(Int, row[1])) for row in result]
end

"""
    fetch_all_read_events(conn) → Vector{ReadEvent}

All historical read events, for bootstrapping.
"""
function fetch_all_read_events(conn)::Vector{ReadEvent}
    result = execute(conn, "SELECT entry_id, read_at FROM read_events ORDER BY read_at ASC")
    [ReadEvent(parse(Int, row[1]), _parse_pg_timestamp(row[2])) for row in result]
end

"""
    fetch_article_features_for_entries(conn, entry_ids, reg) → Dict{Int, Dict{Symbol, Float64}}

Load features for specific entry IDs (for conditioning on historical events).
"""
function fetch_article_features_for_entries(
    conn, entry_ids::Vector{Int}, reg::FeatureRegistry, priorities::Dict{Int, Int};
    now::DateTime = Dates.now(Dates.UTC),
)::Dict{Int, Dict{Symbol, Float64}}
    isempty(entry_ids) && return Dict{Int, Dict{Symbol, Float64}}()
    ids_str = join(entry_ids, ",")

    result = execute(conn, """
        SELECT
            e.id, e.feed_id, f.title AS feed_title,
            COALESCE(c.title, '') AS cat_title,
            e.title, e.url, e.published_at,
            COALESCE(s.content_text, e.content, '') AS content_text,
            COALESCE(s.content_html, '') AS content_html
        FROM entries e
        JOIN feeds f ON f.id = e.feed_id
        LEFT JOIN categories c ON c.id = f.category_id
        LEFT JOIN LATERAL (
            SELECT content_text, content_html
            FROM article_snapshots WHERE entry_id = e.id
            ORDER BY version DESC LIMIT 1
        ) s ON true
        WHERE e.id IN ($ids_str)
    """)

    tags_by_entry = _fetch_tags_batch(conn, entry_ids)
    features = Dict{Int, Dict{Symbol, Float64}}()

    for row in result
        eid = parse(Int, row[1])
        fid = parse(Int, row[2])
        content_text = something(row[8], "")
        content_html = something(row[9], "")

        article = Article(
            eid, fid, something(row[3], ""), something(row[4], ""),
            something(row[5], ""), something(row[6], ""),
            _parse_pg_timestamp(row[7]),
            length(split(content_text)),
            _has_code(content_text, content_html),
            _has_images(content_html),
            false,  # skip audio check for batch loading
            _count_headings(content_html),
            get(tags_by_entry, eid, String[]),
            get(priorities, fid, 2),
        )
        features[eid] = extract_features(article, reg; now=now)
    end
    features
end

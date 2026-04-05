"""
    features.jl — Email type, feature extraction, user preferences, synthetic corpus

Defines the email observation space as Dict{Symbol, Float64}. Content features
(13 keys) plus optional processing-state features (9 keys) for multi-step episodes.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Credence: Grammar

using Random

# ═══════════════════════════════════════
# Email type
# ═══════════════════════════════════════

struct Email
    id::Int
    sender::String
    sender_frequency::Float64      # normalised [0,1]
    sender_category::Symbol        # :manager, :direct_report, :external, :frequent, :rare
    subject::String
    urgency::Float64               # [0, 1]
    topic::Symbol                  # :finance, :scheduling, :marketing, :personal, :technical
    requires_action::Bool
    word_count::Int
    has_attachment::Bool
    hour_received::Int             # 0-23
    thread_depth::Int              # 0 = new, 1+ = reply
end

# ═══════════════════════════════════════
# Feature names
# ═══════════════════════════════════════

const EMAIL_FEATURE_NAMES = Set([
    :sender_frequency,
    :sender_is_manager,
    :sender_is_direct_report,
    :sender_is_external,
    :urgency,
    :topic_finance,
    :topic_scheduling,
    :topic_marketing,
    :requires_action,
    :email_length,
    :has_attachment,
    :time_of_day,
    :thread_depth,
])

const EMAIL_STATE_FEATURE_NAMES = Set([
    :has_label_urgent,
    :has_label_delegated,
    :is_in_archive,
    :is_in_priority,
    :is_in_later,
    :is_read,
    :user_notified,
    :reply_drafted,
    :is_assigned,
])

const ALL_EMAIL_FEATURES = EMAIL_FEATURE_NAMES
const ALL_EMAIL_FEATURES_EXTENDED = union(EMAIL_FEATURE_NAMES, EMAIL_STATE_FEATURE_NAMES)

# ═══════════════════════════════════════
# Processing state — tracks which primitives have been applied
# ═══════════════════════════════════════

mutable struct ProcessingState
    has_label_urgent::Bool
    has_label_delegated::Bool
    is_in_archive::Bool
    is_in_priority::Bool
    is_in_later::Bool
    is_read::Bool
    user_notified::Bool
    reply_drafted::Bool
    is_assigned::Bool
end

ProcessingState() = ProcessingState(false, false, false, false, false, false, false, false, false)

# ═══════════════════════════════════════
# Feature extraction — returns Dict{Symbol, Float64}
# ═══════════════════════════════════════

"""
    extract_features(email::Email) → Dict{Symbol, Float64}

Extract content features from an email. All values normalised to [0, 1].
"""
function extract_features(email::Email)::Dict{Symbol, Float64}
    Dict{Symbol, Float64}(
        :sender_frequency => email.sender_frequency,
        :sender_is_manager => email.sender_category == :manager ? 1.0 : 0.0,
        :sender_is_direct_report => email.sender_category == :direct_report ? 1.0 : 0.0,
        :sender_is_external => email.sender_category == :external ? 1.0 : 0.0,
        :urgency => email.urgency,
        :topic_finance => email.topic == :finance ? 1.0 : 0.0,
        :topic_scheduling => email.topic == :scheduling ? 1.0 : 0.0,
        :topic_marketing => email.topic == :marketing ? 1.0 : 0.0,
        :requires_action => email.requires_action ? 1.0 : 0.0,
        :email_length => clamp(email.word_count / 1000.0, 0.0, 1.0),
        :has_attachment => email.has_attachment ? 1.0 : 0.0,
        :time_of_day => email.hour_received / 24.0,
        :thread_depth => clamp(email.thread_depth / 10.0, 0.0, 1.0),
    )
end

"""
    extract_features(email::Email, ps::ProcessingState) → Dict{Symbol, Float64}

Extract content + processing-state features.
"""
function extract_features(email::Email, ps::ProcessingState)::Dict{Symbol, Float64}
    d = extract_features(email)
    d[:has_label_urgent] = ps.has_label_urgent ? 1.0 : 0.0
    d[:has_label_delegated] = ps.has_label_delegated ? 1.0 : 0.0
    d[:is_in_archive] = ps.is_in_archive ? 1.0 : 0.0
    d[:is_in_priority] = ps.is_in_priority ? 1.0 : 0.0
    d[:is_in_later] = ps.is_in_later ? 1.0 : 0.0
    d[:is_read] = ps.is_read ? 1.0 : 0.0
    d[:user_notified] = ps.user_notified ? 1.0 : 0.0
    d[:reply_drafted] = ps.reply_drafted ? 1.0 : 0.0
    d[:is_assigned] = ps.is_assigned ? 1.0 : 0.0
    d
end

# ═══════════════════════════════════════
# User preference profiles
# ═══════════════════════════════════════

struct UserPreference
    name::Symbol
    decide::Function    # Email → Symbol (one of DOMAIN_ACTIONS)
end

function urgency_responsive_decide(email::Email)::Symbol
    email.urgency > 0.7 && return :flag_urgent
    email.topic == :marketing && return :archive
    email.sender_category == :manager && return :draft_response
    :schedule_later
end

function delegator_decide(email::Email)::Symbol
    email.sender_category == :manager && return :draft_response
    email.topic == :marketing && return :archive
    :delegate
end

function hands_on_decide(email::Email)::Symbol
    email.topic == :marketing && return :archive
    :draft_response
end

function selective_decide(email::Email)::Symbol
    (email.sender_category == :manager || email.sender_category == :direct_report) && return :draft_response
    :archive
end

"""Triage-focused profile: uses composite actions for nuanced handling."""
function triage_decide(email::Email)::Symbol
    # Urgent external emails get escalated (flag + notify + delegate)
    email.urgency > 0.7 && email.sender_category == :external && return :triage_urgent
    # Urgent from managers get escalated
    email.urgency > 0.7 && email.sender_category == :manager && return :escalate
    # Marketing gets silently archived (no notification)
    email.topic == :marketing && return :silent_archive
    # Everything else gets normal handling
    email.sender_category == :manager && return :draft_response
    :schedule_later
end

const PREFERENCE_PROFILES = Dict{Symbol, UserPreference}(
    :urgency_responsive => UserPreference(:urgency_responsive, urgency_responsive_decide),
    :delegator          => UserPreference(:delegator, delegator_decide),
    :hands_on           => UserPreference(:hands_on, hands_on_decide),
    :selective          => UserPreference(:selective, selective_decide),
    :triage             => UserPreference(:triage, triage_decide),
)

"""
    simulate_user_reaction(pref, email, recommended) → Bool

Returns true if the recommended action matches the user's preference.
"""
function simulate_user_reaction(pref::UserPreference, email::Email, recommended::Symbol)::Bool
    pref.decide(email) == recommended
end

# ═══════════════════════════════════════
# Synthetic corpus generation
# ═══════════════════════════════════════

const SENDER_NAMES = [
    "alice@company.com", "bob@company.com", "carol@external.org",
    "dave@company.com", "eve@vendor.io", "frank@company.com",
    "grace@company.com", "heidi@external.net", "ivan@company.com",
    "judy@marketing.biz", "karl@company.com", "lisa@company.com",
]

const SUBJECT_TEMPLATES = [
    "Q%d budget review", "Meeting re: project %d", "Newsletter #%d",
    "Urgent: server %d down", "RE: invoice #%d", "FYI: policy update %d",
    "Action needed: report %d", "Quick question about task %d",
    "Delegation: review PR #%d", "Team standup notes %d",
]

"""
    generate_email_corpus(n; rng_seed) → Vector{Email}

Generate a synthetic email corpus with controlled distributions.
"""
function generate_email_corpus(n::Int; rng_seed::Int=42)::Vector{Email}
    rng = MersenneTwister(rng_seed)
    emails = Email[]

    category_weights = [0.20, 0.30, 0.20, 0.15, 0.15]
    categories = [:manager, :direct_report, :external, :frequent, :rare]

    topic_weights = [0.20, 0.30, 0.15, 0.10, 0.25]
    topics = [:finance, :scheduling, :marketing, :personal, :technical]

    for i in 1:n
        # Sender category
        cat = _weighted_choice(rng, categories, category_weights)

        # Sender frequency: managers and direct_reports appear more often
        freq = if cat in (:manager, :direct_report, :frequent)
            0.4 + 0.6 * rand(rng)
        elseif cat == :external
            0.05 + 0.3 * rand(rng)
        else
            0.1 + 0.4 * rand(rng)
        end

        # Sender name
        sender = SENDER_NAMES[mod1(i, length(SENDER_NAMES))]

        # Topic
        topic = _weighted_choice(rng, topics, topic_weights)

        # Urgency
        urgency = rand(rng)

        # Subject
        subject = SUBJECT_TEMPLATES[mod1(i, length(SUBJECT_TEMPLATES))]
        subject = replace(subject, r"%d" => string(i))

        # Requires action: ~40%
        requires_action = rand(rng) < 0.4

        # Word count: lognormal, median ~200
        word_count = max(10, round(Int, exp(log(200) + 0.8 * randn(rng))))

        # Attachment: ~20%
        has_attachment = rand(rng) < 0.2

        # Hour: bimodal around 10 and 14
        hour = if rand(rng) < 0.5
            clamp(round(Int, 10 + 2 * randn(rng)), 0, 23)
        else
            clamp(round(Int, 14 + 2 * randn(rng)), 0, 23)
        end

        # Thread depth: geometric, mostly 0-2
        thread_depth = 0
        while rand(rng) < 0.3 && thread_depth < 10
            thread_depth += 1
        end

        push!(emails, Email(i, sender, freq, cat, subject, urgency, topic,
                            requires_action, word_count, has_attachment,
                            hour, thread_depth))
    end
    emails
end

function _weighted_choice(rng, items, weights)
    r = rand(rng)
    cumulative = 0.0
    for (item, w) in zip(items, weights)
        cumulative += w
        r <= cumulative && return item
    end
    last(items)
end

# ═══════════════════════════════════════
# JMAP → Email bridge (live mode)
# ═══════════════════════════════════════

"""
    jmap_to_email(raw, idx; sender_history, thread_counts) → (Email, jmap_id)

Convert a JMAP email dict to an Email struct. Cheap fields are parsed
directly; expensive fields (urgency, topic, requires_action) get safe
defaults that the agent can enrich via VOI-gated LLM when worthwhile.
"""
function jmap_to_email(
    raw, idx::Int;
    sender_history::Dict{String, Int}=Dict{String, Int}(),
    thread_counts::Dict{String, Int}=Dict{String, Int}()
)
    # Sender
    from_list = get(raw, "from", nothing)
    sender = if from_list !== nothing && length(from_list) > 0
        String(get(from_list[1], "email", "unknown@unknown"))
    else
        "unknown@unknown"
    end

    # Sender frequency from session history
    sender_freq = min(1.0, get(sender_history, sender, 0) / 20.0)

    # Subject
    subject = String(get(raw, "subject", "(no subject)"))

    # Word count estimate from byte size
    size_bytes = Int(get(raw, "size", 600))
    word_count = max(10, size_bytes ÷ 6)

    # Attachment
    has_attachment = Bool(get(raw, "hasAttachment", false))

    # Hour from receivedAt (ISO 8601: "2026-04-05T10:30:00Z")
    received_at = String(get(raw, "receivedAt", "2000-01-01T12:00:00Z"))
    hour = _parse_hour(received_at)

    # Thread depth from session counts
    thread_id = String(get(raw, "threadId", ""))
    thread_depth = get(thread_counts, thread_id, 0)

    # JMAP ID for action execution
    jmap_id = String(get(raw, "id", ""))

    # Preview text (stored on Email.subject for LLM prompt access)
    preview = String(get(raw, "preview", ""))

    email = Email(idx, sender, sender_freq, :unknown, subject,
                  0.5,        # urgency: default, LLM-enrichable
                  :unknown,   # topic: default, LLM-enrichable
                  false,      # requires_action: default, LLM-enrichable
                  word_count, has_attachment, hour, thread_depth)

    (email, jmap_id, preview)
end

"""Parse hour (0-23) from ISO 8601 datetime string."""
function _parse_hour(iso::String)::Int
    # Find T, take next 2 chars as hour
    t_pos = findfirst('T', iso)
    t_pos === nothing && return 12
    length(iso) < t_pos + 2 && return 12
    h = tryparse(Int, iso[t_pos+1:t_pos+2])
    h === nothing ? 12 : clamp(h, 0, 23)
end

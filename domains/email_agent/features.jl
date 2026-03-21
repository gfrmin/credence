"""
    features.jl — Email type, feature extraction, user preferences, synthetic corpus

Defines the email observation space (13 feature channels), user preference
profiles for simulation, and synthetic corpus generation.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Credence: SensorChannel, SensorConfig, n_channels

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
# Feature extraction — 13 channels
# ═══════════════════════════════════════

const N_EMAIL_FEATURES = 13

const EMAIL_FEATURE_NAMES = [
    :sender_frequency,        # 0: direct
    :sender_is_manager,       # 1: 1.0 if :manager
    :sender_is_direct_report, # 2: 1.0 if :direct_report
    :sender_is_external,      # 3: 1.0 if :external
    :urgency,                 # 4: direct
    :topic_finance,           # 5: 1.0 if :finance
    :topic_scheduling,        # 6: 1.0 if :scheduling
    :topic_marketing,         # 7: 1.0 if :marketing
    :requires_action,         # 8: 1.0 if true
    :email_length,            # 9: clamp(word_count/1000, 0, 1)
    :has_attachment,          # 10: 1.0 if true
    :time_of_day,             # 11: hour/24.0
    :thread_depth,            # 12: clamp(depth/10, 0, 1)
]

"""
    extract_features(email::Email) → Vector{Float64}

Extract a 13-element feature vector from an email. All values normalised to [0, 1].
"""
function extract_features(email::Email)::Vector{Float64}
    Float64[
        email.sender_frequency,
        email.sender_category == :manager ? 1.0 : 0.0,
        email.sender_category == :direct_report ? 1.0 : 0.0,
        email.sender_category == :external ? 1.0 : 0.0,
        email.urgency,
        email.topic == :finance ? 1.0 : 0.0,
        email.topic == :scheduling ? 1.0 : 0.0,
        email.topic == :marketing ? 1.0 : 0.0,
        email.requires_action ? 1.0 : 0.0,
        clamp(email.word_count / 1000.0, 0.0, 1.0),
        email.has_attachment ? 1.0 : 0.0,
        email.hour_received / 24.0,
        clamp(email.thread_depth / 10.0, 0.0, 1.0),
    ]
end

# ═══════════════════════════════════════
# Sensor configs for email
# ═══════════════════════════════════════

"""Full 13-channel sensor config."""
function full_email_sensor_config()
    SensorConfig([SensorChannel(i, :identity, 0.05, 1.0) for i in 0:N_EMAIL_FEATURES-1])
end

"""Sender-focused: channels 0-3."""
function sender_sensor_config()
    SensorConfig([SensorChannel(i, :identity, 0.05, 1.0) for i in 0:3])
end

"""Topic-focused: channels 5-7."""
function topic_sensor_config()
    SensorConfig([SensorChannel(i, :identity, 0.05, 1.0) for i in 5:7])
end

"""Urgency + action: channels 4, 8."""
function action_sensor_config()
    SensorConfig([
        SensorChannel(4, :identity, 0.05, 1.0),
        SensorChannel(8, :identity, 0.05, 1.0),
    ])
end

"""Minimal: channels 0, 1, 4 (sender_freq, is_manager, urgency)."""
function minimal_email_sensor_config()
    SensorConfig([
        SensorChannel(0, :identity, 0.05, 1.0),
        SensorChannel(1, :identity, 0.05, 1.0),
        SensorChannel(4, :identity, 0.05, 1.0),
    ])
end

# ═══════════════════════════════════════
# Sensor projection
# ═══════════════════════════════════════

"""
    project_email(email::Email, config::SensorConfig) → Vector{Float64}

Extract features from an email and add sensor noise per channel.
"""
function project_email(email::Email, config::SensorConfig)::Vector{Float64}
    features = extract_features(email)
    readings = Float64[]
    for ch in config.channels
        raw = features[ch.source_index + 1]  # 0-based → 1-based
        noisy = raw + randn() * ch.noise_σ
        push!(readings, clamp(noisy, 0.0, 1.0))
    end
    readings
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

const PREFERENCE_PROFILES = Dict{Symbol, UserPreference}(
    :urgency_responsive => UserPreference(:urgency_responsive, urgency_responsive_decide),
    :delegator          => UserPreference(:delegator, delegator_decide),
    :hands_on           => UserPreference(:hands_on, hands_on_decide),
    :selective          => UserPreference(:selective, selective_decide),
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

Generate a synthetic email corpus with controlled distributions:
- ~20% manager, ~30% direct reports, ~20% external, ~30% other
- urgency uniform [0, 1]
- topics: 30% scheduling, 25% technical, 20% finance, 15% marketing, 10% personal
- ~40% require action
- word count: lognormal, median ~200
- ~20% attachments
- hour: bimodal around 10 and 14
- thread_depth: geometric, mostly 0-2
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

"""
    features.jl — Email type, feature extraction, user preferences, synthetic corpus

Defines the email observation space as Dict{Symbol, Float64}. All features
are raw observables — binary keyword flags, continuous metadata — never
pre-baked semantic judgments. The agent discovers which patterns matter
through conditioning on user feedback.

Content features (15 keys) plus optional processing-state features (9 keys)
for multi-step episodes.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
using Credence
using Credence: Grammar

using Random

# ═══════════════════════════════════════
# Email type — bag of raw observables
# ═══════════════════════════════════════

struct Email
    id::Int
    sender::String
    sender_frequency::Float64       # normalised [0,1] from session history
    sender_is_noreply::Bool         # local part matches noreply/no-reply/donotreply
    sender_is_bulk_domain::Bool     # domain matches known bulk senders
    sender_h0::Bool                 # hash bit 0 of sender domain
    sender_h1::Bool                 # hash bit 1 of sender domain
    sender_h2::Bool                 # hash bit 2 of sender domain
    sender_h3::Bool                 # hash bit 3 of sender domain
    sender_has_news_kw::Bool        # "newsletter", "digest", "daily", "weekly" in sender
    subject::String
    subject_is_reply::Bool          # starts with Re:/RE:
    subject_is_fwd::Bool            # starts with Fwd:/FW:
    subject_has_urgent_kw::Bool     # "urgent", "asap", "critical", "important"
    subject_has_action_kw::Bool     # "please review", "action required", "deadline"
    subject_has_money_kw::Bool      # "invoice", "payment", "budget", "billing"
    subject_has_meeting_kw::Bool    # "meeting", "calendar", "standup", "schedule"
    subject_has_you::Bool           # "your", "you" — personalized
    subject_has_new_event::Bool     # "new submission", "new message", "new reason"
    subject_has_failed::Bool        # "failed", "couldn't", "invalid", "error"
    subject_has_confirmed::Bool     # "confirmed", "confirmation", "booking"
    word_count::Int
    has_attachment::Bool
    is_large_html::Bool             # size > 50KB — newsletter-sized
    hour_received::Int              # 0-23
    thread_depth::Int               # 0 = new, 1+ = reply
    preview_has_unsubscribe::Bool   # "unsubscribe" in preview
    preview_has_question::Bool      # "?" in first 200 chars of preview
    preview_has_click::Bool         # "click here", "view in browser"
end

# ═══════════════════════════════════════
# Feature names
# ═══════════════════════════════════════

const EMAIL_FEATURE_NAMES = Set([
    :sender_frequency,
    :sender_is_noreply,
    :sender_is_bulk_domain,
    :sender_h0,
    :sender_h1,
    :sender_h2,
    :sender_h3,
    :sender_has_news_kw,
    :subject_is_reply,
    :subject_is_fwd,
    :subject_has_urgent_kw,
    :subject_has_action_kw,
    :subject_has_money_kw,
    :subject_has_meeting_kw,
    :subject_has_you,
    :subject_has_new_event,
    :subject_has_failed,
    :subject_has_confirmed,
    :email_length,
    :has_attachment,
    :is_large_html,
    :time_of_day,
    :thread_depth,
    :preview_has_unsubscribe,
    :preview_has_question,
    :preview_has_click,
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

Extract content features from an email. All values in [0, 1].
Every feature is a raw observable — no semantic inference.
"""
function extract_features(email::Email)::Dict{Symbol, Float64}
    Dict{Symbol, Float64}(
        :sender_frequency => email.sender_frequency,
        :sender_is_noreply => email.sender_is_noreply ? 1.0 : 0.0,
        :sender_is_bulk_domain => email.sender_is_bulk_domain ? 1.0 : 0.0,
        :sender_h0 => email.sender_h0 ? 1.0 : 0.0,
        :sender_h1 => email.sender_h1 ? 1.0 : 0.0,
        :sender_h2 => email.sender_h2 ? 1.0 : 0.0,
        :sender_h3 => email.sender_h3 ? 1.0 : 0.0,
        :sender_has_news_kw => email.sender_has_news_kw ? 1.0 : 0.0,
        :subject_is_reply => email.subject_is_reply ? 1.0 : 0.0,
        :subject_is_fwd => email.subject_is_fwd ? 1.0 : 0.0,
        :subject_has_urgent_kw => email.subject_has_urgent_kw ? 1.0 : 0.0,
        :subject_has_action_kw => email.subject_has_action_kw ? 1.0 : 0.0,
        :subject_has_money_kw => email.subject_has_money_kw ? 1.0 : 0.0,
        :subject_has_meeting_kw => email.subject_has_meeting_kw ? 1.0 : 0.0,
        :subject_has_you => email.subject_has_you ? 1.0 : 0.0,
        :subject_has_new_event => email.subject_has_new_event ? 1.0 : 0.0,
        :subject_has_failed => email.subject_has_failed ? 1.0 : 0.0,
        :subject_has_confirmed => email.subject_has_confirmed ? 1.0 : 0.0,
        :email_length => clamp(email.word_count / 5000.0, 0.0, 1.0),
        :has_attachment => email.has_attachment ? 1.0 : 0.0,
        :is_large_html => email.is_large_html ? 1.0 : 0.0,
        :time_of_day => email.hour_received / 24.0,
        :thread_depth => clamp(email.thread_depth / 10.0, 0.0, 1.0),
        :preview_has_unsubscribe => email.preview_has_unsubscribe ? 1.0 : 0.0,
        :preview_has_question => email.preview_has_question ? 1.0 : 0.0,
        :preview_has_click => email.preview_has_click ? 1.0 : 0.0,
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
    email.subject_has_urgent_kw && return :flag_urgent
    email.sender_is_bulk_domain && return :archive
    email.sender_frequency > 0.7 && return :draft_response
    :schedule_later
end

function delegator_decide(email::Email)::Symbol
    email.sender_frequency > 0.7 && return :draft_response
    email.sender_is_bulk_domain && return :archive
    :delegate
end

function hands_on_decide(email::Email)::Symbol
    email.sender_is_bulk_domain && return :archive
    :draft_response
end

function selective_decide(email::Email)::Symbol
    email.sender_frequency > 0.5 && return :draft_response
    :archive
end

"""Triage-focused profile: uses composite actions for nuanced handling."""
function triage_decide(email::Email)::Symbol
    email.subject_has_urgent_kw && email.sender_is_bulk_domain && return :triage_urgent
    email.subject_has_urgent_kw && email.sender_frequency > 0.5 && return :escalate
    email.preview_has_unsubscribe && return :silent_archive
    email.sender_is_bulk_domain && return :silent_archive
    email.sender_frequency > 0.7 && return :draft_response
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
# Raw feature extraction from text
# ═══════════════════════════════════════

const BULK_DOMAINS = Set([
    "substack.com", "mailchimp.com", "sendgrid.net", "sendgrid.com",
    "linkedin.com", "facebookmail.com", "github.com", "googlemail.com",
    "shopify.com", "mailgun.org", "amazonses.com", "mandrillapp.com",
])

const NOREPLY_PATTERNS = ["noreply", "no-reply", "donotreply", "do-not-reply",
                           "notification", "notifications", "mailer", "digest"]

"""Check if sender local part matches noreply patterns."""
function _is_noreply_sender(sender::String)::Bool
    local_part = lowercase(first(split(sender, '@'; limit=2)))
    any(p -> contains(local_part, p), NOREPLY_PATTERNS)
end

"""Check if sender domain is a known bulk/automated sender."""
function _is_bulk_domain(sender::String)::Bool
    parts = split(sender, '@'; limit=2)
    length(parts) < 2 && return false
    domain = lowercase(String(parts[2]))
    domain in BULK_DOMAINS
end

"""Check subject starts with reply prefix."""
_is_reply(subject::String)::Bool = startswith(subject, r"[Rr][Ee]\s*:")

"""Check subject starts with forward prefix."""
_is_fwd(subject::String)::Bool = startswith(subject, r"[Ff][Ww][Dd]?\s*:")

"""Check subject contains urgency keywords."""
_has_urgent_kw(subject::String)::Bool =
    contains(lowercase(subject), r"urgent|asap|critical|important|emergency")

"""Check subject contains action-request keywords."""
_has_action_kw(subject::String)::Bool =
    contains(lowercase(subject), r"action.?required|please.?(review|approve|confirm|sign)|deadline|assigned")

"""Check subject contains money/finance keywords."""
_has_money_kw(subject::String)::Bool =
    contains(lowercase(subject), r"invoice|payment|budget|billing|receipt|expense|tax|financial")

"""Check subject contains meeting/scheduling keywords."""
_has_meeting_kw(subject::String)::Bool =
    contains(lowercase(subject), r"meeting|calendar|standup|schedule|sync|agenda|rsvp|invite")

"""Check preview contains unsubscribe."""
_has_unsubscribe(preview::String)::Bool =
    contains(lowercase(first(preview, 500)), "unsubscribe")

"""Check preview contains question mark."""
_has_question(preview::String)::Bool = contains(first(preview, 200), '?')

"""Check sender address contains newsletter/digest keywords."""
_has_news_sender(sender::String)::Bool =
    contains(lowercase(sender), r"newsletter|digest|daily|weekly|roundup|news@|bulletin")

"""Check subject contains 'your' or 'you' — personalized."""
_has_you(subject::String)::Bool =
    contains(lowercase(subject), r"\byou(r|rs)?\b")

"""Check subject signals a new event — 'new submission', 'new message', etc."""
_has_new_event(subject::String)::Bool =
    contains(lowercase(subject), r"new (submission|message|reason|comment|order|request|alert)")

"""Check subject signals failure — 'failed', 'couldn't', 'invalid'."""
_has_failed(subject::String)::Bool =
    contains(lowercase(subject), r"fail(ed)?|couldn.t|invalid|error|problem|issue")

"""Check subject signals confirmation — 'confirmed', 'booking', 'confirmation'."""
_has_confirmed(subject::String)::Bool =
    contains(lowercase(subject), r"confirm(ed|ation)?|booking|booked|reservation|ticket")

"""Check preview contains click/view CTAs — newsletter signal."""
_has_click(preview::String)::Bool =
    contains(lowercase(first(preview, 500)), r"click here|view (in|this|the) (browser|email|web)|open in")

"""Hash sender domain into 4 binary bits for sender identity fingerprinting."""
function _sender_hash_bits(sender::String)::NTuple{4, Bool}
    parts = split(sender, '@'; limit=2)
    domain = length(parts) == 2 ? lowercase(String(parts[2])) : lowercase(sender)
    h = hash(domain)
    (Bool(h & 1), Bool((h >> 1) & 1), Bool((h >> 2) & 1), Bool((h >> 3) & 1))
end

# ═══════════════════════════════════════
# Synthetic corpus generation
# ═══════════════════════════════════════

const SENDER_NAMES = [
    "alice@company.com", "bob@company.com", "carol@external.org",
    "dave@company.com", "eve@vendor.io", "frank@company.com",
    "grace@company.com", "heidi@external.net", "ivan@company.com",
    "noreply@marketing.biz", "karl@company.com", "lisa@company.com",
]

const SUBJECT_TEMPLATES = [
    "Q%d budget review", "Meeting re: project %d", "Newsletter #%d",
    "Urgent: server %d down", "RE: invoice #%d", "FYI: policy update %d",
    "Action required: report %d", "Quick question about task %d?",
    "Delegation: review PR #%d", "Team standup notes %d",
]

const PREVIEW_TEMPLATES = [
    "Please review the attached budget for Q%d.",
    "Can we schedule a meeting to discuss project %d?",
    "This week's newsletter. Click here to unsubscribe.",
    "Server %d is down and needs immediate attention.",
    "Invoice #%d attached for your approval.",
    "FYI - new policy update, no action needed.",
    "Please confirm you can complete report %d by Friday deadline.",
    "Quick question: what's the status on task %d?",
    "Could you review PR #%d when you get a chance?",
    "Notes from today's standup meeting.",
]

"""
    generate_email_corpus(n; rng_seed) → Vector{Email}

Generate a synthetic email corpus with controlled distributions.
Subject templates naturally produce varied raw features.
"""
function generate_email_corpus(n::Int; rng_seed::Int=42)::Vector{Email}
    rng = MersenneTwister(rng_seed)
    emails = Email[]

    # Sender frequency bands: high (colleagues), low (external/rare)
    freq_high_prob = 0.5

    for i in 1:n
        # Sender
        sender = SENDER_NAMES[mod1(i, length(SENDER_NAMES))]
        freq = rand(rng) < freq_high_prob ? 0.4 + 0.6 * rand(rng) : 0.05 + 0.3 * rand(rng)

        # Subject from template
        subject = SUBJECT_TEMPLATES[mod1(i, length(SUBJECT_TEMPLATES))]
        subject = replace(subject, r"%d" => string(i))

        # Preview from template
        preview = PREVIEW_TEMPLATES[mod1(i, length(PREVIEW_TEMPLATES))]
        preview = replace(preview, r"%d" => string(i))

        # Raw features extracted from the text (same logic as live mode)
        is_noreply = _is_noreply_sender(sender)
        is_bulk = _is_bulk_domain(sender)
        h0, h1, h2, h3 = _sender_hash_bits(sender)
        has_news = _has_news_sender(sender)
        is_reply = _is_reply(subject)
        is_fwd = _is_fwd(subject)
        has_urgent = _has_urgent_kw(subject)
        has_action = _has_action_kw(subject)
        has_money = _has_money_kw(subject)
        has_meeting = _has_meeting_kw(subject)
        has_you = _has_you(subject)
        has_new_event = _has_new_event(subject)
        has_failed = _has_failed(subject)
        has_confirmed = _has_confirmed(subject)
        has_unsub = _has_unsubscribe(preview)
        has_question = _has_question(preview)
        has_click = _has_click(preview)

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

        is_large = word_count > 8000  # ~50KB / 6 bytes per word

        push!(emails, Email(i, sender, freq,
                            is_noreply, is_bulk, h0, h1, h2, h3, has_news,
                            subject,
                            is_reply, is_fwd, has_urgent, has_action,
                            has_money, has_meeting,
                            has_you, has_new_event, has_failed, has_confirmed,
                            word_count, has_attachment, is_large,
                            hour, thread_depth,
                            has_unsub, has_question, has_click))
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
    jmap_to_email(raw, idx; sender_history, thread_counts) → (Email, jmap_id, preview)

Convert a JMAP email dict to an Email struct. Extracts raw observable
features directly from subject/preview/sender strings — no semantic
inference. The agent learns what these features mean from user feedback.
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

    # Subject and preview
    subject = String(get(raw, "subject", "(no subject)"))
    preview = String(get(raw, "preview", ""))

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

    # Raw observables — directly from text, no inference
    h0, h1, h2, h3 = _sender_hash_bits(sender)
    is_large = size_bytes > 50000

    email = Email(idx, sender, sender_freq,
                  _is_noreply_sender(sender), _is_bulk_domain(sender),
                  h0, h1, h2, h3, _has_news_sender(sender),
                  subject,
                  _is_reply(subject), _is_fwd(subject),
                  _has_urgent_kw(subject), _has_action_kw(subject),
                  _has_money_kw(subject), _has_meeting_kw(subject),
                  _has_you(subject), _has_new_event(subject),
                  _has_failed(subject), _has_confirmed(subject),
                  word_count, has_attachment, is_large,
                  hour, thread_depth,
                  _has_unsubscribe(preview), _has_question(preview),
                  _has_click(preview))

    (email, jmap_id, preview)
end

"""Parse hour (0-23) from ISO 8601 datetime string."""
function _parse_hour(iso::String)::Int
    t_pos = findfirst('T', iso)
    t_pos === nothing && return 12
    length(iso) < t_pos + 2 && return 12
    h = tryparse(Int, iso[t_pos+1:t_pos+2])
    h === nothing ? 12 : clamp(h, 0, 23)
end

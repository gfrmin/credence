"""
    jmap_client.jl — Minimal Julia JMAP client for Fastmail

Functional interface: all functions take a JMAPSession and return data.
No mutation, no global state. Auth via ENV["FASTMAIL_TOKEN"] bearer token.

Reference: inbox/src/inbox/jmap.py
"""

using HTTP, JSON3

const JMAP_DISCOVERY_URL = "https://api.fastmail.com/.well-known/jmap"

const EMAIL_PROPERTIES = [
    "id", "subject", "from", "preview", "receivedAt",
    "threadId", "mailboxIds", "keywords", "size", "hasAttachment"
]

struct JMAPSession
    api_url::String
    account_id::String
    token::String
end

"""
    discover_session(token) → JMAPSession

GET the JMAP discovery endpoint to obtain the API URL and account ID.
"""
function discover_session(token::String)::JMAPSession
    resp = HTTP.get(JMAP_DISCOVERY_URL;
        headers=["Authorization" => "Bearer $token"],
        redirect=true)
    data = JSON3.read(resp.body)
    api_url = String(data.apiUrl)
    account_id = String(data.primaryAccounts[Symbol("urn:ietf:params:jmap:mail")])
    JMAPSession(api_url, account_id, token)
end

"""
    discover_session() → JMAPSession

Discover session using ENV["FASTMAIL_TOKEN"], falling back to secret-tool.
"""
function discover_session()::JMAPSession
    token = get(ENV, "FASTMAIL_TOKEN", "")
    if isempty(token)
        try
            token = String(strip(read(`secret-tool lookup service jmap account guy@publicdatamarket.com`, String)))
        catch; end
    end
    isempty(token) && error("FASTMAIL_TOKEN not set and secret-tool lookup failed")
    discover_session(token)
end

"""
    jmap_call(session, method_calls) → Vector

POST a JMAP request and return the methodResponses array.
Each method call is [method_name, args_dict, request_id].
"""
function jmap_call(session::JMAPSession, method_calls::Vector)::Vector
    body = JSON3.write(Dict(
        "using" => ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
        "methodCalls" => method_calls
    ))
    resp = HTTP.post(session.api_url,
        ["Authorization" => "Bearer $(session.token)",
         "Content-Type" => "application/json"],
        body)
    data = JSON3.read(resp.body)
    responses = data.methodResponses
    for r in responses
        name = String(r[1])
        if endswith(name, "/error") || name == "error"
            error("JMAP error in $name: $(r[2])")
        end
    end
    collect(responses)
end

"""
    get_mailbox_id(session, role) → String

Find a mailbox by its JMAP role (e.g. "inbox", "archive").
"""
function get_mailbox_id(session::JMAPSession, role::String)::String
    resp = jmap_call(session, [
        ["Mailbox/get", Dict("accountId" => session.account_id, "ids" => nothing), "m0"]
    ])
    mailboxes = resp[1][2]["list"]
    for m in mailboxes
        if get(m, "role", nothing) == role
            return String(m["id"])
        end
    end
    error("No mailbox with role '$role' found")
end

"""
    query_inbox_emails(session; limit=100, exclude_keywords=[]) → Vector{Dict}

Fetch inbox emails sorted by receivedAt (newest first), excluding emails
with any of the specified keywords. Returns full email dicts with EMAIL_PROPERTIES.
"""
function query_inbox_emails(session::JMAPSession;
                            limit::Int=100,
                            exclude_keywords::Vector{String}=String[])::Vector
    inbox_id = get_mailbox_id(session, "inbox")

    # Build filter: inbox + exclude keywords
    conditions = Dict{String,Any}[Dict("inMailbox" => inbox_id)]
    for kw in exclude_keywords
        push!(conditions, Dict("notKeyword" => kw))
    end
    filter = if length(conditions) == 1
        conditions[1]
    else
        Dict("operator" => "AND", "conditions" => conditions)
    end

    # Query email IDs with pagination
    all_ids = String[]
    position = 0
    batch_size = min(500, limit)
    while length(all_ids) < limit
        remaining = limit - length(all_ids)
        batch_size = min(500, remaining)
        resp = jmap_call(session, [
            ["Email/query", Dict(
                "accountId" => session.account_id,
                "filter" => filter,
                "sort" => [Dict("property" => "receivedAt", "isAscending" => false)],
                "position" => position,
                "limit" => batch_size
            ), "q0"]
        ])
        ids = [String(id) for id in resp[1][2]["ids"]]
        isempty(ids) && break
        append!(all_ids, ids)
        position += length(ids)
        length(ids) < batch_size && break
    end

    isempty(all_ids) && return Dict[]

    # Fetch full email data in chunks of 100
    all_emails = []
    for i in 1:100:length(all_ids)
        chunk = all_ids[i:min(i+99, length(all_ids))]
        resp = jmap_call(session, [
            ["Email/get", Dict(
                "accountId" => session.account_id,
                "ids" => chunk,
                "properties" => EMAIL_PROPERTIES
            ), "g0"]
        ])
        append!(all_emails, resp[1][2]["list"])
    end
    all_emails
end

"""
    move_emails(session, ids, dest_mailbox_id)

Move emails to a destination mailbox via Email/set.
"""
function move_emails(session::JMAPSession, ids::Vector{String}, dest_mailbox_id::String)
    for i in 1:50:length(ids)
        chunk = ids[i:min(i+49, length(ids))]
        update = Dict(eid => Dict("mailboxIds" => Dict(dest_mailbox_id => true))
                       for eid in chunk)
        resp = jmap_call(session, [
            ["Email/set", Dict(
                "accountId" => session.account_id,
                "update" => update
            ), "u0"]
        ])
        not_updated = get(resp[1][2], "notUpdated", nothing)
        not_updated !== nothing && error("Failed to move emails: $not_updated")
    end
end

"""
    set_keyword(session, ids, keyword)

Set a keyword (e.g. "\$flagged") on emails via Email/set.
"""
function set_keyword(session::JMAPSession, ids::Vector{String}, keyword::String)
    for i in 1:50:length(ids)
        chunk = ids[i:min(i+49, length(ids))]
        update = Dict(eid => Dict("keywords/$keyword" => true) for eid in chunk)
        resp = jmap_call(session, [
            ["Email/set", Dict(
                "accountId" => session.account_id,
                "update" => update
            ), "k0"]
        ])
        not_updated = get(resp[1][2], "notUpdated", nothing)
        not_updated !== nothing && error("Failed to set keyword: $not_updated")
    end
end

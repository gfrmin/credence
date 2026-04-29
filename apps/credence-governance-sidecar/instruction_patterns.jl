# Role: body
#
# Compaction-survival instruction patterns.
# Seven regex patterns from the Move 2 design doc §5.4 table.
# Loaded by server.jl before brain.jl.

const DESTRUCTIVE_CLASSES = Set(["delete", "deploy", "privileged-exec", "dependency"])

struct InstructionPattern
    regex::Regex
    action_class::String
    id::String
end

const INSTRUCTION_PATTERNS = [
    InstructionPattern(
        r"confirm\s+before\s+(?:deleting|removing|dropping)"i,
        "delete",
        "confirm-before-delete",
    ),
    InstructionPattern(
        r"(?:don'?t|do\s+not|never)\s+(?:delete|remove|drop)\b"i,
        "delete",
        "negation-delete",
    ),
    InstructionPattern(
        r"(?:always|must)\s+ask\s+before\b"i,
        "any-destructive",
        "ask-before-any",
    ),
    InstructionPattern(
        r"confirm\s+before\s+(?:pushing|deploying|merging)"i,
        "deploy",
        "confirm-before-deploy",
    ),
    InstructionPattern(
        r"(?:don'?t|do\s+not|never)\s+(?:push|deploy|merge)\s+(?:to|into)\s+(?:main|master|prod)"i,
        "deploy",
        "negation-deploy-to-protected",
    ),
    InstructionPattern(
        r"(?:don'?t|do\s+not|never)\s+run\b.*(?:sudo|as\s+root)"i,
        "privileged-exec",
        "negation-run-privileged",
    ),
    InstructionPattern(
        r"(?:don'?t|do\s+not|never)\s+(?:install|add|upgrade)\b.*package"i,
        "dependency",
        "negation-install-package",
    ),
]

function instruction_matches_category(instruction_class::String, candidate_category::String)::Bool
    instruction_class == candidate_category && return true
    instruction_class == "any-destructive" && candidate_category in DESTRUCTIVE_CLASSES && return true
    false
end

function extract_message_text(msg)::String
    content = if msg isa AbstractDict
        c = get(msg, "content", get(msg, :content, nothing))
        if c isa AbstractVector
            join((get(block, "text", get(block, :text, "")) for block in c
                  if (get(block, "type", get(block, :type, "")) == "text")), " ")
        else
            c
        end
    elseif msg isa AbstractString
        msg
    else
        nothing
    end
    content !== nothing ? string(content) : ""
end

function match_instructions(messages::Vector)::Vector{Tuple{String, String}}
    results = Tuple{String, String}[]
    seen = Set{String}()

    for msg in messages
        text = extract_message_text(msg)
        isempty(text) && continue

        for pattern in INSTRUCTION_PATTERNS
            pattern.id in seen && continue
            m = match(pattern.regex, text)
            if m !== nothing
                push!(seen, pattern.id)
                push!(results, (m.match, pattern.action_class))
            end
        end
    end

    results
end

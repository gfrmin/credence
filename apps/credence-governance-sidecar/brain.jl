# Role: body
#
# Governance brain: posterior management and EU calculation.
# Loaded by server.jl; uses Credence substrate types.

using ..Credence
using Dates

# ── Category inference ──

const CATEGORY_CODE_EXTENSIONS = Set([
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".jl", ".cpp", ".c",
    ".h", ".hpp", ".java", ".kt", ".scala", ".rb", ".php", ".swift", ".cs",
    ".lua", ".zig", ".nim", ".ex", ".exs", ".erl", ".hs", ".ml", ".r",
])
const CATEGORY_DOC_EXTENSIONS = Set([".md", ".rst", ".txt", ".adoc", ".org"])

const DELETE_PATTERNS = [r"\brm\b", r"\bgit\s+rm\b", r"\bunlink\b", r"\brmdir\b"]
const DEPLOY_PATTERNS = [r"\bdocker\s+push\b", r"\bkubectl\s+apply\b", r"\bdeploy\b",
                         r"\bhelm\s+install\b", r"\bhelm\s+upgrade\b"]
const PRIVILEGED_PATTERNS = [r"\bsudo\b", r"\bas\s+root\b"]
const DEPENDENCY_PATTERNS = [r"\bnpm\s+install\b", r"\bpip\s+install\b", r"\bcargo\s+add\b",
                             r"\byarn\s+add\b", r"\bbun\s+add\b", r"\buv\s+add\b",
                             r"\bapt\s+install\b", r"\bbrew\s+install\b"]
const VERSION_CONTROL_PATTERNS = [r"\bgit\b"]

function infer_category(tool_name::AbstractString, params::Dict{String,Any})::String
    if tool_name in ("Edit", "Write", "Read")
        path = get(params, "file_path", get(params, "path", ""))
        ext = lowercase(splitext(string(path))[2])
        ext in CATEGORY_CODE_EXTENSIONS && return "code"
        ext in CATEGORY_DOC_EXTENSIONS && return "documentation"
        return "code"
    end

    if tool_name == "Bash"
        cmd = string(get(params, "command", ""))
        any(p -> occursin(p, cmd), DELETE_PATTERNS) && return "delete"
        any(p -> occursin(p, cmd), PRIVILEGED_PATTERNS) && return "privileged-exec"
        any(p -> occursin(p, cmd), DEPLOY_PATTERNS) && return "deploy"
        any(p -> occursin(p, cmd), DEPENDENCY_PATTERNS) && return "dependency"
        any(p -> occursin(p, cmd), VERSION_CONTROL_PATTERNS) && return "version-control"
    end

    return "generic"
end

# ── Duration budget ──

struct BudgetTable
    budgets::Dict{String,Int}
    default::Int
end

function load_budget_table(path::AbstractString)::BudgetTable
    data = JSON3.read(read(path, String), Dict{String,Any})
    default_val = get(data, "default", 15000)
    budgets = Dict{String,Int}()
    for (k, v) in data
        k == "default" && continue
        budgets[string(k)] = Int(v)
    end
    BudgetTable(budgets, Int(default_val))
end

function get_budget(table::BudgetTable, tool_name::AbstractString, category::AbstractString)::Int
    key = tool_name * ":" * category
    haskey(table.budgets, key) && return table.budgets[key]
    haskey(table.budgets, tool_name) && return table.budgets[tool_name]
    return table.default
end

function classify_outcome(table::BudgetTable, tool_name::AbstractString, category::AbstractString,
                          error_str::Union{Nothing,AbstractString}, duration_ms::Union{Nothing,Real})::Bool
    error_str !== nothing && !isempty(error_str) && return false
    duration_ms !== nothing && duration_ms > get_budget(table, tool_name, category) && return false
    return true
end

# ── Posterior state ──

struct PosteriorKey
    tool_name::String
    category::String
end

Base.hash(k::PosteriorKey, h::UInt) = hash(k.tool_name, hash(k.category, h))
Base.:(==)(a::PosteriorKey, b::PosteriorKey) = a.tool_name == b.tool_name && a.category == b.category

mutable struct BrainState
    tool_outcomes::Dict{PosteriorKey, BetaMeasure}
    model_quality::Dict{String, Dict{String, BetaMeasure}}
    observation_count::Int
    budget_table::BudgetTable
    prototype_fallback_enabled::Bool
    max_repetitions::Int
    registered_instructions::Vector{Dict{String,Any}}
end

function make_brain_state(budget_table::BudgetTable;
                          prototype_fallback_enabled::Bool=true,
                          max_repetitions::Int=3)
    BrainState(
        Dict{PosteriorKey, BetaMeasure}(),
        Dict{String, Dict{String, BetaMeasure}}(),
        0,
        budget_table,
        prototype_fallback_enabled,
        max_repetitions,
        Dict{String,Any}[],
    )
end

function get_posterior(state::BrainState, tool_name::AbstractString, category::AbstractString)::BetaMeasure
    key = PosteriorKey(tool_name, category)
    get!(state.tool_outcomes, key, BetaMeasure(1.0, 1.0))
end

function update_posterior!(state::BrainState, tool_name::AbstractString, category::AbstractString, success::Bool)
    key = PosteriorKey(tool_name, category)
    prior = get!(state.tool_outcomes, key, BetaMeasure(1.0, 1.0))
    k = Kernel(
        Interval(0.0, 1.0),
        BOOLEAN_SPACE,
        θ -> θ,
        (θ, obs) -> obs ? log(θ) : log(1.0 - θ),
        likelihood_family = BetaBernoulli(),
    )
    state.tool_outcomes[key] = condition(prior, k, success)
    state.observation_count += 1
end

# ── EU calculation ──

const READ_LIKE_TOOLS = Set(["Read", "Grep", "LS", "Glob"])

struct EUResult
    decision::String
    action::String
    reason::String
    signals::Dict{String,Any}
    eu_proceed::Float64
    eu_halt::Float64
    eu_downgrade::Float64
    eu_route::Float64
    eu_escalate::Float64
end

function compute_eu(state::BrainState, tool_name::AbstractString, category::AbstractString)::EUResult
    m = get_posterior(state, tool_name, category)
    # credence-lint: allow — precedent:expect-through-accessor — alpha/beta needed for threshold derivation and signal reporting
    α = m.alpha
    β = m.beta
    concentration = α + β

    # credence-lint: allow — precedent:display-arithmetic — EU is host-level decision-theoretic computation, not belief modification
    eu_proceed = expect(m, θ -> θ * 1.0 + (1.0 - θ) * (-1.0))
    eu_halt = 0.0
    v = variance(m)
    uncertainty = 2.0 * sqrt(v)
    eu_escalate = uncertainty * 0.5 - (1.0 - uncertainty) * 0.1  # credence-lint: allow — precedent:display-arithmetic — host-level EU, not belief modification

    alt_category = tool_name == "Bash" ? "code" : "generic"
    alt_m = get_posterior(state, "Read", alt_category)
    eu_downgrade = expect(alt_m, θ -> θ * 0.9 + (1.0 - θ) * (-0.8))  # credence-lint: allow — precedent:display-arithmetic — host-level EU computation

    eu_route = eu_proceed * 0.95  # credence-lint: allow — precedent:display-arithmetic — host-level EU scaling

    comparison_p = expect(m, θ₁ -> expect(alt_m, θ₂ -> θ₂ > θ₁ ? 1.0 : 0.0))  # credence-lint: allow — precedent:display-arithmetic — P(alt > candidate) for threshold check
    downgrade_threshold = 1.0 - 1.0 / concentration  # credence-lint: allow — precedent:display-arithmetic — Amendment 1 threshold derivation

    cv = v > 0 && abs(eu_proceed) > 1e-12 ? sqrt(v) / abs(eu_proceed) : (v > 0 ? 1e6 : 0.0)  # credence-lint: allow — precedent:display-arithmetic — Amendment 1 CV threshold
    escalate_threshold = 1.0 / sqrt(concentration)  # credence-lint: allow — precedent:display-arithmetic — Amendment 1 threshold derivation

    escalate_fires = cv > escalate_threshold && uncertainty > 0.3

    # credence-lint: allow — precedent:display-arithmetic — instruction-based escalation elevation
    instruction_boost = 0.0
    for inst in state.registered_instructions
        if instruction_matches_category(string(inst["action_class"]), category)
            ps = Float64(inst["prior_strength"])
            approvals = Int(get(inst, "approvals", 0))
            denials = Int(get(inst, "denials", 0))
            denial_rate = (1.0 + denials) / (2.0 + approvals + denials)
            instruction_boost = max(instruction_boost, ps * denial_rate)
            escalate_fires = true
        end
    end
    eu_escalate += instruction_boost

    # credence-lint: allow — precedent:display-arithmetic — host-level argmax decision logic
    downgrade_fires = comparison_p > downgrade_threshold && eu_downgrade > eu_proceed && eu_proceed > eu_halt

    base_eus = Dict(
        "proceed" => eu_proceed,
        "halt" => eu_halt,
        "route" => eu_route,
    )

    # credence-lint: allow — precedent:display-arithmetic — host-level argmax decision logic
    if escalate_fires && eu_escalate > eu_halt && eu_escalate > eu_proceed
        decision = "escalate"
    elseif downgrade_fires
        decision = "downgrade"
    else
        decision = argmax(base_eus)
    end

    if decision == "halt" || decision == "downgrade"
        action = "block"
    elseif decision == "escalate"
        action = "escalate"
    else
        action = "proceed"
    end

    reason = _make_reason(decision, tool_name, α, β, comparison_p, cv)

    signals = Dict{String,Any}(
        "alpha" => α,
        "beta" => β,
        "comparison_p" => comparison_p,
        "cv" => cv,
        "eu_proceed" => eu_proceed,
        "eu_halt" => eu_halt,
        "eu_downgrade" => eu_downgrade,
        "eu_escalate" => eu_escalate,
    )

    EUResult(decision, action, reason, signals, eu_proceed, eu_halt, eu_downgrade, eu_route, eu_escalate)
end

function _make_reason(decision::AbstractString, tool_name::AbstractString,
                      α::Float64, β::Float64, comparison_p::Float64, cv::Float64)::String
    if decision == "halt"
        "Expected utility of continuing has degraded below idle. " *
        "Tool '$tool_name' posterior: Beta($(round(α, digits=1)), $(round(β, digits=1))), " *
        "EU(proceed) < EU(halt)."
    elseif decision == "downgrade"
        "EU of alternative exceeds EU of '$tool_name' " *
        "(P=$(round(comparison_p, digits=3))). Suggestion: use a read-like tool instead."
    elseif decision == "escalate"
        "The proposed action '$tool_name' has uncertain expected utility " *
        "(CV=$(round(cv, digits=3)), posterior Beta($(round(α, digits=1)), $(round(β, digits=1)))). " *
        "Confirm to proceed."
    elseif decision == "route"
        "Posterior supports using a cheaper model for '$tool_name' without expected quality loss."
    else
        ""
    end
end

# ── Instruction registration ──

const DEFAULT_PRIOR_STRENGTH = 5.0

function register_instruction!(state::BrainState, pattern_id::String, action_class::String;
                                prior_strength::Float64=DEFAULT_PRIOR_STRENGTH)
    for inst in state.registered_instructions
        if inst["pattern"] == pattern_id && inst["action_class"] == action_class
            inst["last_seen"] = Dates.format(now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")
            return false
        end
    end

    push!(state.registered_instructions, Dict{String,Any}(
        "pattern" => pattern_id,
        "action_class" => action_class,
        "registered_at" => Dates.format(now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "last_seen" => Dates.format(now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "prior_strength" => prior_strength,
        "approvals" => 0,
        "denials" => 0,
    ))
    true
end

# ── Prototype fallback: repetition counting ──

function canonical_key(tool_name::AbstractString, params::Dict)
    sorted_params = sort(collect(params), by=first)
    return "$tool_name:$(JSON3.write(sorted_params))"
end

function count_repetitions(history::Vector{Dict{String,Any}}, tool_name::AbstractString, params::Dict)
    key = canonical_key(tool_name, params)
    count = 0
    for entry in history
        entry_key = canonical_key(
            get(entry, "toolName", ""),
            get(entry, "params", Dict{String,Any}())
        )
        if entry_key == key
            count += 1
        end
    end
    return count
end

function check_prototype_fallback(state::BrainState, history::Vector{Dict{String,Any}},
                                  tool_name::AbstractString, params::Dict{String,Any})
    !state.prototype_fallback_enabled && return nothing
    tool_name in READ_LIKE_TOOLS && return nothing
    count = count_repetitions(history, tool_name, params)
    if count >= state.max_repetitions
        return Dict{String,Any}(
            "action" => "block",
            "decision" => "halt",
            "reason" => "Loop detected: '$tool_name' with identical arguments has been called $count times (threshold: $(state.max_repetitions)). Halting to prevent runaway loop.",
            "signals" => Dict{String,Any}(),
            "requireApproval" => nothing,
        )
    end
    return nothing
end

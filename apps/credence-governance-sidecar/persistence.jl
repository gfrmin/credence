# Role: body
#
# State persistence for the governance sidecar.
# Loaded by server.jl after brain.jl.

using Dates
using Random

const SIDECAR_SCHEMA_VERSION = 1
const DEFAULT_STATE_DIR = joinpath(homedir(), ".credence", "state")
const STATE_FILENAME = "posterior.json"

function get_state_dir()::String
    get(ENV, "CREDENCE_STATE_DIR", DEFAULT_STATE_DIR)
end

function get_state_path()::String
    joinpath(get_state_dir(), STATE_FILENAME)
end

function generate_uuid4()::String
    bytes = rand(UInt8, 16)
    bytes[7] = (bytes[7] & 0x0f) | 0x40  # version 4
    bytes[9] = (bytes[9] & 0x3f) | 0x80  # variant 1
    hex = bytes2hex(bytes)
    "$(hex[1:8])-$(hex[9:12])-$(hex[13:16])-$(hex[17:20])-$(hex[21:32])"
end

function now_iso8601()::String
    Dates.format(now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")
end

function serialize_tool_posteriors(outcomes::Dict{PosteriorKey, BetaMeasure})::Dict{String, Any}
    result = Dict{String, Any}()
    for (key, m) in outcomes
        k = "$(key.tool_name):$(key.category)"
        # credence-lint: allow — precedent:expect-through-accessor — serialising alpha/beta for state persistence
        result[k] = Dict{String, Any}("alpha" => m.alpha, "beta" => m.beta)
    end
    result
end

function deserialize_tool_posteriors(data)::Dict{PosteriorKey, BetaMeasure}
    result = Dict{PosteriorKey, BetaMeasure}()
    for (k, v) in data
        parts = split(string(k), ":", limit=2)
        length(parts) == 2 || error("Malformed posterior key: $k (expected 'tool_name:category')")
        tool_name, category = String(parts[1]), String(parts[2])
        alpha = Float64(v isa Dict ? v["alpha"] : v[:alpha])
        beta = Float64(v isa Dict ? v["beta"] : v[:beta])
        (alpha <= 0 || beta <= 0) && error("Beta parameters must be positive reals for key '$k': alpha=$alpha, beta=$beta")
        result[PosteriorKey(tool_name, category)] = BetaMeasure(alpha, beta)
    end
    result
end

function serialize_model_posteriors(quality::Dict{String, Dict{String, BetaMeasure}})::Dict{String, Any}
    result = Dict{String, Any}()
    for (model_id, categories) in quality
        for (category, m) in categories
            k = "$model_id:$category"
            # credence-lint: allow — precedent:expect-through-accessor — serialising alpha/beta for state persistence
            result[k] = Dict{String, Any}("alpha" => m.alpha, "beta" => m.beta)
        end
    end
    result
end

function deserialize_model_posteriors(data)::Dict{String, Dict{String, BetaMeasure}}
    result = Dict{String, Dict{String, BetaMeasure}}()
    for (k, v) in data
        parts = split(string(k), ":", limit=2)
        length(parts) == 2 || error("Malformed model posterior key: $k (expected 'model_id:category')")
        model_id, category = String(parts[1]), String(parts[2])
        alpha = Float64(v isa Dict ? v["alpha"] : v[:alpha])
        beta = Float64(v isa Dict ? v["beta"] : v[:beta])
        (alpha <= 0 || beta <= 0) && error("Beta parameters must be positive reals for model key '$k': alpha=$alpha, beta=$beta")
        if !haskey(result, model_id)
            result[model_id] = Dict{String, BetaMeasure}()
        end
        result[model_id][category] = BetaMeasure(alpha, beta)
    end
    result
end

function reconstruct_observation_count(outcomes::Dict{PosteriorKey, BetaMeasure})::Int
    # credence-lint: allow — precedent:expect-through-accessor — alpha/beta needed to reconstruct observation count from persisted posteriors
    sum(Int(round(m.alpha + m.beta - 2.0)) for m in values(outcomes); init=0)
end

function save_sidecar_state(brain::BrainState, user_id::String, created_at::String)
    dir = get_state_dir()
    mkpath(dir)
    chmod(dir, 0o700)

    data = Dict{String, Any}(
        "schema_version" => SIDECAR_SCHEMA_VERSION,
        "user_id" => user_id,
        "created_at" => created_at,
        "updated_at" => now_iso8601(),
        "tool_posteriors" => serialize_tool_posteriors(brain.tool_outcomes),
        "model_posteriors" => serialize_model_posteriors(brain.model_quality),
        "registered_instructions" => Any[],
    )

    filepath = get_state_path()
    tmp_path = filepath * ".tmp"
    open(tmp_path, "w") do io
        JSON3.pretty(io, data)
    end
    chmod(tmp_path, 0o600)
    Base.Filesystem.rename(tmp_path, filepath)
    chmod(filepath, 0o600)
end

function validate_state_file(data, filepath::String)
    sv = get(data, "schema_version", nothing)
    sv === nothing && error("State file at $filepath missing required field 'schema_version'")
    sv = Int(sv)
    sv > SIDECAR_SCHEMA_VERSION && error(
        "State file schema version $sv is newer than this sidecar supports " *
        "(v$SIDECAR_SCHEMA_VERSION); please upgrade the sidecar"
    )

    for field in ("user_id", "tool_posteriors", "model_posteriors")
        get(data, field, nothing) === nothing && error(
            "State file at $filepath missing required field '$field'"
        )
    end
end

function load_sidecar_state!(brain::BrainState)::@NamedTuple{user_id::String, created_at::String}
    filepath = get_state_path()

    if !isfile(filepath)
        user_id = generate_uuid4()
        created_at = now_iso8601()
        save_sidecar_state(brain, user_id, created_at)
        return (; user_id, created_at)
    end

    raw = try
        read(filepath, String)
    catch e
        error("Cannot read state file at $filepath: $(sprint(showerror, e))")
    end

    data = try
        JSON3.read(raw, Dict{String, Any})
    catch e
        error("Malformed JSON in state file at $filepath: $(sprint(showerror, e))")
    end

    validate_state_file(data, filepath)

    user_id = String(data["user_id"])
    created_at = String(get(data, "created_at", now_iso8601()))

    brain.tool_outcomes = deserialize_tool_posteriors(data["tool_posteriors"])
    brain.model_quality = deserialize_model_posteriors(data["model_posteriors"])
    brain.observation_count = reconstruct_observation_count(brain.tool_outcomes)

    (; user_id, created_at)
end

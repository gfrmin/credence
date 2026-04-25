# Role: brain-side application
"""
    state_persistence.jl — Save/load email agent state across sessions

AgentState contains compiled closures (CompiledKernel) that can't be
serialized. Strategy: save the serializable parts, recompile on load.
CostModel beliefs (NormalGammaMeasures) are fully serializable.
"""

using Serialization
using Credence: MixtureMeasure, TaggedBetaMeasure, Interval, Measure
using Credence: NormalGammaMeasure
using Credence: Grammar, Program, CompiledKernel, AgentState
using Credence: compile_kernel

const DEFAULT_STATE_DIR = joinpath(homedir(), ".credence")
const DEFAULT_STATE_PATH = joinpath(DEFAULT_STATE_DIR, "email_agent_state.jls")

"""
    save_email_state(filepath, state, cost_model; sender_history, thread_counts, temporal_state)

Serialize all recoverable parts of the agent state + cost model.
"""
function save_email_state(
    filepath::String,
    state::AgentState,
    cost_model::CostModel;
    sender_history::Dict{String, Int}=Dict{String, Int}(),
    thread_counts::Dict{String, Int}=Dict{String, Int}(),
    temporal_state::Dict{Symbol, Any}=Dict{Symbol, Any}()
)
    # Extract serializable parts from belief
    n = length(state.belief.components)
    # credence-lint: allow — precedent:expect-through-accessor — serialization: extracting Beta parameters and log-weights for JSON persistence
    alphas = Float64[state.belief.components[i].beta.alpha for i in 1:n]
    betas = Float64[state.belief.components[i].beta.beta for i in 1:n]
    log_weights = copy(state.belief.log_weights)  # credence-lint: allow — precedent:expect-through-accessor — serialization: log-weights for JSON persistence

    data = Dict{Symbol, Any}(
        :alphas => alphas,
        :betas => betas,
        :log_weights => log_weights,
        :metadata => state.metadata,
        :all_programs => state.all_programs,
        :grammars => state.grammars,
        :current_max_depth => state.current_max_depth,
        :cost_beliefs => cost_model.beliefs,
        :time_value => cost_model.time_value,
        :sender_history => sender_history,
        :thread_counts => thread_counts,
        :temporal_state => temporal_state,
    )

    mkpath(dirname(filepath))
    open(io -> serialize(io, data), filepath, "w")
end

"""
    load_email_state(filepath) → (state, cost_model, sender_history, thread_counts, temporal_state)

Deserialize and recompile. Returns nothing if file doesn't exist.
"""
function load_email_state(filepath::String)
    isfile(filepath) || return nothing

    data = open(io -> deserialize(io), filepath, "r")

    alphas = data[:alphas]::Vector{Float64}
    betas = data[:betas]::Vector{Float64}
    log_weights = data[:log_weights]::Vector{Float64}
    metadata = data[:metadata]::Vector{Tuple{Int, Int}}
    all_programs = data[:all_programs]::Vector{Program}
    grammars = data[:grammars]::Dict{Int, Grammar}
    max_depth = data[:current_max_depth]::Int

    n = length(alphas)

    # Reconstruct belief
    components = Measure[
        TaggedBetaMeasure(Interval(0.0, 1.0), i,
            Credence.Ontology.BetaPrevision(alphas[i], betas[i]))
        for i in 1:n
    ]
    belief = MixtureMeasure(Interval(0.0, 1.0), components, log_weights)

    # Recompile kernels from ASTs
    compiled_kernels = CompiledKernel[
        compile_kernel(all_programs[i], grammars[metadata[i][1]], metadata[i][2])
        for i in 1:n
    ]

    state = AgentState(belief, metadata, compiled_kernels, all_programs, grammars, max_depth)

    # Reconstruct cost model
    cost_beliefs = get(data, :cost_beliefs, default_cost_model().beliefs)
    time_value = get(data, :time_value, 0.02)
    cost_model = CostModel(cost_beliefs, time_value)

    sender_history = get(data, :sender_history, Dict{String, Int}())
    thread_counts = get(data, :thread_counts, Dict{String, Int}())
    temporal_state = get(data, :temporal_state, Dict{Symbol, Any}(:recent => Dict{Symbol, Float64}[]))

    (state=state, cost_model=cost_model,
     sender_history=sender_history, thread_counts=thread_counts,
     temporal_state=temporal_state)
end

# Domain Interface

Each domain must provide the following to use the program-space inference layer (Tier 2):

```julia
# Required from a domain module:

# 1. Seed grammars — domain-specific terminal alphabet
generate_seed_grammars() → Vector{Grammar}

# 2. Sensor projection — map true state to sensor vector
project(true_state::Vector{Float64}, config::SensorConfig; ...) → Vector{Float64}

# 3. Available source indices — valid channel indices for perturbation
available_source_indices() → Vector{Int}

# 4. Outcome classification — map domain events to utility values
classify_outcome(event) → Float64

# 5. Host driver — orchestrate the agent loop
#    Uses Tier 2 exports: enumerate_programs, compile_kernel,
#    AgentState, sync_prune!, sync_truncate!, perturb_grammar, etc.
run_agent(; kwargs...) → (metrics, state, grammar_pool)
```

See `grid_world/` for a complete implementation and `email_agent/` for stubs.

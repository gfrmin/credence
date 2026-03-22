# Domain Interface

Each domain must provide the following to use the program-space inference layer (Tier 2):

```julia
# Required from a domain module:

# 1. Seed grammars — domain-specific terminal alphabet
#    Each grammar declares which named features it attends to.
generate_seed_grammars() → Vector{Grammar}

# 2. Feature extraction — produce named features as Dict{Symbol, Float64}
extract_features(observation) → Dict{Symbol, Float64}

# 3. Available features — full set of named features for perturbation
const ALL_DOMAIN_FEATURES = Set{Symbol}(...)

# 4. Outcome classification — map domain events to utility values
classify_outcome(event) → Float64

# 5. Host driver — orchestrate the agent loop
#    Uses Tier 2 exports: enumerate_programs, compile_kernel,
#    AgentState, sync_prune!, sync_truncate!, perturb_grammar, etc.
run_agent(; kwargs...) → (metrics, state, grammar_pool)
```

See `grid_world/` for a complete implementation and `email_agent/` for the email domain.

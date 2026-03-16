#!/usr/bin/env julia
"""
    host_credence_agent.jl — Julia host driver for the credence agent

Loads the DSL agent specification, drives the question loop,
maintains per-tool per-category reliability beliefs across questions,
and simulates tool responses.

Uses ontology types: CategoricalMeasure, BetaMeasure, MixtureMeasure, Kernel.
All decision-theoretic computation goes through the ontology module.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: expect, condition, draw, optimise, value, weights, mean
using Credence: CategoricalMeasure, BetaMeasure, MixtureMeasure, ProductMeasure
using Credence: Finite, Interval, ProductSpace, Kernel, FactorSelector, Measure
using Credence: density, log_density_at, save_state, load_state, prune
using Credence: initial_rel_state, initial_cov_state, marginalize_betas, update_beta_state

# ─── State file ───

const STATE_FILE = joinpath(@__DIR__, ".credence_state.bin")

# ─── Tool trait boundary ───

abstract type AbstractTool end

tool_name(t::AbstractTool)::String = error("not implemented")
tool_cost(t::AbstractTool)::Float64 = error("not implemented")
tool_coverage(t::AbstractTool)::Vector{Float64} = error("not implemented")
query_tool(t::AbstractTool, true_cat::Int, true_answer::Int, n_answers::Int)::Union{Int, Nothing} =
    error("not implemented")

struct SimulatedTool <: AbstractTool
    name::String
    cost::Float64
    coverage::Vector{Float64}     # P(answers | category), length n_categories
    reliability::Vector{Float64}  # true reliability per category (simulation only)
end

tool_name(t::SimulatedTool) = t.name
tool_cost(t::SimulatedTool) = t.cost
tool_coverage(t::SimulatedTool) = t.coverage

function query_tool(t::SimulatedTool, true_cat::Int, true_answer::Int, n_answers::Int)
    rand() < t.coverage[true_cat + 1] || return nothing
    if rand() < t.reliability[true_cat + 1]
        true_answer
    else
        rand(setdiff(0:n_answers-1, [true_answer]))
    end
end

# ─── Host-level helpers (from Credence module) ───

"""Thompson sampling: draw (host) + optimise (ontology)."""
function thompson_action(posterior::Measure, actions::Finite, pref)
    h = draw(posterior)
    point = CategoricalMeasure(Finite([h]))
    optimise(point, actions, pref)
end

# ─── Main loop ───

"""Run the credence agent for n_questions, returning (total_score, total_cost)."""
function run_agent(; n_questions=50)
    # ─── Load DSL agent ───
    env = load_dsl(read(joinpath(@__DIR__, "credence_agent.bdsl"), String))
    agent_step_fn = env[Symbol("agent-step")]
    update_on_response_fn = env[Symbol("update-on-response")]
    update_category_fn = env[Symbol("update-category")]

    # ─── Configuration ───
    n_answers = 4
    n_categories = 5
    answers = Float64[0, 1, 2, 3]
    reward_correct, penalty_wrong, reward_abstain = 10.0, -5.0, 0.0

    tools = [
        SimulatedTool("expert",     2.0, [0.9, 0.7, 0.8, 0.6, 0.7], [0.85, 0.70, 0.80, 0.60, 0.70]),
        SimulatedTool("crowd",      0.5, [0.6, 0.8, 0.5, 0.9, 0.6], [0.60, 0.80, 0.50, 0.90, 0.60]),
        SimulatedTool("generalist", 1.0, [0.8, 0.8, 0.8, 0.8, 0.8], [0.75, 0.75, 0.75, 0.75, 0.75]),
    ]
    n_tools = length(tools)

    answer_space = Finite(answers)
    submit_value = reward_correct
    abstain_value = reward_abstain

    # ─── Persistent state (load if available) ───
    rel_states = [initial_rel_state(n_categories) for _ in 1:n_tools]
    cov_states = [initial_cov_state(n_categories, tool_coverage(tools[t])) for t in 1:n_tools]
    cat_belief = CategoricalMeasure(Finite(Float64.(collect(0:n_categories-1))))
    total_score = 0.0
    total_cost = 0.0

    if isfile(STATE_FILE)
        try
            state = load_state(STATE_FILE)
            loaded_rel = state[:rel_beliefs]
            if loaded_rel isa Vector && !isempty(loaded_rel) && first(loaded_rel) isa MixtureMeasure
                rel_states = loaded_rel
                cat_belief = state[:cat_belief]
                total_score = state[:total_score]
                total_cost = state[:total_cost]
                if haskey(state, :cov_beliefs)
                    cov_states = state[:cov_beliefs]
                end
                println("Loaded state from $STATE_FILE")
            else
                @warn "State format changed, starting fresh"
            end
        catch e
            @warn "Could not load state, starting fresh" exception=e
        end
    end

    # ─── Main loop ───

    for q in 1:n_questions
        true_answer = rand(0:n_answers-1)
        true_cat = rand(0:n_categories-1)

        # Start with uniform answer belief
        answer_measure = CategoricalMeasure(Finite(answers))
        available = collect(1:n_tools)
        tool_responses = Dict{Int,Union{Int,Nothing}}()
        question_cost = 0.0

        done = false
        while !done && !isempty(available)
            cat_w = weights(cat_belief)

            # Build per-tool reliability measures (exact MixtureMeasure)
            rel_measures = [marginalize_betas(rel_states[t], cat_w) for t in available]

            # Build per-tool coverage probabilities
            cov_probs = Float64[expect(marginalize_betas(cov_states[t], cat_w), r -> r) for t in available]

            # Call DSL agent-step
            decision = agent_step_fn(
                answer_measure,
                [rel_measures[i] for i in eachindex(available)],
                [tool_cost(tools[t]) for t in available],
                cov_probs,
                submit_value, abstain_value, penalty_wrong
            )
            action_type = Int(decision[1])

            if action_type == 2  # query tool
                tool_local_idx = Int(decision[2]) + 1
                tool_idx = available[tool_local_idx]
                tool = tools[tool_idx]
                question_cost += tool_cost(tool)

                # Query tool via universal interface
                response = query_tool(tool, true_cat, true_answer, n_answers)

                if response !== nothing
                    tool_responses[tool_idx] = response

                    # Update coverage state (responded = 1.0)
                    (cov_states[tool_idx], cat_belief) = update_beta_state(cov_states[tool_idx], cat_belief, 1.0)

                    # Update answer belief via DSL
                    rel_m = rel_measures[tool_local_idx]
                    answer_kernel = env[Symbol("answer-kernel")]
                    k = answer_kernel(rel_m, Float64(n_answers))
                    answer_measure = update_on_response_fn(answer_measure, k, Float64(response))
                else
                    tool_responses[tool_idx] = nothing

                    # Update coverage state (not responded = 0.0)
                    (cov_states[tool_idx], cat_belief) = update_beta_state(cov_states[tool_idx], cat_belief, 0.0)
                end

                filter!(t -> t != tool_idx, available)

            elseif action_type == 0  # submit
                submitted = Int(decision[2])
                was_correct = submitted == true_answer
                score = was_correct ? reward_correct : penalty_wrong
                total_score += score
                total_cost += question_cost

                # Update reliability beliefs with ground truth
                for (t, resp) in tool_responses
                    resp === nothing && continue
                    correct = Float64(resp == true_answer)
                    (rel_states[t], cat_belief) = update_beta_state(rel_states[t], cat_belief, correct)
                end

                println("Q$q: submit=$submitted correct=$was_correct score=$score " *
                        "tools=$(length(tool_responses)) cost=$question_cost")
                flush(stdout)
                done = true

            else  # abstain
                total_cost += question_cost
                println("Q$q: abstain tools=$(length(tool_responses)) cost=$question_cost")
                flush(stdout)
                done = true
            end
        end

        # All tools exhausted — submit or abstain based on current beliefs
        if !done
            pref = (h, a) -> a == h ? reward_correct : penalty_wrong
            best_eu = value(answer_measure, answer_space, pref)
            if best_eu > reward_abstain
                best_a = optimise(answer_measure, answer_space, pref)
                submitted = Int(best_a)
                was_correct = submitted == true_answer
                score = was_correct ? reward_correct : penalty_wrong
                total_score += score
                total_cost += question_cost
                println("Q$q: submit=$submitted (forced) correct=$was_correct")
                flush(stdout)
            else
                total_cost += question_cost
                println("Q$q: abstain (forced)")
                flush(stdout)
            end
        end
    end

    # ─── Save state ───
    save_state(STATE_FILE;
               rel_beliefs=rel_states, cov_beliefs=cov_states,
               cat_belief=cat_belief,
               total_score=total_score, total_cost=total_cost)

    println("\nTotal score: $total_score")
    println("Total cost:  $total_cost")
    println("Net:         $(total_score - total_cost)")
    println("State saved to $STATE_FILE")

    (total_score, total_cost)
end

run_agent()

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
using Credence: CategoricalMeasure, BetaMeasure, MixtureMeasure, Finite, Interval, Kernel, Measure
using Credence: density, log_density_at, save_state, load_state

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

# ─── Host-level helpers ───

"""Thompson sampling: draw (host) + optimise (ontology)."""
function thompson_action(posterior::Measure, actions::Finite, pref)
    h = draw(posterior)
    point = CategoricalMeasure(Finite([h]))
    optimise(point, actions, pref)
end

"""Construct exact MixtureMeasure from per-category Betas weighted by category posterior."""
function effective_reliability(rel_beliefs_t::Vector{BetaMeasure},
                                cat_w::Vector{Float64})
    MixtureMeasure(Interval(0.0, 1.0),
        Measure[rel_beliefs_t[c] for c in eachindex(rel_beliefs_t)],
        [log(max(cat_w[c], 1e-300)) for c in eachindex(cat_w)])
end

"""Coverage kernel: dispatches on AbstractTool, uses trait methods."""
function coverage_kernel(tool::AbstractTool, cat_space::Finite)
    cov = tool_coverage(tool)
    binary = Finite([0.0, 1.0])
    Kernel(cat_space, binary,
        c -> (o -> let p = cov[Int(c) + 1];
              o == 1.0 ? log(p) : log(1.0 - p) end),
        (c, o) -> let p = cov[Int(c) + 1];
                  o == 1.0 ? log(p) : log(1.0 - p) end)
end

"""
Update per-category reliability beliefs via MixtureMeasure condition.

Conditions MixtureMeasure(per_cat_betas, cat_weights) on Bernoulli(was_correct).
Extracts updated per-category Betas into rel_beliefs_t (mutates in place).
Returns new cat_belief from posterior weights — mandatory, encodes category evidence.
"""
function update_reliability!(rel_beliefs_t::Vector{BetaMeasure},
                              cat_belief::CategoricalMeasure,
                              was_correct::Float64)
    cat_w = weights(cat_belief)

    # Construct mixture from current beliefs
    rel_mix = effective_reliability(rel_beliefs_t, cat_w)

    # Bernoulli kernel on reliability
    bern = Kernel(Interval(0.0, 1.0), Finite([0.0, 1.0]),
        r -> (o -> o == 1.0 ? log(r) : log(1.0 - r)),
        (r, o) -> o == 1.0 ? log(r) : log(1.0 - r))

    # Condition preserves MixtureMeasure structure
    posterior = condition(rel_mix, bern, was_correct)

    # Extract updated per-category Betas (mutates rel_beliefs_t)
    for c in eachindex(rel_beliefs_t)
        rel_beliefs_t[c] = posterior.components[c]
    end

    # Return updated category belief from posterior weights.
    # The posterior component weights encode evidence about category
    # membership from this reliability observation — discarding them
    # loses information the agent has already paid for.
    CategoricalMeasure(cat_belief.space, copy(posterior.log_weights))
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
    rel_beliefs = [[BetaMeasure() for _ in 1:n_categories] for _ in 1:n_tools]
    cat_belief = CategoricalMeasure(Finite(Float64.(collect(0:n_categories-1))))
    total_score = 0.0
    total_cost = 0.0

    if isfile(STATE_FILE)
        state = load_state(STATE_FILE)
        rel_beliefs = state[:rel_beliefs]
        cat_belief = state[:cat_belief]
        total_score = state[:total_score]
        total_cost = state[:total_cost]
        println("Loaded state from $STATE_FILE")
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
            rel_measures = [effective_reliability(rel_beliefs[t], cat_w) for t in available]

            # Call DSL agent-step
            decision = agent_step_fn(
                answer_measure,
                [rel_measures[i] for i in eachindex(available)],
                [tool_cost(tools[t]) for t in available],
                submit_value, abstain_value
            )
            action_type = Int(decision[1])

            if action_type == 2  # query tool
                tool_local_idx = Int(decision[2]) + 1
                tool_idx = available[tool_local_idx]
                tool = tools[tool_idx]
                question_cost += tool_cost(tool)

                # Coverage kernel
                cov_kernel = coverage_kernel(tool, cat_belief.space)

                # Query tool via universal interface
                response = query_tool(tool, true_cat, true_answer, n_answers)

                if response !== nothing
                    tool_responses[tool_idx] = response

                    # Update answer belief via DSL
                    rel_m = rel_measures[tool_local_idx]
                    answer_kernel = env[Symbol("answer-kernel")]
                    k = answer_kernel(rel_m, Float64(n_answers))
                    answer_measure = update_on_response_fn(answer_measure, k, Float64(response))

                    cat_belief = condition(cat_belief, cov_kernel, 1.0)
                else
                    tool_responses[tool_idx] = nothing
                    cat_belief = condition(cat_belief, cov_kernel, 0.0)
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
                    cat_belief = update_reliability!(rel_beliefs[t], cat_belief, correct)
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
               rel_beliefs=rel_beliefs, cat_belief=cat_belief,
               total_score=total_score, total_cost=total_cost)

    println("\nTotal score: $total_score")
    println("Total cost:  $total_cost")
    println("Net:         $(total_score - total_cost)")
    println("State saved to $STATE_FILE")

    (total_score, total_cost)
end

run_agent()

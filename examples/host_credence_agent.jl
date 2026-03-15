#!/usr/bin/env julia
"""
    host_credence_agent.jl — Julia host driver for the credence agent

Loads the DSL agent specification, drives the question loop,
maintains per-tool per-category reliability beliefs across questions,
and simulates tool responses.

Uses ontology types: CategoricalMeasure, BetaMeasure, Kernel.
All decision-theoretic computation goes through the ontology module.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: expect, condition, draw, optimise, value, weights, mean
using Credence: CategoricalMeasure, BetaMeasure, Finite, Interval, Kernel, Measure
using Credence: density, log_density_at, save_state, load_state

# ─── State file ───

const STATE_FILE = joinpath(@__DIR__, ".credence_state.bin")

# ─── Host-level helpers ───

"""Thompson sampling: draw (host) + optimise (ontology)."""
function thompson_action(posterior::Measure, actions::Finite, pref)
    h = draw(posterior)
    point = CategoricalMeasure(Finite([h]))
    optimise(point, actions, pref)
end

"""Mix per-category reliability beliefs weighted by category posterior.
Returns a BetaMeasure that approximates the mixture via moment-matching."""
function mix_reliability(cat_weights::Vector{Float64}, per_cat_rels::Vector{BetaMeasure})
    # Moment-match: compute mixture mean and variance, fit Beta
    mixed_mean = sum(cat_weights[c] * mean(per_cat_rels[c]) for c in eachindex(cat_weights))
    mixed_var = sum(cat_weights[c] * (variance(per_cat_rels[c]) + mean(per_cat_rels[c])^2)
                    for c in eachindex(cat_weights)) - mixed_mean^2
    # Fit Beta(α, β) from mean and variance
    if mixed_var < 1e-12 || mixed_mean < 1e-10 || mixed_mean > 1.0 - 1e-10
        return BetaMeasure(mixed_mean * 100, (1.0 - mixed_mean) * 100)
    end
    ν = mixed_mean * (1 - mixed_mean) / mixed_var - 1
    ν = max(ν, 2.0)  # ensure α, β > 0
    α = mixed_mean * ν
    β = (1 - mixed_mean) * ν
    BetaMeasure(α, β)
end

"""Build joint belief from answer weights × reliability weights.
Returns a CategoricalMeasure over (answer, reliability) pairs."""
function make_joint_belief(answer_w::Vector{Float64}, rel_m::BetaMeasure,
                           answers::Vector{Float64}, rel_grid::Vector{Float64})
    # Discretise the BetaMeasure onto the grid
    rel_logw = [log_density_at(rel_m, r) for r in rel_grid]
    max_lw = maximum(rel_logw)
    rel_w = exp.(rel_logw .- max_lw)
    rel_w ./= sum(rel_w)

    hyps = Vector{Float64}[]
    logw = Float64[]
    for (ai, aw) in enumerate(answer_w)
        for (ri, rw) in enumerate(rel_w)
            push!(hyps, [answers[ai], rel_grid[ri]])
            push!(logw, log(max(aw, 1e-300)) + log(max(rw, 1e-300)))
        end
    end
    CategoricalMeasure{Vector{Float64}}(Finite{Vector{Float64}}(hyps), logw)
end

"""Extract answer marginal from a joint (answer, reliability) measure."""
function answer_marginal(joint::CategoricalMeasure, n_answers::Int)
    w = weights(joint)
    marginal = zeros(n_answers)
    for (i, h) in enumerate(joint.space.values)
        marginal[Int(h[1]) + 1] += w[i]
    end
    marginal
end

"""
Update per-category reliability beliefs using joint (category, reliability) belief.
Construct joint over (category, reliability), condition with a Kernel, decompose.
"""
function update_reliability_joint!(rel_beliefs_t::Vector{BetaMeasure},
                                   cat_w::Vector{Float64},
                                   rel_grid::Vector{Float64},
                                   was_correct::Float64)
    n_cat = length(rel_beliefs_t)

    # Build joint measure over (category, reliability)
    hyps = Vector{Float64}[]
    logw = Float64[]
    for c in 1:n_cat
        # Discretise BetaMeasure onto grid
        bm = rel_beliefs_t[c]
        for (ri, r) in enumerate(rel_grid)
            push!(hyps, [Float64(c - 1), r])
            lp_cat = log(max(cat_w[c], 1e-300))
            lp_rel = log_density_at(bm, r)
            push!(logw, lp_cat + lp_rel)
        end
    end
    joint_space = Finite{Vector{Float64}}(hyps)
    joint = CategoricalMeasure{Vector{Float64}}(joint_space, logw)

    # Bernoulli kernel on the reliability component
    binary_space = Finite([0.0, 1.0])
    bern_kernel = Kernel(
        joint_space, binary_space,
        h -> o -> o == 1.0 ? log(h[2]) : log(1.0 - h[2]),
        (h, o) -> o == 1.0 ? log(h[2]) : log(1.0 - h[2])
    )
    updated = condition(joint, bern_kernel, was_correct)

    # Decompose: extract per-category reliability posteriors
    # Fit Beta from posterior moments per category
    uw = weights(updated)
    n_rel = length(rel_grid)
    for c in 1:n_cat
        offset = (c - 1) * n_rel
        cat_mass = sum(uw[offset + ri] for ri in 1:n_rel)
        if cat_mass < 1e-15; continue; end
        # Compute mean and variance of reliability for this category
        m = sum(uw[offset + ri] * rel_grid[ri] for ri in 1:n_rel) / cat_mass
        v = sum(uw[offset + ri] * rel_grid[ri]^2 for ri in 1:n_rel) / cat_mass - m^2
        # Fit Beta from moments
        m = clamp(m, 1e-6, 1 - 1e-6)
        v = max(v, 1e-12)
        ν = m * (1 - m) / v - 1
        ν = max(ν, 2.0)
        rel_beliefs_t[c] = BetaMeasure(m * ν, (1 - m) * ν)
    end
end

"""Run the credence agent for n_questions, returning (total_score, total_cost)."""
function run_agent(; n_questions=50)
    # ─── Load DSL agent ───
    env = load_dsl(read(joinpath(@__DIR__, "credence_agent.bdsl"), String))
    agent_step_fn = env[Symbol("agent-step")]
    update_on_response_fn = env[Symbol("update-on-response")]
    update_category_fn = env[Symbol("update-category")]

    # ─── Configuration ───
    n_answers = 4
    n_tools = 3
    n_categories = 5
    rel_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    answers = Float64[0, 1, 2, 3]
    tool_costs = [2.0, 0.5, 1.0]
    tool_true_rel = [[0.85, 0.70, 0.80, 0.60, 0.70],
                      [0.60, 0.80, 0.50, 0.90, 0.60],
                      [0.75, 0.75, 0.75, 0.75, 0.75]]
    tool_coverage_table = [[0.9, 0.7, 0.8, 0.6, 0.7],
                            [0.6, 0.8, 0.5, 0.9, 0.6],
                            [0.8, 0.8, 0.8, 0.8, 0.8]]
    reward_correct, penalty_wrong, reward_abstain = 10.0, -5.0, 0.0

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

            # Build per-tool reliability measures (mix across categories)
            rel_measures = [mix_reliability(cat_w, rel_beliefs[t]) for t in available]

            # Call DSL agent-step
            decision = agent_step_fn(
                answer_measure,
                [rel_measures[i] for i in eachindex(available)],
                [tool_costs[t] for t in available],
                submit_value, abstain_value
            )
            action_type = Int(decision[1])

            if action_type == 2  # query tool
                tool_local_idx = Int(decision[2]) + 1
                tool_idx = available[tool_local_idx]
                question_cost += tool_costs[tool_idx]

                # Coverage kernel (shared by both branches)
                cat_kernel_space = cat_belief.space
                binary = Finite([0.0, 1.0])
                cov_kernel = Kernel(
                    cat_kernel_space, binary,
                    c -> (let p = tool_coverage_table[tool_idx][Int(c) + 1];
                          o -> o == 1.0 ? log(p) : log(1.0 - p) end),
                    (c, o) -> (let p = tool_coverage_table[tool_idx][Int(c) + 1];
                               o == 1.0 ? log(p) : log(1.0 - p) end)
                )

                # Simulate tool response
                if rand() < tool_coverage_table[tool_idx][true_cat + 1]
                    response = if rand() < tool_true_rel[tool_idx][true_cat + 1]
                        true_answer
                    else
                        rand(setdiff(0:n_answers-1, [true_answer]))
                    end
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
                cat_w = weights(cat_belief)
                for (t, resp) in tool_responses
                    resp === nothing && continue
                    correct = Float64(resp == true_answer)
                    update_reliability_joint!(rel_beliefs[t], cat_w, rel_grid, correct)
                end

                println("Q$q: submit=$submitted correct=$was_correct score=$score " *
                        "tools=$(length(tool_responses)) cost=$question_cost")
                done = true

            else  # abstain
                total_cost += question_cost
                println("Q$q: abstain tools=$(length(tool_responses)) cost=$question_cost")
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
            else
                total_cost += question_cost
                println("Q$q: abstain (forced)")
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

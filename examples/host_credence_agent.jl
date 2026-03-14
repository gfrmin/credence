#!/usr/bin/env julia
"""
    host_credence_agent.jl — Julia host driver for the credence agent

Loads the DSL agent specification, drives the question loop,
maintains per-tool per-category reliability beliefs across questions,
and simulates tool responses.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using BayesianDSL

# ─── State file ───

const STATE_FILE = joinpath(@__DIR__, ".credence_state.bin")

# ─── Julia helpers ───

"""Mix per-category reliability beliefs weighted by category posterior."""
function mix_reliability(cat_weights::Vector{Float64}, per_cat_rels::Vector{<:Belief})
    n = length(per_cat_rels[1])
    mixed = zeros(n)
    for (c, cw) in enumerate(cat_weights)
        mixed .+= cw .* weights(per_cat_rels[c])
    end
    mixed
end

"""Build joint belief from answer weights × reliability weights."""
function make_joint_belief(answer_w::Vector{Float64}, rel_w::Vector{Float64},
                           answers::Vector{Float64}, rel_grid::Vector{Float64})
    hyps = Vector{Float64}[]
    logw = Float64[]
    for (ai, aw) in enumerate(answer_w)
        for (ri, rw) in enumerate(rel_w)
            push!(hyps, [answers[ai], rel_grid[ri]])
            push!(logw, log(max(aw, 1e-300)) + log(max(rw, 1e-300)))
        end
    end
    Belief{Vector{Float64}}(hyps, logw)
end

"""Extract answer marginal from a joint (answer, reliability) belief."""
function answer_marginal(joint::Belief, n_answers::Int)
    w = weights(joint)
    marginal = zeros(n_answers)
    for (i, h) in enumerate(joint.hyps)
        marginal[Int(h[1]) + 1] += w[i]
    end
    marginal
end

"""
Infer category distribution from question text using keyword patterns.
Returns a weight vector over categories (unnormalised).
Falls back to uniform if no patterns match.
"""
function infer_category(text::String, patterns::Dict{Int,Vector{Regex}})
    n_cat = maximum(keys(patterns); init=0)
    w = ones(n_cat)
    for (cat, regexes) in patterns
        for r in regexes
            if occursin(r, text)
                w[cat] *= 3.0  # boost matching categories
            end
        end
    end
    w ./ sum(w)
end

"""
Update per-category reliability beliefs using joint (category, reliability) belief.
Exact Bayesian approach: construct joint over (category, reliability), update with
the was_correct observation, decompose back into per-category posteriors.
"""
function update_reliability_joint!(rel_beliefs_t::Vector{<:Belief},
                                   cat_w::Vector{Float64},
                                   rel_grid::Vector{Float64},
                                   was_correct::Float64)
    n_cat = length(rel_beliefs_t)
    hyps = Vector{Float64}[]
    logw = Float64[]
    for c in 1:n_cat
        rw = weights(rel_beliefs_t[c])
        for (ri, r) in enumerate(rel_grid)
            push!(hyps, [Float64(c - 1), r])
            push!(logw, log(max(cat_w[c], 1e-300)) + log(max(rw[ri], 1e-300)))
        end
    end
    joint = Belief{Vector{Float64}}(hyps, logw)

    # Bernoulli likelihood on the reliability component
    lik = (h, obs) -> obs == 1.0 ? log(h[2]) : log(1.0 - h[2])
    updated = update(joint, was_correct, lik)

    # Decompose: extract per-category reliability posteriors
    uw = weights(updated)
    n_rel = length(rel_grid)
    for c in 1:n_cat
        cat_logw = Float64[]
        offset = (c - 1) * n_rel
        for ri in 1:n_rel
            push!(cat_logw, log(max(uw[offset + ri], 1e-300)))
        end
        rel_beliefs_t[c] = Belief{Float64}(copy(rel_grid), cat_logw)
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
    n_tools = 2
    n_categories = 5
    rel_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    answers = Float64[0, 1, 2, 3]
    all_actions = Float64[-1, 0, 1, 2, 3]  # -1 = abstain
    tool_costs = [2.0, 0.5]
    tool_true_rel = [[0.85, 0.70, 0.80, 0.60, 0.70],
                      [0.60, 0.80, 0.50, 0.90, 0.60]]
    tool_coverage_table = [[0.9, 0.7, 0.8, 0.6, 0.7],
                            [0.6, 0.8, 0.5, 0.9, 0.6]]
    reward_correct, penalty_wrong, reward_abstain = 10.0, -5.0, 0.0
    util = (h, a) -> a == -1.0 ? reward_abstain : (h[1] == a ? reward_correct : penalty_wrong)

    # ─── Persistent state (load if available) ───
    rel_beliefs = [[Belief(copy(rel_grid)) for _ in 1:n_categories] for _ in 1:n_tools]
    cat_belief = Belief(Float64.(collect(0:n_categories-1)))
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

        answer_w = fill(1.0 / n_answers, n_answers)
        available = collect(1:n_tools)
        tool_responses = Dict{Int,Union{Int,Nothing}}()
        question_cost = 0.0

        done = false
        while !done && !isempty(available)
            cat_w = weights(cat_belief)

            # Build tool-infos for available tools
            tool_infos = Any[]
            for t in available
                eff_rel_w = mix_reliability(cat_w, rel_beliefs[t])
                joint = make_joint_belief(answer_w, eff_rel_w, answers, rel_grid)
                cov = sum(cat_w .* tool_coverage_table[t])
                push!(tool_infos, Any[joint, tool_costs[t], cov, Float64(t - 1)])
            end

            decision = agent_step_fn(tool_infos, answers, all_actions, Float64(n_answers), util)
            action_type = Int(decision[1])

            if action_type == 2  # query
                tool_idx = Int(decision[2]) + 1
                question_cost += tool_costs[tool_idx]

                # Simulate tool response
                if rand() < tool_coverage_table[tool_idx][true_cat + 1]
                    response = if rand() < tool_true_rel[tool_idx][true_cat + 1]
                        true_answer
                    else
                        rand(setdiff(0:n_answers-1, [true_answer]))
                    end
                    tool_responses[tool_idx] = response

                    # Update joint belief → extract new answer marginal
                    ti_idx = findfirst(ti -> Int(ti[4]) + 1 == tool_idx, tool_infos)
                    updated_joint = update_on_response_fn(tool_infos[ti_idx][1],
                                                           Float64(response), Float64(n_answers))
                    answer_w = answer_marginal(updated_joint, n_answers)

                    # Update category belief (tool responded)
                    cov_fn = c -> tool_coverage_table[tool_idx][Int(c) + 1]
                    cat_belief = update_category_fn(cat_belief, 1.0, cov_fn)
                else
                    tool_responses[tool_idx] = nothing
                    cov_fn = c -> tool_coverage_table[tool_idx][Int(c) + 1]
                    cat_belief = update_category_fn(cat_belief, 0.0, cov_fn)
                end

                filter!(t -> t != tool_idx, available)

            elseif action_type == 0  # submit
                submitted = Int(decision[2])
                was_correct = submitted == true_answer
                score = was_correct ? reward_correct : penalty_wrong
                total_score += score
                total_cost += question_cost

                # Update reliability beliefs using joint (category, reliability) approach
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
            joint = make_joint_belief(answer_w, fill(1.0/length(rel_grid), length(rel_grid)),
                                      answers, rel_grid)
            best = decide(joint, answers, (h, a) -> h[1] == a ? reward_correct : penalty_wrong)
            if best.eu > reward_abstain
                submitted = Int(best.action)
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

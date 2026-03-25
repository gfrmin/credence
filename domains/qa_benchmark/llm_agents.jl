"""
    llm_agents.jl — LLM-based tool selection agents via Ollama.

Full 2³ factorial over three prompting techniques:
  R = Reasoning trace (Thought/Action/Observation, per Yao et al. 2023)
  S = Strategy prompt (cost-awareness, reliability tracking)
  H = Cross-question history (last 10 results)

All 8 variants share one run_llm_seed function with boolean kwargs.
"""

using HTTP, JSON3, Random

const TOOL_NAMES = ("quick_search", "knowledge_base", "calculator", "llm_direct")

const BASE_SYSTEM_PROMPT = """
You are a question-answering agent with access to four tools:
- quick_search: Fast web search. Good for factual questions. Cost: 1 point.
- knowledge_base: Curated database. Very reliable when it has results, but returns "no result" for questions outside its coverage. Cost: 2 points.
- calculator: Perfect for numerical computation. Returns "not applicable" for non-numerical questions. Cost: 1 point.
- llm_direct: Ask your own knowledge directly. Moderately reliable. Cost: 2 points.

SCORING:
- Correct answer: +10 points
- Wrong answer: -5 points
- "I don't know": 0 points
- Each tool use costs points as listed above

Your goal is to MAXIMISE total score across all questions.
Be selective about which tools you use — every query costs points.
If you're not confident, it's better to say "I don't know" (0 points) than guess wrong (-5 points).

For each question, select from the 4 provided candidate answers, or abstain.
"""

const STRATEGY_BLOCK = """

STRATEGY GUIDANCE:
- Track which tools have been reliable for which types of questions.
- The calculator is perfect for maths but useless otherwise.
- Web search can return popular but incorrect answers for common misconceptions.
- If two tools disagree, consider which has been more reliable for this question type.
- The knowledge base returning "no result" means the question may be outside common factual territory.
- Be especially careful with questions that SEEM simple but might be tricky.
- Only say "I don't know" if you genuinely can't determine the answer with reasonable confidence. The threshold: if you think there's less than a 1 in 3 chance you're right, abstain.

TOOL SELECTION:
- Don't query tools that are unlikely to help with this question type.
- Don't cross-verify unless the first result seems uncertain.
- Use the cheapest applicable tool first.
"""

const REACT_BLOCK = """

FORMAT:
Before each action, write a Thought: line explaining your reasoning.
Then write an Action: line with your chosen action.
Example:
  Thought: This is a factual geography question. quick_search is cheap and good for factual questions.
  Action: QUERY quick_search
"""

const ACTION_FORMAT = """

Respond with EXACTLY one of:
- QUERY <tool_name>
- SUBMIT <index> (the candidate number 0-3)
- ABSTAIN
"""

# --- Ollama API ---

"""
    call_llm(system_prompt, user_message; model) → String

Routes to Ollama (local) or Anthropic API (Claude) based on model name.
"""
function call_llm(system_prompt::String, user_message::String;
                  model::String="llama3.1", timeout::Int=60)::String
    if startswith(model, "claude")
        return _call_claude(system_prompt, user_message; model, timeout)
    else
        return _call_ollama(system_prompt, user_message; model, timeout)
    end
end

function _call_ollama(system_prompt::String, user_message::String;
                      model::String="llama3.1", timeout::Int=60)::String
    messages = [
        Dict("role" => "system", "content" => system_prompt),
        Dict("role" => "user", "content" => user_message),
    ]
    body = Dict("model" => model, "messages" => messages,
                "stream" => false,
                "options" => Dict("temperature" => 0.0, "num_predict" => 150))

    resp = HTTP.post("http://localhost:11434/api/chat";
                     body=JSON3.write(body),
                     headers=["Content-Type" => "application/json"],
                     readtimeout=timeout, retry=false)
    data = JSON3.read(resp.body)
    return String(data.message.content)
end

function _call_claude(system_prompt::String, user_message::String;
                      model::String="claude-sonnet-4-20250514", timeout::Int=60)::String
    api_key = get(ENV, "ANTHROPIC_API_KEY", "")
    isempty(api_key) && error("ANTHROPIC_API_KEY not set")

    body = Dict(
        "model" => model,
        "max_tokens" => 150,
        "system" => system_prompt,
        "messages" => [Dict("role" => "user", "content" => user_message)],
        "temperature" => 0.0,
    )

    resp = HTTP.post("https://api.anthropic.com/v1/messages";
                     body=JSON3.write(body),
                     headers=[
                         "Content-Type" => "application/json",
                         "x-api-key" => api_key,
                         "anthropic-version" => "2023-06-01",
                     ],
                     readtimeout=timeout, retry=false)
    data = JSON3.read(resp.body)
    return String(data.content[1].text)
end

# --- Response parsing ---

function parse_action(text::String, available_tool_indices::Vector{Int}, react::Bool)
    # If react mode, extract last Action: line
    lines = split(strip(text), '\n')
    action_text = if react
        action_lines = filter(l -> startswith(strip(l), "Action:"), lines)
        isempty(action_lines) ? strip(text) : strip(last(action_lines)[8:end])
    else
        strip(text)
    end

    upper = uppercase(action_text)
    lower = lowercase(action_text)

    # Abstain
    if contains(upper, "ABSTAIN") || contains(upper, "I DON'T KNOW") || contains(upper, "DON'T KNOW")
        return (:abstain, 0)
    end

    # Submit — recognise SUBMIT, ANSWER, SAY, or "candidate N"
    for pattern in [r"SUBMIT\s*(\d)", r"ANSWER\s*(\d)", r"SAY\s+CANDIDATE\s*(\d)",
                    r"CANDIDATE\s*(\d)", r"CHOOSE\s*(\d)"]
        m = match(pattern, upper)
        if m !== nothing
            idx = parse(Int, m.captures[1])
            if 0 <= idx <= 3
                return (:submit, idx)
            end
        end
    end

    # Query — check for tool names anywhere in text (with or without QUERY prefix)
    for (i, name) in enumerate(TOOL_NAMES)
        if contains(lower, name) && i in available_tool_indices
            return (:query, i)
        end
    end

    # Query fallback: if LLM says QUERY but tool name doesn't match available
    if contains(upper, "QUERY") && !isempty(available_tool_indices)
        return (:query, first(available_tool_indices))
    end

    # Last resort: look for any digit 0-3 that might be an answer
    m = match(r"\b([0-3])\b", action_text)
    if m !== nothing
        return (:submit, parse(Int, m.captures[1]))
    end

    # True fallback — abstain rather than blindly submit 0
    return (:abstain, 0)
end

# --- Main runner ---

function run_llm_seed(tools::Vector{SimulatedTool}, seed::Int;
                      react::Bool=false, strategy::Bool=false, history::Bool=false,
                      model::String="llama3.1", logfile::Union{IO,Nothing}=nothing)
    rng = MersenneTwister(seed)
    questions = get_questions(; seed)
    records = QuestionResult[]
    total_reward = 0.0
    total_tool_cost = 0.0
    past_results = String[]  # for history block
    t_start = time()
    variant = llm_variant_name(; react, strategy, history)

    # Build system prompt once per seed (R and S don't change per question)
    sys_prompt = BASE_SYSTEM_PROMPT
    if strategy; sys_prompt *= STRATEGY_BLOCK; end
    if react
        sys_prompt *= REACT_BLOCK
    else
        sys_prompt *= ACTION_FORMAT
    end

    for q in questions
        available = collect(1:length(tools))
        tool_responses = Dict{Int,Union{Int,Nothing}}()
        tools_queried = Int[]
        q_cost = 0.0
        react_trace = String[]  # within-question trace for react mode

        while true
            # Build user message
            parts = String[]

            # History block
            if history && !isempty(past_results)
                push!(parts, "PAST RESULTS (most recent):")
                for entry in past_results[max(1, end-9):end]
                    push!(parts, "  $entry")
                end
                push!(parts, "")
            end

            push!(parts, "Question: $(q.text)")
            candidate_lines = join(["  $i: $(q.candidates[i+1])" for i in 0:3], "\n")
            push!(parts, "Candidates:\n$candidate_lines")

            if !isempty(tool_responses)
                push!(parts, "\nTool results so far:")
                for (t_idx, resp) in tool_responses
                    tname = TOOL_NAMES[t_idx]
                    if resp === nothing
                        push!(parts, "  $tname: no result / not applicable")
                    else
                        push!(parts, "  $tname: candidate $resp ($(q.candidates[resp+1]))")
                    end
                end
            end

            # React trace (within-question)
            if react && !isempty(react_trace)
                push!(parts, "\nYour reasoning so far:")
                for line in react_trace
                    push!(parts, line)
                end
            end

            avail_names = [TOOL_NAMES[i] for i in available if i ∉ keys(tool_responses)]
            if !isempty(avail_names)
                push!(parts, "\nAvailable tools (not yet queried): $(join(avail_names, ", "))")
            else
                push!(parts, "\nAll tools have been queried. You must now decide.")
            end

            user_msg = join(parts, "\n")

            # Call LLM
            llm_response = try
                call_llm(sys_prompt, user_msg; model)
            catch e
                @warn "Ollama call failed" exception=e
                "SUBMIT 0"
            end

            # Save react trace
            if react
                for line in split(strip(llm_response), '\n')
                    sl = strip(line)
                    if startswith(sl, "Thought:") || startswith(sl, "Action:")
                        push!(react_trace, sl)
                    end
                end
            end

            available_for_query = [i for i in available if i ∉ keys(tool_responses)]
            action_type, action_arg = parse_action(llm_response, available_for_query, react)

            # Debug logging
            if logfile !== nothing
                println(logfile, "=== $variant seed=$seed q=$(q.id) step=$(length(tools_queried)+1) ===")
                println(logfile, "LLM response ($(length(llm_response)) chars):")
                println(logfile, llm_response)
                println(logfile, "Parsed: $action_type $action_arg (available: $available_for_query)")
                println(logfile, "")
                flush(logfile)
            end

            if action_type == :query
                tool_idx = action_arg
                push!(tools_queried, tool_idx)
                q_cost += tools[tool_idx].cost

                response = query_tool(tools[tool_idx], q, rng)
                tool_responses[tool_idx] = response

                # Add observation to react trace
                if react
                    tname = TOOL_NAMES[tool_idx]
                    obs = response === nothing ? "no result / not applicable" :
                          "candidate $response ($(q.candidates[response+1]))"
                    push!(react_trace, "Observation: $tname returned $obs")
                end

                filter!(!=(tool_idx), available_for_query)

                # If all tools exhausted, force a decision next round
                if isempty([i for i in available if i ∉ keys(tool_responses)])
                    # Will loop back and LLM will see "All tools queried"
                end

            elseif action_type == :submit
                submitted = action_arg
                was_correct = submitted == q.correct_index
                reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
                total_reward += reward
                total_tool_cost += q_cost

                push!(records, QuestionResult(q.id, q.category, tools_queried,
                    submitted, was_correct, reward, q_cost))

                # Record for history
                tools_used = join([TOOL_NAMES[t] for t in tools_queried], ", ")
                result_str = was_correct ? "correct" : "wrong"
                push!(past_results, "Q: $(q.id) | Tools: $(isempty(tools_used) ? "none" : tools_used) | Result: $result_str")
                break

            else  # abstain
                total_reward += REWARD_ABSTAIN
                total_tool_cost += q_cost

                push!(records, QuestionResult(q.id, q.category, tools_queried,
                    nothing, nothing, REWARD_ABSTAIN, q_cost))

                tools_used = join([TOOL_NAMES[t] for t in tools_queried], ", ")
                push!(past_results, "Q: $(q.id) | Tools: $(isempty(tools_used) ? "none" : tools_used) | Result: abstained")
                break
            end
        end
    end

    wall_time = time() - t_start
    SeedResult(seed, records, total_reward - total_tool_cost,
               total_reward, total_tool_cost, wall_time)
end

# Variant name from flags
function llm_variant_name(; react=false, strategy=false, history=false)
    parts = String[]
    if react; push!(parts, "R"); end
    if strategy; push!(parts, "S"); end
    if history; push!(parts, "H"); end
    isempty(parts) ? "llm_bare" : "llm_" * join(parts)
end

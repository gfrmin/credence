"""
    llm_agent.jl — LLM agent driver with native tool-calling.

Supports Ollama (local) and Anthropic (frontier) backends. The LLM sees
six tools: four simulated information tools plus submit_answer and abstain.
Tool results return candidate text, not raw indices.
"""

using HTTP, JSON3, Random

# ─── Backends ───

abstract type LLMBackend end

struct OllamaBackend <: LLMBackend
    model::String
    host::String
end

struct AnthropicBackend <: LLMBackend
    model::String
    api_key::String
end

# ─── Tool definitions ───

const SIMULATED_TOOL_MAP = Dict(
    "web_search"     => 1,
    "knowledge_base" => 2,
    "calculator"     => 3,
    "llm_direct"     => 4,
)

function ollama_tool_defs()
    [
        Dict("type" => "function", "function" => Dict(
            "name" => "web_search",
            "description" => "Quick web search for general knowledge. Costs 1 point.",
            "parameters" => Dict("type" => "object", "properties" => Dict(
                "query" => Dict("type" => "string", "description" => "Search query")),
                "required" => ["query"]))),
        Dict("type" => "function", "function" => Dict(
            "name" => "knowledge_base",
            "description" => "Deep domain knowledge lookup. Costs 2 points.",
            "parameters" => Dict("type" => "object", "properties" => Dict(
                "query" => Dict("type" => "string", "description" => "Lookup query")),
                "required" => ["query"]))),
        Dict("type" => "function", "function" => Dict(
            "name" => "calculator",
            "description" => "Exact numerical computation. Costs 1 point.",
            "parameters" => Dict("type" => "object", "properties" => Dict(
                "expression" => Dict("type" => "string", "description" => "Math expression")),
                "required" => ["expression"]))),
        Dict("type" => "function", "function" => Dict(
            "name" => "llm_direct",
            "description" => "General reasoning and analysis. Costs 2 points.",
            "parameters" => Dict("type" => "object", "properties" => Dict(
                "question" => Dict("type" => "string", "description" => "Question to reason about")),
                "required" => ["question"]))),
        Dict("type" => "function", "function" => Dict(
            "name" => "submit_answer",
            "description" => "Submit your final answer. Correct: +10, Wrong: -5.",
            "parameters" => Dict("type" => "object", "properties" => Dict(
                "answer_index" => Dict("type" => "integer", "description" => "Candidate index (0-3)")),
                "required" => ["answer_index"]))),
        Dict("type" => "function", "function" => Dict(
            "name" => "abstain",
            "description" => "Choose not to answer. Score: 0.",
            "parameters" => Dict("type" => "object", "properties" => Dict()))),
    ]
end

function anthropic_tool_defs()
    [
        Dict("name" => "web_search", "description" => "Quick web search for general knowledge. Costs 1 point.",
             "input_schema" => Dict("type" => "object", "properties" => Dict(
                 "query" => Dict("type" => "string", "description" => "Search query")),
                 "required" => ["query"])),
        Dict("name" => "knowledge_base", "description" => "Deep domain knowledge lookup. Costs 2 points.",
             "input_schema" => Dict("type" => "object", "properties" => Dict(
                 "query" => Dict("type" => "string", "description" => "Lookup query")),
                 "required" => ["query"])),
        Dict("name" => "calculator", "description" => "Exact numerical computation. Costs 1 point.",
             "input_schema" => Dict("type" => "object", "properties" => Dict(
                 "expression" => Dict("type" => "string", "description" => "Math expression")),
                 "required" => ["expression"])),
        Dict("name" => "llm_direct", "description" => "General reasoning and analysis. Costs 2 points.",
             "input_schema" => Dict("type" => "object", "properties" => Dict(
                 "question" => Dict("type" => "string", "description" => "Question to reason about")),
                 "required" => ["question"])),
        Dict("name" => "submit_answer", "description" => "Submit your final answer. Correct: +10, Wrong: -5.",
             "input_schema" => Dict("type" => "object", "properties" => Dict(
                 "answer_index" => Dict("type" => "integer", "description" => "Candidate index (0-3)")),
                 "required" => ["answer_index"])),
        Dict("name" => "abstain", "description" => "Choose not to answer. Score: 0.",
             "input_schema" => Dict("type" => "object", "properties" => Dict())),
    ]
end

# ─── System prompt ───

const SYSTEM_PROMPT_TEMPLATE = """
You are answering multiple-choice questions to maximise your score.

Scoring:
- Correct answer: +10 points
- Wrong answer: -5 points
- Abstain (choose not to answer): 0 points
- Each tool call costs points (shown in tool descriptions)

Your goal is to maximise total score. This means:
- Don't call tools if you're already confident
- Prefer cheaper tools when they're likely sufficient
- Abstain rather than guess if you're uncertain
- Consider whether additional tool calls are worth their cost

Recent performance (last 10 questions):
{HISTORY}"""

# ─── API calls ───

function call_ollama(backend::OllamaBackend, messages::Vector, tools::Vector; timeout::Int=60)
    body = Dict("model" => backend.model, "messages" => messages,
                "tools" => tools, "stream" => false,
                "options" => Dict("temperature" => 0.0, "num_predict" => 300))
    resp = HTTP.post("$(backend.host)/api/chat";
                     body=JSON3.write(body),
                     headers=["Content-Type" => "application/json"],
                     readtimeout=timeout, retry=false)
    JSON3.read(resp.body)
end

function call_anthropic(backend::AnthropicBackend, system::String, messages::Vector, tools::Vector; timeout::Int=60)
    body = Dict("model" => backend.model, "max_tokens" => 1024,
                "system" => system, "messages" => messages, "tools" => tools,
                "temperature" => 0.0)
    resp = HTTP.post("https://api.anthropic.com/v1/messages";
                     body=JSON3.write(body),
                     headers=["Content-Type" => "application/json",
                              "x-api-key" => backend.api_key,
                              "anthropic-version" => "2023-06-01"],
                     readtimeout=timeout, retry=false)
    JSON3.read(resp.body)
end

# ─── Response parsing ───

struct ToolCall
    id::String
    name::String
    arguments::Dict{String,Any}
end

function parse_ollama_response(data)
    msg = data.message
    if hasproperty(msg, :tool_calls) && !isnothing(msg.tool_calls) && length(msg.tool_calls) > 0
        tc = msg.tool_calls[1]
        fname = String(tc.function.name)
        args = Dict{String,Any}()
        if hasproperty(tc.function, :arguments)
            for (k, v) in pairs(tc.function.arguments)
                args[String(k)] = v
            end
        end
        return ToolCall("", fname, args)
    end
    # No tool call — try to parse text as answer
    text = hasproperty(msg, :content) ? String(msg.content) : ""
    return parse_text_fallback(text)
end

function parse_anthropic_response(data)
    for block in data.content
        if block.type == "tool_use"
            args = Dict{String,Any}()
            if hasproperty(block, :input)
                for (k, v) in pairs(block.input)
                    args[String(k)] = v
                end
            end
            return ToolCall(String(block.id), String(block.name), args)
        end
    end
    # No tool call — look for text
    for block in data.content
        if block.type == "text"
            return parse_text_fallback(String(block.text))
        end
    end
    return ToolCall("", "abstain", Dict{String,Any}())
end

function parse_text_fallback(text::String)
    # Attempt to extract an answer index from plain text
    m = match(r"\b([0-3])\b", text)
    if m !== nothing
        idx = parse(Int, m.captures[1])
        return ToolCall("", "submit_answer", Dict{String,Any}("answer_index" => idx))
    end
    return ToolCall("", "abstain", Dict{String,Any}())
end

# ─── Message builders ───

function build_user_message(q::Question)
    candidates = join(["  $i: $(q.candidates[i+1])" for i in 0:3], "\n")
    "Question: $(q.text)\nCandidates:\n$candidates"
end

function tool_result_text(q::Question, response_idx::Int)
    "Based on my analysis, the answer is: $(q.candidates[response_idx + 1])"
end

# Ollama: append assistant message with tool_calls, then tool result
function append_ollama_tool_exchange!(messages::Vector, tc::ToolCall, result_text::String)
    push!(messages, Dict("role" => "assistant", "content" => "",
                         "tool_calls" => [Dict("function" => Dict(
                             "name" => tc.name,
                             "arguments" => tc.arguments))]))
    push!(messages, Dict("role" => "tool", "content" => result_text))
end

# Anthropic: append assistant message with tool_use block, then user message with tool_result
function append_anthropic_tool_exchange!(messages::Vector, tc::ToolCall, result_text::String)
    push!(messages, Dict("role" => "assistant",
                         "content" => [Dict("type" => "tool_use", "id" => tc.id,
                                            "name" => tc.name, "input" => tc.arguments)]))
    push!(messages, Dict("role" => "user",
                         "content" => [Dict("type" => "tool_result",
                                            "tool_use_id" => tc.id,
                                            "content" => result_text)]))
end

# ─── Per-question loop ───

function run_question(backend::OllamaBackend, tools_sim::Vector{SimulatedTool},
                      q::Question, qi::Int, response_table::Matrix{Int},
                      system_prompt::String)
    messages = Dict{String,Any}[
        Dict{String,Any}("role" => "system", "content" => system_prompt),
        Dict{String,Any}("role" => "user", "content" => build_user_message(q)),
    ]
    tool_defs = ollama_tool_defs()
    tools_queried = Int[]
    tool_responses = Dict{Int,Int}()
    q_cost = 0.0
    sim_calls = 0
    total_in_tok = 0
    total_out_tok = 0

    for _ in 1:8  # safety bound
        data = try
            call_ollama(backend, messages, tool_defs)
        catch e
            @warn "Ollama call failed" exception=e
            return (nothing, nothing, tools_queried, tool_responses, q_cost, total_in_tok, total_out_tok)
        end

        # Accumulate tokens
        if hasproperty(data, :prompt_eval_count); total_in_tok += Int(data.prompt_eval_count); end
        if hasproperty(data, :eval_count); total_out_tok += Int(data.eval_count); end

        tc = parse_ollama_response(data)

        if tc.name == "submit_answer"
            raw = get(tc.arguments, "answer_index", 0)
            idx = raw isa AbstractString ? parse(Int, raw) : Int(raw)
            return (idx, true, tools_queried, tool_responses, q_cost, total_in_tok, total_out_tok)
        elseif tc.name == "abstain"
            return (nothing, nothing, tools_queried, tool_responses, q_cost, total_in_tok, total_out_tok)
        elseif haskey(SIMULATED_TOOL_MAP, tc.name)
            t = SIMULATED_TOOL_MAP[tc.name]
            push!(tools_queried, t)
            response = response_table[qi, t]
            tool_responses[t] = response
            q_cost += tools_sim[t].cost
            sim_calls += 1

            result_text = tool_result_text(q, response)
            append_ollama_tool_exchange!(messages, tc, result_text)

            if sim_calls >= 4
                tool_defs = filter(td -> td["function"]["name"] in ("submit_answer", "abstain"), tool_defs)
            end
        else
            return (nothing, nothing, tools_queried, tool_responses, q_cost, total_in_tok, total_out_tok)
        end
    end
    (nothing, nothing, tools_queried, tool_responses, q_cost, total_in_tok, total_out_tok)
end

function run_question(backend::AnthropicBackend, tools_sim::Vector{SimulatedTool},
                      q::Question, qi::Int, response_table::Matrix{Int},
                      system_prompt::String)
    messages = Dict{String,Any}[
        Dict{String,Any}("role" => "user", "content" => build_user_message(q)),
    ]
    tool_defs = anthropic_tool_defs()
    tools_queried = Int[]
    tool_responses = Dict{Int,Int}()
    q_cost = 0.0
    sim_calls = 0
    total_in_tok = 0
    total_out_tok = 0

    for _ in 1:8  # safety bound
        data = try
            call_anthropic(backend, system_prompt, messages, tool_defs)
        catch e
            @warn "Anthropic call failed" exception=e
            return (nothing, nothing, tools_queried, tool_responses, q_cost, total_in_tok, total_out_tok)
        end

        # Accumulate tokens
        if hasproperty(data, :usage)
            total_in_tok += Int(data.usage.input_tokens)
            total_out_tok += Int(data.usage.output_tokens)
        end

        tc = parse_anthropic_response(data)

        if tc.name == "submit_answer"
            raw = get(tc.arguments, "answer_index", 0)
            idx = raw isa AbstractString ? parse(Int, raw) : Int(raw)
            return (idx, true, tools_queried, tool_responses, q_cost, total_in_tok, total_out_tok)
        elseif tc.name == "abstain"
            return (nothing, nothing, tools_queried, tool_responses, q_cost, total_in_tok, total_out_tok)
        elseif haskey(SIMULATED_TOOL_MAP, tc.name)
            t = SIMULATED_TOOL_MAP[tc.name]
            push!(tools_queried, t)
            response = response_table[qi, t]
            tool_responses[t] = response
            q_cost += tools_sim[t].cost
            sim_calls += 1

            result_text = tool_result_text(q, response)
            append_anthropic_tool_exchange!(messages, tc, result_text)

            if sim_calls >= 4
                tool_defs = filter(td -> td["name"] in ("submit_answer", "abstain"), tool_defs)
            end
        else
            return (nothing, nothing, tools_queried, tool_responses, q_cost, total_in_tok, total_out_tok)
        end
    end
    (nothing, nothing, tools_queried, tool_responses, q_cost, total_in_tok, total_out_tok)
end

# ─── Seed runner ───

function run_llm_seed(backend::LLMBackend, tools::Vector{SimulatedTool},
                      questions::Vector{Question}, response_table::Matrix{Int},
                      seed::Int)
    records = QuestionResult[]
    total_reward = 0.0
    total_tool_cost = 0.0
    history_buffer = String[]
    t_start = time()

    for (qi, q) in enumerate(questions)
        # Build system prompt with history
        history_text = if isempty(history_buffer)
            "(no history yet)"
        else
            join(history_buffer[max(1, end-9):end], "\n")
        end
        system_prompt = replace(SYSTEM_PROMPT_TEMPLATE, "{HISTORY}" => history_text)

        q_t0 = time()
        submitted_raw, _, tools_queried, tool_responses, q_cost, in_tok, out_tok = run_question(
            backend, tools, q, qi, response_table, system_prompt)
        q_wall = time() - q_t0

        total_tool_cost += q_cost
        tool_names = join([tools[t].name for t in tools_queried], ", ")
        if isempty(tool_names); tool_names = "none"; end
        tok_str = in_tok + out_tok > 0 ? " [$(in_tok)+$(out_tok) tok]" : ""

        if submitted_raw !== nothing
            submitted = Int(submitted_raw)
            was_correct = submitted == q.correct_index
            reward = was_correct ? REWARD_CORRECT : PENALTY_WRONG
            total_reward += reward
            push!(records, QuestionResult(q.id, q.category, tools_queried,
                tool_responses, submitted, was_correct, reward, q_cost, q_wall, in_tok, out_tok))

            net = reward - q_cost
            result_str = was_correct ? "correct" : "WRONG"
            println(stderr, @sprintf("    seed %d q %2d/%-2d [%-15s] %-3s: %s → %s (%+.0f net)%s",
                seed, qi, length(questions), q.category, q.id, tool_names, result_str, net, tok_str))
            flush(stderr)

            push!(history_buffer,
                "[$(q.category)] tools: $tool_names (cost: $(Int(q_cost))) → $(lowercase(result_str)) ($(@sprintf("%+.0f", net)) net)")
        else
            total_reward += REWARD_ABSTAIN
            push!(records, QuestionResult(q.id, q.category, tools_queried,
                tool_responses, nothing, nothing, REWARD_ABSTAIN, q_cost, q_wall, in_tok, out_tok))

            println(stderr, @sprintf("    seed %d q %2d/%-2d [%-15s] %-3s: %s → abstain (cost: %.0f)%s",
                seed, qi, length(questions), q.category, q.id, tool_names, q_cost, tok_str))
            flush(stderr)

            push!(history_buffer,
                "[$(q.category)] tools: $tool_names (cost: $(Int(q_cost))) → abstained")
        end
    end

    wall_time = time() - t_start
    SeedResult(seed, records, total_reward - total_tool_cost,
               total_reward, total_tool_cost, wall_time)
end

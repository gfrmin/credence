"""
    environment.jl — Simulated tools and question bank for the QA benchmark.

Tools have category-dependent reliability. All tools always respond (no coverage
mechanism). Questions are 4-choice multiple-choice across 5 categories.
"""

using Random

# ─── Categories ───

const CATEGORIES = ("factual", "numerical", "recent_events", "misconceptions", "reasoning")

# ─── Scoring ───

const REWARD_CORRECT = 10.0
const PENALTY_WRONG  = -5.0
const REWARD_ABSTAIN =  0.0

# ─── Tools ───

struct SimulatedTool
    name::String
    cost::Float64
    reliability_by_category::Dict{String,Float64}
end

"""
    query_tool(tool, question, rng) → candidate_idx::Int

All tools always respond. If rng < reliability → correct answer, else uniform
random wrong answer from the 3 incorrect candidates.
"""
function query_tool(tool::SimulatedTool, question, rng)
    reliability = get(tool.reliability_by_category, question.category, 0.25)
    if rand(rng) < reliability
        return question.correct_index
    end
    wrong = [i for i in 0:3 if i != question.correct_index]
    return wrong[rand(rng, 1:3)]
end

function make_spec_tools()
    tool_a = SimulatedTool("web_search", 1.0,
        Dict("factual" => 0.70, "numerical" => 0.20, "recent_events" => 0.65,
             "misconceptions" => 0.25, "reasoning" => 0.40))

    tool_b = SimulatedTool("knowledge_base", 2.0,
        Dict("factual" => 0.92, "numerical" => 0.40, "recent_events" => 0.55,
             "misconceptions" => 0.88, "reasoning" => 0.45))

    tool_c = SimulatedTool("calculator", 1.0,
        Dict("factual" => 0.25, "numerical" => 1.00, "recent_events" => 0.25,
             "misconceptions" => 0.25, "reasoning" => 0.25))

    tool_d = SimulatedTool("llm_direct", 2.0,
        Dict("factual" => 0.65, "numerical" => 0.50, "recent_events" => 0.45,
             "misconceptions" => 0.40, "reasoning" => 0.72))

    [tool_a, tool_b, tool_c, tool_d]
end

# ─── Questions ───

struct Question
    id::String
    text::String
    candidates::NTuple{4,String}
    correct_index::Int   # 0-indexed (matches DSL answer-space)
    category::String
    difficulty::String
end

const QUESTION_BANK = Question[
    # --- 15 Factual ---
    Question("f01", "Which country has the largest coastline?", ("Russia","Canada","Indonesia","Australia"), 1, "factual", "medium"),
    Question("f02", "What is the capital of Myanmar?", ("Yangon","Mandalay","Naypyidaw","Bago"), 2, "factual", "hard"),
    Question("f03", "Which element has the chemical symbol 'W'?", ("Tungsten","Wolfram","Vanadium","Tellurium"), 0, "factual", "medium"),
    Question("f04", "What is the longest river in Africa?", ("Congo","Niger","Zambezi","Nile"), 3, "factual", "easy"),
    Question("f05", "Which planet has the most moons?", ("Jupiter","Saturn","Uranus","Neptune"), 1, "factual", "medium"),
    Question("f06", "In which year was the United Nations founded?", ("1943","1944","1945","1946"), 2, "factual", "medium"),
    Question("f07", "What is the smallest country in the world by area?", ("Monaco","Vatican City","San Marino","Nauru"), 1, "factual", "easy"),
    Question("f08", "Which ocean is the deepest?", ("Atlantic","Indian","Pacific","Arctic"), 2, "factual", "easy"),
    Question("f09", "What is the official language of Brazil?", ("Spanish","Portuguese","French","Italian"), 1, "factual", "easy"),
    Question("f10", "Which desert is the largest hot desert in the world?", ("Arabian","Gobi","Kalahari","Sahara"), 3, "factual", "easy"),
    Question("f11", "Who painted the ceiling of the Sistine Chapel?", ("Leonardo da Vinci","Raphael","Michelangelo","Donatello"), 2, "factual", "easy"),
    Question("f12", "What is the hardest natural substance?", ("Corundum","Diamond","Topaz","Quartz"), 1, "factual", "easy"),
    Question("f13", "How many US states border the Pacific Ocean?", ("3","4","5","6"), 2, "factual", "medium"),
    Question("f14", "Which country has the most time zones?", ("Russia","United States","France","China"), 2, "factual", "hard"),
    Question("f15", "What is the most abundant gas in Earth's atmosphere?", ("Oxygen","Carbon dioxide","Nitrogen","Argon"), 2, "factual", "easy"),
    # --- 10 Numerical ---
    Question("n01", "What is 17% of 4,230?", ("718.1","719.1","721.1","723.1"), 1, "numerical", "easy"),
    Question("n02", "What is the square root of 1,764?", ("38","40","42","44"), 2, "numerical", "medium"),
    Question("n03", "If a car travels at 65 mph for 3.5 hours, how far does it go?", ("215.5 miles","222.5 miles","227.5 miles","232.5 miles"), 2, "numerical", "easy"),
    Question("n04", "What is 2^15?", ("16384","32768","65536","8192"), 1, "numerical", "medium"),
    Question("n05", "A recipe calls for 3/4 cup of sugar. How many cups for 5 batches?", ("3.25 cups","3.5 cups","3.75 cups","4.0 cups"), 2, "numerical", "easy"),
    Question("n06", "How many bones are in the adult human body?", ("196","206","216","226"), 1, "numerical", "medium"),
    Question("n07", "What is 15% tip on a \$86.40 bill?", ("\$11.96","\$12.96","\$13.96","\$14.96"), 1, "numerical", "easy"),
    Question("n08", "If you invest \$1,000 at 5% annual compound interest, what do you have after 3 years (nearest dollar)?", ("\$1,150","\$1,158","\$1,166","\$1,103"), 1, "numerical", "hard"),
    Question("n09", "What is the sum of the first 20 positive integers?", ("190","200","210","220"), 2, "numerical", "medium"),
    Question("n10", "A circle has radius 7. What is its area (nearest integer)?", ("144","148","154","158"), 2, "numerical", "medium"),
    # --- 8 Recent Events ---
    Question("r01", "Who won the 2024 Nobel Prize in Physics?", ("John Hopfield and Geoffrey Hinton","Alain Aspect and John Clauser","Syukuro Manabe and Klaus Hasselmann","Pierre Agostini and Ferenc Krausz"), 0, "recent_events", "medium"),
    Question("r02", "Which country hosted the 2024 Summer Olympics?", ("Japan","United States","France","Australia"), 2, "recent_events", "easy"),
    Question("r03", "Who won the 2024 US Presidential Election?", ("Joe Biden","Donald Trump","Kamala Harris","Ron DeSantis"), 1, "recent_events", "easy"),
    Question("r04", "Which spacecraft made the first successful private Moon landing in 2024?", ("Intuitive Machines Odysseus","Astrobotic Peregrine","ispace Hakuto-R","SpaceX Starship"), 0, "recent_events", "hard"),
    Question("r05", "What is the current population of the world (approximate, 2024)?", ("7.5 billion","7.8 billion","8.1 billion","8.5 billion"), 2, "recent_events", "medium"),
    Question("r06", "Who won the 2024 Nobel Prize in Literature?", ("Jon Fosse","Han Kang","Olga Tokarczuk","Annie Ernaux"), 1, "recent_events", "hard"),
    Question("r07", "Which team won the 2024 UEFA European Championship (Euro 2024)?", ("England","France","Spain","Germany"), 2, "recent_events", "medium"),
    Question("r08", "Which AI model family was released by Anthropic in 2024?", ("Claude 2","Claude 3","Claude 4","Claude Opus"), 1, "recent_events", "medium"),
    # --- 7 Misconceptions ---
    Question("m01", "Which is physically larger in diameter, a US nickel or a US dime?", ("A dime","They are the same size","A nickel","It depends on the year"), 2, "misconceptions", "medium"),
    Question("m02", "What percentage of the brain does a human typically use?", ("10%","30%","50%","Virtually all of it"), 3, "misconceptions", "medium"),
    Question("m03", "Which wall of China is visible from space with the naked eye?", ("The Great Wall","The Ming Wall","The Qin Wall","None of them"), 3, "misconceptions", "medium"),
    Question("m04", "How long does it take for food to digest in the stomach?", ("30 minutes","2-5 hours","12 hours","24 hours"), 1, "misconceptions", "hard"),
    Question("m05", "What colour is a polar bear's skin?", ("White","Pink","Black","Grey"), 2, "misconceptions", "hard"),
    Question("m06", "Do goldfish have a memory span of only 3 seconds?", ("Yes, about 3 seconds","Yes, about 10 seconds","No, they can remember for months","No, about 30 seconds"), 2, "misconceptions", "easy"),
    Question("m07", "What happens if you touch a baby bird — will the mother reject it?", ("Yes, the scent causes rejection","Yes, but only for songbirds","No, most birds have a poor sense of smell","It depends on the species"), 2, "misconceptions", "medium"),
    # --- 10 Reasoning ---
    Question("g01", "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", ("Yes, definitely","No, that does not follow","Only if most flowers fade","Only for wild roses"), 1, "reasoning", "medium"),
    Question("g02", "A bat and a ball together cost \$1.10. The bat costs \$1 more than the ball. How much does the ball cost?", ("\$0.10","\$0.05","\$0.15","\$0.01"), 1, "reasoning", "medium"),
    Question("g03", "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?", ("100 minutes","5 minutes","20 minutes","1 minute"), 1, "reasoning", "medium"),
    Question("g04", "A farmer has 17 sheep. All but 9 die. How many are left?", ("8","9","17","0"), 1, "reasoning", "easy"),
    Question("g05", "In a race, you overtake the person in 2nd place. What position are you in?", ("1st","2nd","3rd","It depends on total runners"), 1, "reasoning", "easy"),
    Question("g06", "Three friends split a \$30 hotel room equally. The manager returns \$5. The bellboy keeps \$2 and gives \$1 back to each friend. Each friend paid \$9 (total \$27) plus \$2 the bellboy kept = \$29. Where is the missing dollar?", ("The hotel has it","There is no missing dollar — the question is misleading","The bellboy has it","It was lost in rounding"), 1, "reasoning", "hard"),
    Question("g07", "If you have a 4-litre jug and a 9-litre jug, which of these amounts can you NOT measure exactly?", ("1 litre","5 litres","6 litres","3 litres"), 3, "reasoning", "hard"),
    Question("g08", "A woman has two children. One of them is a boy born on a Tuesday. What is the probability the other child is also a boy?", ("1/2","1/3","13/27","1/4"), 2, "reasoning", "hard"),
    Question("g09", "If statement A implies statement B, and B is false, what can we conclude?", ("A is true","A is false","B is true","Nothing about A"), 1, "reasoning", "medium"),
    Question("g10", "You have 12 identical-looking coins. One is counterfeit and weighs differently. What is the minimum number of weighings on a balance scale to find it?", ("2","3","4","6"), 1, "reasoning", "hard"),
]

function get_questions(; seed::Int=0)
    qs = copy(QUESTION_BANK)
    shuffle!(MersenneTwister(seed), qs)
    qs
end

"""
    generate_response_table(tools, questions, rng) → Matrix{Int}

Pre-generate all tool responses for all questions under one RNG pass.
Entry [q, t] is the response that tool t gives to question q.
All agents see identical tool outputs for identical (question, tool) pairs.
"""
function generate_response_table(tools::Vector{SimulatedTool}, questions::Vector{Question}, rng)
    table = Matrix{Int}(undef, length(questions), length(tools))
    for (qi, q) in enumerate(questions)
        for (ti, t) in enumerate(tools)
            table[qi, ti] = query_tool(t, q, rng)
        end
    end
    table
end

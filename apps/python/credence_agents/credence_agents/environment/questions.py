# Role: body
"""Question bank: 50 hand-curated multiple-choice questions across 5 categories.

Distribution: 15 factual, 10 numerical, 8 recent_events, 7 misconceptions, 10 reasoning.
Each question has 4 candidates with exactly one correct answer. Wrong answers are
plausible — numerically close, commonly confused, or logically tempting.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Question(NamedTuple):
    id: str
    text: str
    candidates: tuple[str, str, str, str]
    correct_index: int
    category: str
    difficulty: str  # "easy" | "medium" | "hard"


# --- 15 Factual Questions ---

_FACTUAL = (
    Question(
        id="f01", text="Which country has the largest coastline?",
        candidates=("Russia", "Canada", "Indonesia", "Australia"),
        correct_index=1, category="factual", difficulty="medium",
    ),
    Question(
        id="f02", text="What is the capital of Myanmar?",
        candidates=("Yangon", "Mandalay", "Naypyidaw", "Bago"),
        correct_index=2, category="factual", difficulty="hard",
    ),
    Question(
        id="f03", text="Which element has the chemical symbol 'W'?",
        candidates=("Tungsten", "Wolfram", "Vanadium", "Tellurium"),
        correct_index=0, category="factual", difficulty="medium",
    ),
    Question(
        id="f04", text="What is the longest river in Africa?",
        candidates=("Congo", "Niger", "Zambezi", "Nile"),
        correct_index=3, category="factual", difficulty="easy",
    ),
    Question(
        id="f05", text="Which planet has the most moons?",
        candidates=("Jupiter", "Saturn", "Uranus", "Neptune"),
        correct_index=1, category="factual", difficulty="medium",
    ),
    Question(
        id="f06", text="In which year was the United Nations founded?",
        candidates=("1943", "1944", "1945", "1946"),
        correct_index=2, category="factual", difficulty="medium",
    ),
    Question(
        id="f07", text="What is the smallest country in the world by area?",
        candidates=("Monaco", "Vatican City", "San Marino", "Nauru"),
        correct_index=1, category="factual", difficulty="easy",
    ),
    Question(
        id="f08", text="Which ocean is the deepest?",
        candidates=("Atlantic", "Indian", "Pacific", "Arctic"),
        correct_index=2, category="factual", difficulty="easy",
    ),
    Question(
        id="f09", text="What is the official language of Brazil?",
        candidates=("Spanish", "Portuguese", "French", "Italian"),
        correct_index=1, category="factual", difficulty="easy",
    ),
    Question(
        id="f10", text="Which desert is the largest hot desert in the world?",
        candidates=("Arabian", "Gobi", "Kalahari", "Sahara"),
        correct_index=3, category="factual", difficulty="easy",
    ),
    Question(
        id="f11", text="Who painted the ceiling of the Sistine Chapel?",
        candidates=("Leonardo da Vinci", "Raphael", "Michelangelo", "Donatello"),
        correct_index=2, category="factual", difficulty="easy",
    ),
    Question(
        id="f12", text="What is the hardest natural substance?",
        candidates=("Corundum", "Diamond", "Topaz", "Quartz"),
        correct_index=1, category="factual", difficulty="easy",
    ),
    # Ambiguous: factual/numerical
    Question(
        id="f13", text="How many US states border the Pacific Ocean?",
        candidates=("3", "4", "5", "6"),
        correct_index=2, category="factual", difficulty="medium",
    ),
    Question(
        id="f14", text="Which country has the most time zones?",
        candidates=("Russia", "United States", "France", "China"),
        correct_index=2, category="factual", difficulty="hard",
    ),
    Question(
        id="f15", text="What is the most abundant gas in Earth's atmosphere?",
        candidates=("Oxygen", "Carbon dioxide", "Nitrogen", "Argon"),
        correct_index=2, category="factual", difficulty="easy",
    ),
)

# --- 10 Numerical Questions ---

_NUMERICAL = (
    Question(
        id="n01", text="What is 17% of 4,230?",
        candidates=("718.1", "719.1", "721.1", "723.1"),
        correct_index=1, category="numerical", difficulty="easy",
    ),
    Question(
        id="n02", text="What is the square root of 1,764?",
        candidates=("38", "40", "42", "44"),
        correct_index=2, category="numerical", difficulty="medium",
    ),
    Question(
        id="n03", text="If a car travels at 65 mph for 3.5 hours, how far does it go?",
        candidates=("215.5 miles", "222.5 miles", "227.5 miles", "232.5 miles"),
        correct_index=2, category="numerical", difficulty="easy",
    ),
    Question(
        id="n04", text="What is 2^15?",
        candidates=("16384", "32768", "65536", "8192"),
        correct_index=1, category="numerical", difficulty="medium",
    ),
    Question(
        id="n05", text="A recipe calls for 3/4 cup of sugar. How many cups for 5 batches?",
        candidates=("3.25 cups", "3.5 cups", "3.75 cups", "4.0 cups"),
        correct_index=2, category="numerical", difficulty="easy",
    ),
    # Ambiguous: numerical/factual
    Question(
        id="n06", text="How many bones are in the adult human body?",
        candidates=("196", "206", "216", "226"),
        correct_index=1, category="numerical", difficulty="medium",
    ),
    Question(
        id="n07", text="What is 15% tip on a $86.40 bill?",
        candidates=("$11.96", "$12.96", "$13.96", "$14.96"),
        correct_index=1, category="numerical", difficulty="easy",
    ),
    Question(
        id="n08", text="If you invest $1,000 at 5% annual compound interest, "
                       "what do you have after 3 years (nearest dollar)?",
        candidates=("$1,150", "$1,158", "$1,166", "$1,103"),
        correct_index=1, category="numerical", difficulty="hard",
    ),
    Question(
        id="n09", text="What is the sum of the first 20 positive integers?",
        candidates=("190", "200", "210", "220"),
        correct_index=2, category="numerical", difficulty="medium",
    ),
    Question(
        id="n10", text="A circle has radius 7. What is its area (nearest integer)?",
        candidates=("144", "148", "154", "158"),
        correct_index=2, category="numerical", difficulty="medium",
    ),
)

# --- 8 Recent Events Questions ---

_RECENT_EVENTS = (
    Question(
        id="r01", text="Who won the 2024 Nobel Prize in Physics?",
        candidates=(
            "John Hopfield and Geoffrey Hinton",
            "Alain Aspect and John Clauser",
            "Syukuro Manabe and Klaus Hasselmann",
            "Pierre Agostini and Ferenc Krausz",
        ),
        correct_index=0, category="recent_events", difficulty="medium",
    ),
    Question(
        id="r02", text="Which country hosted the 2024 Summer Olympics?",
        candidates=("Japan", "United States", "France", "Australia"),
        correct_index=2, category="recent_events", difficulty="easy",
    ),
    Question(
        id="r03", text="Who won the 2024 US Presidential Election?",
        candidates=("Joe Biden", "Donald Trump", "Kamala Harris", "Ron DeSantis"),
        correct_index=1, category="recent_events", difficulty="easy",
    ),
    Question(
        id="r04", text="Which spacecraft made the first successful private Moon landing in 2024?",
        candidates=("Intuitive Machines Odysseus", "Astrobotic Peregrine",
                     "ispace Hakuto-R", "SpaceX Starship"),
        correct_index=0, category="recent_events", difficulty="hard",
    ),
    # Ambiguous: recent_events/factual
    Question(
        id="r05", text="What is the current population of the world (approximate, 2024)?",
        candidates=("7.5 billion", "7.8 billion", "8.1 billion", "8.5 billion"),
        correct_index=2, category="recent_events", difficulty="medium",
    ),
    Question(
        id="r06", text="Who won the 2024 Nobel Prize in Literature?",
        candidates=("Jon Fosse", "Han Kang", "Olga Tokarczuk", "Annie Ernaux"),
        correct_index=1, category="recent_events", difficulty="hard",
    ),
    Question(
        id="r07", text="Which team won the 2024 UEFA European Championship (Euro 2024)?",
        candidates=("England", "France", "Spain", "Germany"),
        correct_index=2, category="recent_events", difficulty="medium",
    ),
    Question(
        id="r08", text="Which AI model family was released by Anthropic in 2024?",
        candidates=("Claude 2", "Claude 3", "Claude 4", "Claude Opus"),
        correct_index=1, category="recent_events", difficulty="medium",
    ),
)

# --- 7 Misconceptions Questions ---

_MISCONCEPTIONS = (
    Question(
        id="m01", text="Which is physically larger in diameter, a US nickel or a US dime?",
        candidates=("A dime", "They are the same size", "A nickel", "It depends on the year"),
        correct_index=2, category="misconceptions", difficulty="medium",
    ),
    Question(
        id="m02", text="What percentage of the brain does a human typically use?",
        candidates=("10%", "30%", "50%", "Virtually all of it"),
        correct_index=3, category="misconceptions", difficulty="medium",
    ),
    Question(
        id="m03", text="Which wall of China is visible from space with the naked eye?",
        candidates=("The Great Wall", "The Ming Wall", "The Qin Wall", "None of them"),
        correct_index=3, category="misconceptions", difficulty="medium",
    ),
    Question(
        id="m04", text="How long does it take for food to digest in the stomach?",
        candidates=("30 minutes", "2-5 hours", "12 hours", "24 hours"),
        correct_index=1, category="misconceptions", difficulty="hard",
    ),
    # Ambiguous: misconceptions/factual
    Question(
        id="m05", text="What colour is a polar bear's skin?",
        candidates=("White", "Pink", "Black", "Grey"),
        correct_index=2, category="misconceptions", difficulty="hard",
    ),
    Question(
        id="m06", text="Do goldfish have a memory span of only 3 seconds?",
        candidates=("Yes, about 3 seconds", "Yes, about 10 seconds",
                     "No, they can remember for months", "No, about 30 seconds"),
        correct_index=2, category="misconceptions", difficulty="easy",
    ),
    Question(
        id="m07", text="What happens if you touch a baby bird — will the mother reject it?",
        candidates=(
            "Yes, the scent causes rejection",
            "Yes, but only for songbirds",
            "No, most birds have a poor sense of smell",
            "It depends on the species",
        ),
        correct_index=2, category="misconceptions", difficulty="medium",
    ),
)

# --- 10 Reasoning Questions ---

_REASONING = (
    Question(
        id="g01",
        text="If all roses are flowers and some flowers fade quickly, "
             "can we conclude that some roses fade quickly?",
        candidates=("Yes, definitely", "No, that does not follow",
                     "Only if most flowers fade", "Only for wild roses"),
        correct_index=1, category="reasoning", difficulty="medium",
    ),
    Question(
        id="g02",
        text="A bat and a ball together cost $1.10. The bat costs $1 more than the ball. "
             "How much does the ball cost?",
        candidates=("$0.10", "$0.05", "$0.15", "$0.01"),
        correct_index=1, category="reasoning", difficulty="medium",
    ),
    Question(
        id="g03",
        text="If it takes 5 machines 5 minutes to make 5 widgets, "
             "how long does it take 100 machines to make 100 widgets?",
        candidates=("100 minutes", "5 minutes", "20 minutes", "1 minute"),
        correct_index=1, category="reasoning", difficulty="medium",
    ),
    Question(
        id="g04",
        text="A farmer has 17 sheep. All but 9 die. How many are left?",
        candidates=("8", "9", "17", "0"),
        correct_index=1, category="reasoning", difficulty="easy",
    ),
    Question(
        id="g05",
        text="In a race, you overtake the person in 2nd place. What position are you in?",
        candidates=("1st", "2nd", "3rd", "It depends on total runners"),
        correct_index=1, category="reasoning", difficulty="easy",
    ),
    # Ambiguous: reasoning/numerical
    Question(
        id="g06",
        text="Three friends split a $30 hotel room equally. The manager returns $5. "
             "The bellboy keeps $2 and gives $1 back to each friend. "
             "Each friend paid $9 (total $27) plus $2 the bellboy kept = $29. "
             "Where is the missing dollar?",
        candidates=(
            "The hotel has it",
            "There is no missing dollar — the question is misleading",
            "The bellboy has it",
            "It was lost in rounding",
        ),
        correct_index=1, category="reasoning", difficulty="hard",
    ),
    Question(
        id="g07",
        text="If you have a 4-litre jug and a 9-litre jug, which of these amounts "
             "can you NOT measure exactly?",
        candidates=("1 litre", "5 litres", "6 litres", "3 litres"),
        correct_index=3, category="reasoning", difficulty="hard",
    ),
    Question(
        id="g08",
        text="A woman has two children. One of them is a boy born on a Tuesday. "
             "What is the probability the other child is also a boy?",
        candidates=("1/2", "1/3", "13/27", "1/4"),
        correct_index=2, category="reasoning", difficulty="hard",
    ),
    Question(
        id="g09",
        text="If statement A implies statement B, and B is false, what can we conclude?",
        candidates=("A is true", "A is false", "B is true", "Nothing about A"),
        correct_index=1, category="reasoning", difficulty="medium",
    ),
    Question(
        id="g10",
        text="You have 12 identical-looking coins. One is counterfeit and weighs differently. "
             "What is the minimum number of weighings on a balance scale to find it?",
        candidates=("2", "3", "4", "6"),
        correct_index=1, category="reasoning", difficulty="hard",
    ),
)

# --- Combined bank ---

QUESTION_BANK: tuple[Question, ...] = _FACTUAL + _NUMERICAL + _RECENT_EVENTS + _MISCONCEPTIONS + _REASONING


def get_questions(seed: int | None = None) -> list[Question]:
    """Return all 50 questions. Shuffled if seed is provided, original order if None."""
    questions = list(QUESTION_BANK)
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(questions)
    return questions

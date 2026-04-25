# Role: brain-side application
# Legal: this file would be excluded by the file-scope rule in production.
# In the corpus it is named good_* because it carries a pragma.

using Credence

function compute_mean(p::BetaPrevision)
    # credence-lint: allow — precedent:expect-through-accessor — corpus exemplar for pragma'd accessor read
    p.alpha / (p.alpha + p.beta)
end

"""
    perturbation.jl — Posterior subtree analysis, nonterminal proposal, grammar perturbation

All perturbation grounded in posterior analysis via SubprogramFrequencyTable.
"""

# ═══════════════════════════════════════
# Posterior subtree analysis
# ═══════════════════════════════════════

"""
    analyse_posterior_subtrees(programs, weights; ...) → SubprogramFrequencyTable

Walk each program's AST, extract all subtrees of depth ≥ min_complexity.
Weight each subtree occurrence by the program's posterior weight.
Aggregate across programs.
"""
function analyse_posterior_subtrees(
    programs::Vector{Program},
    prog_weights::Vector{Float64};
    min_frequency::Float64=0.1,
    min_complexity::Int=2
)::SubprogramFrequencyTable
    subtree_map = Dict{String, Tuple{ProgramExpr, Float64, Vector{Int}}}()
    # The sound reference counts (exploration-budget Move 1 + #174): full-depth walks over the SAME support
    # set (w > 1e-15) the subtree loop uses, independent of extract_subtrees' min_complexity filter. Concrete
    # (possibly empty) Sets ⇒ analysed; consumed by `_removal_payoff`/`:remove_rule` (nonterminals) and
    # `_feature_removal_payoff`/`:remove_feature` (features).
    referenced = Set{Symbol}()
    referenced_feats = Set{Symbol}()

    for (i, prog) in enumerate(programs)
        w = prog_weights[i]
        w > 1e-15 || continue
        collect_nonterminal_refs!(referenced, prog.expr)
        collect_feature_refs!(referenced_feats, prog.expr)
        subtrees = extract_subtrees(prog.expr, min_complexity)
        for st in subtrees
            key = show_expr(st)
            if haskey(subtree_map, key)
                entry = subtree_map[key]
                subtree_map[key] = (entry[1], entry[2] + w, push!(entry[3], i))
            else
                subtree_map[key] = (st, w, [i])
            end
        end
    end

    subtrees = ProgramExpr[]
    freqs = Float64[]
    sources = Vector{Int}[]

    for (_, (st, freq, src)) in subtree_map
        freq >= min_frequency || continue
        push!(subtrees, st)
        push!(freqs, freq)
        push!(sources, src)
    end

    perm = sortperm(freqs, rev=true)
    SubprogramFrequencyTable(subtrees[perm], freqs[perm], sources[perm], referenced, referenced_feats)
end

"""Extract all subtrees of an expression with complexity ≥ min_c."""
function extract_subtrees(e::ProgramExpr, min_c::Int)::Vector{ProgramExpr}
    result = ProgramExpr[]
    _extract!(result, e, min_c)
    result
end

function _extract!(result, e::GTExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
end
function _extract!(result, e::LTExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
end
function _extract!(result, e::NonterminalRef, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
end
function _extract!(result, e::AndExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
    _extract!(result, e.left, min_c)
    _extract!(result, e.right, min_c)
end
function _extract!(result, e::OrExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
    _extract!(result, e.left, min_c)
    _extract!(result, e.right, min_c)
end
function _extract!(result, e::NotExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
    _extract!(result, e.child, min_c)
end
function _extract!(result, e::PersistsExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
    _extract!(result, e.child, min_c)
end
function _extract!(result, e::ChangedExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
    _extract!(result, e.child, min_c)
end
function _extract!(result, e::SinceExpr, min_c)
    expr_complexity(e) >= min_c && push!(result, e)
    _extract!(result, e.p, min_c)
    _extract!(result, e.q, min_c)
end
function _extract!(result, e::ActionExpr, _min_c)
    # ActionExpr is not a predicate — never extract as a candidate nonterminal
end
function _extract!(result, e::IfExpr, min_c)
    # IfExpr is not a predicate — don't extract it as a candidate nonterminal.
    # Only extract predicate subtrees (Bool-valued) from the predicate branch.
    _extract!(result, e.predicate, min_c)
end

# ═══════════════════════════════════════
# Nonterminal reference count — the SOUND, full-depth walk (exploration-budget Move 1)
# ═══════════════════════════════════════

"""
    collect_nonterminal_refs!(acc::Set{Symbol}, e::ProgramExpr) → acc

Collect the names of EVERY `NonterminalRef` in `e`, recursing to all depths and into all branches
(predicate AND action branches of `IfExpr`). The soundness keystone of `:remove_rule`: a rule is
"referenced" iff some posterior-support program mentions it ANYWHERE, so a missed reference would
wrongly mark a live rule dead and remove it (silently breaking a support program).

DELIBERATELY SEPARATE from `extract_subtrees`: that function's `min_complexity` filter drops
complexity-1 (bare `NonterminalRef`) programs — exactly the lossiness that blocked a sound reference
count (collapse-towers Scope B). This walk MUST NOT be routed through `extract_subtrees`. One method
per `ProgramExpr` subtype, NO generic fallback — a future node type fails loud (matching `_extract!` /
`expr_complexity`), because a silently-unwalked node is a silent unsoundness.
Asserted by test_voc_gate.jl §5b (depth-1 reference is seen).
"""
collect_nonterminal_refs!(acc::Set{Symbol}, e::NonterminalRef) = (push!(acc, e.name); acc)
collect_nonterminal_refs!(acc::Set{Symbol}, ::GTExpr) = acc
collect_nonterminal_refs!(acc::Set{Symbol}, ::LTExpr) = acc
collect_nonterminal_refs!(acc::Set{Symbol}, ::ActionExpr) = acc
collect_nonterminal_refs!(acc::Set{Symbol}, e::AndExpr) =
    (collect_nonterminal_refs!(acc, e.left); collect_nonterminal_refs!(acc, e.right))
collect_nonterminal_refs!(acc::Set{Symbol}, e::OrExpr) =
    (collect_nonterminal_refs!(acc, e.left); collect_nonterminal_refs!(acc, e.right))
collect_nonterminal_refs!(acc::Set{Symbol}, e::NotExpr) = collect_nonterminal_refs!(acc, e.child)
collect_nonterminal_refs!(acc::Set{Symbol}, e::PersistsExpr) = collect_nonterminal_refs!(acc, e.child)
collect_nonterminal_refs!(acc::Set{Symbol}, e::ChangedExpr) = collect_nonterminal_refs!(acc, e.child)
collect_nonterminal_refs!(acc::Set{Symbol}, e::SinceExpr) =
    (collect_nonterminal_refs!(acc, e.p); collect_nonterminal_refs!(acc, e.q))
collect_nonterminal_refs!(acc::Set{Symbol}, e::IfExpr) =
    (collect_nonterminal_refs!(acc, e.predicate);
     collect_nonterminal_refs!(acc, e.then_branch);
     collect_nonterminal_refs!(acc, e.else_branch))

"""
    collect_feature_refs!(acc::Set{Symbol}, e::ProgramExpr) → acc

Collect the names of EVERY feature a predicate in `e` tests (the `.feature` of each `GTExpr`/`LTExpr`),
recursing to all depths and into all branches. The feature-side mirror of `collect_nonterminal_refs!` and
the soundness keystone of `:remove_feature` (#174): a feature is "referenced" iff some posterior-support
program tests it ANYWHERE, so a missed reference would wrongly mark a live feature dead and remove it
(deleting the support programs that use it). `NonterminalRef` contributes nothing DIRECTLY — a rule body's
features are caught by `_feature_removal_payoff`'s rule-body union (the transitive case), exactly as the
nonterminal walk handles its own transitivity. One method per `ProgramExpr` subtype, NO generic fallback —
a future node type fails loud (a silently-unwalked node is a silent unsoundness). Asserted by
test_compression_removal.jl §5–§6.
"""
# The walk sees THROUGH arithmetic (num_feature_refs!, types.jl): a product referencing :a
# keeps :a alive for :remove_feature. (Design §2's "deferred with #174" note was stale —
# collect_feature_refs! landed WITH #174 before this move, so the NumExpr recursion is a
# correctness requirement here, not a follow-up.)
collect_feature_refs!(acc::Set{Symbol}, e::GTExpr) = num_feature_refs!(acc, e.lhs)
collect_feature_refs!(acc::Set{Symbol}, e::LTExpr) = num_feature_refs!(acc, e.lhs)
collect_feature_refs!(acc::Set{Symbol}, ::NonterminalRef) = acc
collect_feature_refs!(acc::Set{Symbol}, ::ActionExpr) = acc
collect_feature_refs!(acc::Set{Symbol}, e::AndExpr) =
    (collect_feature_refs!(acc, e.left); collect_feature_refs!(acc, e.right))
collect_feature_refs!(acc::Set{Symbol}, e::OrExpr) =
    (collect_feature_refs!(acc, e.left); collect_feature_refs!(acc, e.right))
collect_feature_refs!(acc::Set{Symbol}, e::NotExpr) = collect_feature_refs!(acc, e.child)
collect_feature_refs!(acc::Set{Symbol}, e::PersistsExpr) = collect_feature_refs!(acc, e.child)
collect_feature_refs!(acc::Set{Symbol}, e::ChangedExpr) = collect_feature_refs!(acc, e.child)
collect_feature_refs!(acc::Set{Symbol}, e::SinceExpr) =
    (collect_feature_refs!(acc, e.p); collect_feature_refs!(acc, e.q))
collect_feature_refs!(acc::Set{Symbol}, e::IfExpr) =
    (collect_feature_refs!(acc, e.predicate);
     collect_feature_refs!(acc, e.then_branch);
     collect_feature_refs!(acc, e.else_branch))

# ═══════════════════════════════════════
# Compression payoff — the shared description-length arithmetic
# ═══════════════════════════════════════

"""
    _compression_payoff(table) → Union{Tuple{ProductionRule, Int}, Nothing}

The best `:add_rule` candidate and its description-length payoff (symbols saved), or `nothing` if no
posterior subtree compresses (payoff ≤ 0). Defining the most-frequent subtree `s` as a nonterminal
replaces each of its `n_sources` uses (cost `expr_complexity(s)`) with a reference (cost 1) and adds
the rule once (cost `1 + expr_complexity(s)`):

    net_payoff = n_sources · (expr_complexity(s) − 1) − (1 + expr_complexity(s))   [symbols]

This is the two-part-MDL saving (CLAUDE.md §1.3). It is the single home of the payoff arithmetic,
shared by `propose_nonterminal` (the gate) and `net_voc` (the value), so the two never drift.
"""
function _compression_payoff(table::SubprogramFrequencyTable)::Union{Tuple{ProductionRule, Int}, Nothing}
    isempty(table.subtrees) && return nothing
    best_idx = argmax(table.weighted_frequency)
    best_expr = table.subtrees[best_idx]
    n_sources = length(table.source_programs[best_idx])
    expr_c = expr_complexity(best_expr)
    net_payoff = n_sources * (expr_c - 1) - (1 + expr_c)
    net_payoff > 0 || return nothing
    # Full hash (not % 10000): the name must be injective on the subtree body. It identifies the rule
    # for downstream `NonterminalRef` lookup, AND `perturb_grammar`'s idempotence guard recognises an
    # already-abstracted subtree by name. A narrow modulus collides (birthday ~100 nonterminals),
    # which would alias two distinct subtrees to one name (an ambiguous reference) and silently drop
    # the second, genuinely-new compression at the guard. (Finding 2, PR #160 adversarial review.)
    name = Symbol("NT_", hash(show_expr(best_expr)))
    (ProductionRule(name, best_expr), net_payoff)
end

"""
    propose_nonterminal(table) → Union{ProductionRule, Nothing}

Select the highest-weighted-frequency subtree and propose it as a nonterminal iff its compression
payoff justifies the rule cost (`net_payoff > 0`). The `:add_rule` candidate generator for
`perturb_grammar`; the payoff arithmetic lives in `_compression_payoff`.
"""
function propose_nonterminal(table::SubprogramFrequencyTable)::Union{ProductionRule, Nothing}
    result = _compression_payoff(table)
    isnothing(result) ? nothing : result[1]
end

"""
    _removal_payoff(g, table) → Vector{Tuple{ProductionRule, Int}}

The `:remove_rule` candidates: each grammar rule referenced by NEITHER a posterior-support program NOR
any rule body (`r.name ∉ referenced_nonterminals ∪ ⋃_r collect_nonterminal_refs!(r.body)`). The rule-body
union is the TRANSITIVE soundness fix (#174): enumerated programs hold `NonterminalRef`s unexpanded (a
rule's body lives here in `grammar.rules`, resolved at compile time), so a rule referenced only inside
another rule's body is invisible to the program-AST walk that built `referenced_nonterminals`; without the
union it is wrongly flagged dead and removed, leaving a dangling ref → `compile_expr` crashes ("Undefined
nonterminal"). Removing a genuinely-dead rule shrinks the dictionary by its full two-part-MDL cost
`1 + expr_complexity(r.body)` symbols (the rule def: 1 for the head + the body) at ZERO fit cost — no
support program referenced it, so the belief is untouched (prior-only). The symmetric MDL partner of
`_compression_payoff`'s `:add_rule`. Referenced rules are NOT candidates: removing them is generative-change
(it would change which programs the grammar generates), invisible to depth-one prior-only `net_voc` and
deferred to a later move.

`table.referenced_nonterminals === nothing` (an un-analysed, hand-built table) ⇒ references unknown ⇒
no candidates (Scope-A-preserving). A concrete (possibly empty) Set ⇒ analysed; an empty analysed set
legitimately means every rule is dead. (See the Move 1 design doc on why the sentinel is not an empty
Set: `name ∉ ∅` is vacuously true, which would make every rule removable.)
"""
function _removal_payoff(g::Grammar, table::SubprogramFrequencyTable)::Vector{Tuple{ProductionRule, Int}}
    refs = table.referenced_nonterminals
    refs === nothing && return Tuple{ProductionRule, Int}[]
    # Transitive soundness (#174): union the program-direct refs with the nonterminal refs from ALL rule
    # bodies (Q3 — reachability-free: a symbol named only by a *dead* rule's body is kept until that rule
    # is removed first, then reclaimed next pass, matching one-perturbation-per-call). `copy` so the
    # table's shared Set is never mutated. The `nothing` default above guards the *un-analysed* table;
    # this guards *analysed-but-incomplete*. Asserted by test_compression_removal.jl §1–§3.
    all_refs = copy(refs)
    for r in g.rules
        collect_nonterminal_refs!(all_refs, r.body)
    end
    [(r, 1 + expr_complexity(r.body)) for r in g.rules if !(r.name in all_refs)]
end

"""
    _feature_removal_payoff(g, table) → Vector{Symbol}

The `:remove_feature` candidates: each feature in `g.feature_set` referenced by NEITHER a posterior-support
program NOR any rule body (`f ∉ referenced_features ∪ ⋃_r collect_feature_refs!(r.body)`). Removing such a
dead feature reclaims exactly **1 symbol** (`length(g.feature_set)` drops by one; thresholds are NOT charged
— `compute_grammar_complexity` is grid-count-invariant) at ZERO fit cost — no support program tests it, so
the belief is untouched (prior-only). The feature-side mirror of `_removal_payoff`; the rule-body union is
the same transitive-soundness device (#174). `table.referenced_features === nothing` (un-analysed) ⇒ no
candidates (fail-closed). Removing a *live* feature is destructive generative change — never a candidate.
Sorted for a deterministic candidate vector (the winner is already determinate via `_candidate_better`'s
name tiebreak; sorting removes any doubt about `Set` iteration order). Asserted by test_compression_removal.jl §6.
"""
function _feature_removal_payoff(g::Grammar, table::SubprogramFrequencyTable)::Vector{Symbol}
    refs = table.referenced_features
    refs === nothing && return Symbol[]
    all_refs = copy(refs)
    for r in g.rules
        collect_feature_refs!(all_refs, r.body)
    end
    sort([f for f in g.feature_set if !(f in all_refs)])
end

# ═══════════════════════════════════════
# net_voc — Value-of-Computation of a grammar perturbation (collapse-towers Phase 5)
# ═══════════════════════════════════════

"""
    net_voc(net_payoff_symbols, compute_cost) → Float64

The Value-of-Computation of a compression perturbation, depth-one, in **log-prior nats** (R1):
`net_value(Δcomplexity_logprior, compute_cost)`. `perturb_grammar` sees only `(g, freq_table,
available_features)` — no belief, no utilities, no re-conditioning — so achievable EU is unaffordable
depth-one (Russell–Wefald); the affordable value-proxy is the change in the program-space complexity
prior (CLAUDE.md §1.3). A compression saving `net_payoff_symbols` raises the log-prior by

    complexity_logprior(−net_payoff_symbols; λ = log(2)) = log(2) · net_payoff_symbols      [nats]

(the program-axis λ is pinned to `log(2)` by §1.3). The structural twin of `net_voi` (`stdlib.jl`): the same `net_value` form — the third representation,
after scalar `net_voi` and Functional-offset routing EU. **Move 5 (one currency, two fidelities):** this
is NOT a different currency from exploration's lookahead VOI — both are **Δ log-evidence**
(`Δlog P(g) + Δlog P(data|g)`). `net_voc` scores the **prior-only term**, dropping the likelihood term —
not because it is zero (abbreviating reweights the mixture) but because the prior-only signature (no
belief, no re-conditioning) cannot afford to measure it (Russell–Wefald). So `net_voc` is the
**cheap-surrogate** fidelity of the one currency `explore_features` computes **exactly** (the general
instance, both terms); `explore_grammar` is the `Δprior = 0` instance; `net_voc` is the Δℓ-dropped
instance. The cheap/exact fidelity gap — not a currency gap — is what the two-tier cascade orders
(`docs/exploration-budget/move-5-design.md`). At `compute_cost = 0` the gate `net_voc > 0`
is exactly `propose_nonterminal`'s `net_payoff > 0`. Governs the COMPRESSION class — `:add_rule`
(payoff = compression saving), `:remove_rule` (payoff = the dead rule's reclaimed cost,
`1 + expr_complexity(body)`), AND `:remove_feature` (#174; payoff = 1, a dead feature's reclaimed symbol).
The remaining generative-change ops (`:modify_threshold`, `:add_feature`) change the likelihood over
un-entertained programs (the escape-mass frontier), invisible here by construction, and are deferred to an
EU-priced exploration mechanism (master plan).
"""
net_voc(net_payoff_symbols::Real, compute_cost::Real) =
    net_value(complexity_logprior(-net_payoff_symbols; λ = log(2)), compute_cost)

# ═══════════════════════════════════════
# Grammar perturbation — deterministic argmax over net_voc (no rand; Invariant 1 canalised)
# ═══════════════════════════════════════

# A perturbation candidate — a kind-tagged, net_voc-ranked grammar edit. Replaces the former 5-tuple so the
# candidate is a FIRST-CLASS DECLARED type (Invariant 2/3): the structure Move 5's combined single-currency
# argmax extends by adding `kind`s (the enum), not by widening a positional tuple whose conventions live in
# slot order. `payload` is the typed edit target — a `ProductionRule` for `:add`/`:remove`, the feature
# `Symbol` for `:remove_feature`.
struct PerturbationCandidate
    voc::Float64                              # net_voc — the argmax key (primary order)
    is_remove::Bool                           # tie: prefer a removal (shrink) over an addition (grow)
    name::String                              # tie: lexicographically smaller (version-stable, not `hash`)
    kind::Symbol                              # :add | :remove | :remove_feature
    payload::Union{ProductionRule, Symbol}    # rule (:add/:remove) | feature (:remove_feature)
end

# Total order for the deterministic argmax: (1) higher `net_voc`; (2) on a tie, prefer a removal (hygiene —
# shrink before grow); (3) on a further tie, lexicographically smaller name. No `rand`, no `hash` (Julia's
# `hash` is not stable across versions — version-reproducible determinism is the point; a name string-compare
# is stable). A tie is between candidates of EXACTLY-EQUAL computed prior value, so a stable order is honest
# disambiguation among genuine argmax-optima, not a tiebreak of UNKNOWN values (Invariant 1). test_voc_gate.jl §5c.
function _candidate_better(a::PerturbationCandidate, b::PerturbationCandidate)
    a.voc != b.voc && return a.voc > b.voc
    a.is_remove != b.is_remove && return a.is_remove
    return a.name < b.name
end
_pick_better(best, cand) = (best === nothing || _candidate_better(cand, best)) ? cand : best

"""
    _best_compression_candidate(g, freq_table; compute_cost = 0.0) → Union{Nothing, PerturbationCandidate}

The compression-class `net_voc` argmax over `{:add_rule} ∪ {:remove_rule} ∪ {:remove_feature}` candidates, or
`nothing` if no candidate clears `net_voc > 0` (the saturation no-op). Returns the winning
`PerturbationCandidate` (see `_candidate_better` for the total order). The single home of the candidate
logic — `perturb_grammar` applies it, `compression_exhausted` (Move 2) tests it for `nothing` — so the
two can never disagree (DRY).
"""
function _best_compression_candidate(g::Grammar, freq_table::SubprogramFrequencyTable;
                                     compute_cost::Float64 = 0.0)::Union{Nothing, PerturbationCandidate}
    best = nothing  # the running argmax (a PerturbationCandidate); see _candidate_better

    add = _compression_payoff(freq_table)
    if !isnothing(add)
        rule, net_payoff = add
        if !(rule.name in Set(r.name for r in g.rules))         # idempotence guard (already-present name)
            v = net_voc(net_payoff, compute_cost)               # VOC gate (forward compute priced in)
            v > 0 && (best = _pick_better(best, PerturbationCandidate(v, false, string(rule.name), :add, rule)))
        end
    end
    for (rule, net_payoff) in _removal_payoff(g, freq_table)
        v = net_voc(net_payoff, compute_cost)
        v > 0 && (best = _pick_better(best, PerturbationCandidate(v, true, string(rule.name), :remove, rule)))
    end
    for feat in _feature_removal_payoff(g, freq_table)       # the symmetric MDL partner (#174): a dead
        v = net_voc(1, compute_cost)                         # feature reclaims exactly 1 symbol
        v > 0 && (best = _pick_better(best, PerturbationCandidate(v, true, string(feat), :remove_feature, feat)))
    end
    best
end

"""
    compression_exhausted(g, freq_table; compute_cost = 0.0) → Bool

The prior-side half of the exploration budget's saturation signal (Move 2): `true` iff no
compression-class meta-action improves the prior — i.e. `perturb_grammar` would be a no-op. Shares
`_best_compression_candidate` with `perturb_grammar` exactly, so "exhausted" ≡ "`perturb_grammar`
returns `g` unchanged" by construction. Prior-only, cheap (no belief, no lookahead). The belief-side
half (residual plateau) lives in `saturation.jl`.
"""
compression_exhausted(g::Grammar, freq_table::SubprogramFrequencyTable; compute_cost::Float64 = 0.0)::Bool =
    isnothing(_best_compression_candidate(g, freq_table; compute_cost = compute_cost))

"""
    perturb_grammar(g, freq_table, available_features; compute_cost = 0.0) → Grammar

Perturb a grammar by the compression-class meta-action whose `net_voc` is greatest, applied iff
`net_voc > 0`; otherwise a structural no-op — the **input grammar returned unchanged (same id)**, so the
downstream `add_programs_to_state!` dedup (keyed on `grammar.id`) re-adds nothing. The selection is a
**deterministic argmax** (the `rand`-based op choice — the Invariant-1 breach — is gone). `freq_table`
is REQUIRED (the type system enforces posterior analysis before nonterminal proposal).
`available_features` is retained for signature stability (the deferred feature-discovery mechanism will
consume it); it is not read here.

The compression class is the **MDL set** `{:add_rule} ∪ {:remove_rule} ∪ {:remove_feature}` candidates
(exploration-budget Move 1 + #174): `:add_rule` is `_compression_payoff`'s proposed rule (gated by the
idempotence guard); `:remove_rule` candidates are `_removal_payoff`'s dead rules; `:remove_feature`
candidates are `_feature_removal_payoff`'s dead features (each reclaiming 1 symbol) — all referenced by no
posterior-support program, prior-only, the symmetric MDL partners. The applied meta-action is the `net_voc`
argmax over the set, tiebroken by `_candidate_better`. When none improves the prior, the no-op **is** the
prior-saturation signal the exploration budget's saturation gate reads (Move 2). The remaining
generative-change ops (`:modify_threshold`, `:add_feature`) remain deferred — their value is invisible to
depth-one prior-only `net_voc` (the escape-mass frontier; `docs/exploration-budget/master-plan.md`).
"""
function perturb_grammar(g::Grammar, freq_table::SubprogramFrequencyTable,
                          available_features::Set{Symbol};
                          compute_cost::Float64 = 0.0)::Grammar
    # A structural no-op returns the INPUT grammar (same id) — not a fresh-id copy. Downstream,
    # `add_programs_to_state!` deduplicates by `grammar.id`, so a fresh-id no-op would defeat the dedup
    # and re-inject every program as a fresh Beta(1,1) duplicate (a silent posterior reset — A3). Same
    # id ⇒ a no-op truly changes nothing. (Tested by test_perturb_consumption.jl.)
    best = _best_compression_candidate(g, freq_table; compute_cost = compute_cost)
    best === nothing && return g                                # saturation no-op (the Move-2 signal)
    # Thread `g.thresholds` through (the 4-arg constructor) so a refined grid (exploration-budget Move 3)
    # survives a later compression on its lineage — the 3-arg form would re-default the grid and silently
    # discard the refinement. Bit-identical for default grammars (g.thresholds == default_thresholds by
    # value ⇒ identical enumeration), so the compression tests stay green.
    if best.kind === :add
        Grammar(g.feature_set, [g.rules; best.payload::ProductionRule], g.thresholds, next_grammar_id())
    elseif best.kind === :remove  # drop the dead rule by name (names are unique under the idempotence guard)
        Grammar(g.feature_set, [r for r in g.rules if r.name != (best.payload::ProductionRule).name],
                g.thresholds, next_grammar_id())
    else  # :remove_feature — reclaim a dead feature: drop it from feature_set AND its grid from thresholds.
        # The 4-arg Grammar recomputes complexity (grid-count-invariant), so |G| drops by exactly 1.
        feat = best.payload::Symbol
        new_features = Set(f for f in g.feature_set if f != feat)
        new_thresholds = Dict{Symbol, Vector{Float64}}(f => grid for (f, grid) in g.thresholds if f != feat)
        Grammar(new_features, g.rules, new_thresholds, next_grammar_id())
    end
end

# Backward-compatible 2-argument form (default feature set; forwards compute_cost)
function perturb_grammar(g::Grammar, freq_table::SubprogramFrequencyTable;
                          compute_cost::Float64 = 0.0)::Grammar
    perturb_grammar(g, freq_table, g.feature_set; compute_cost = compute_cost)
end

"""
    perturbation_voc(g, freq_table; compute_cost = 0.0) → Float64

The scalar net VOC of the best compression-class perturbation — the winning candidate's `net_voc`, or `0.0`
when none clears (the saturation no-op floor). The `expect`-side value of `perturb_grammar`'s edit: the
selection layer ranks the `:gw_perturb_grammar` meta-action by this scalar — the **prior-only surrogate**
fidelity of the one Δ log-evidence currency (Move 5), the cheap-screen peer of `exploration_voi`'s exact
lookahead. Shares `_best_compression_candidate` with `perturb_grammar` and `compression_exhausted`, so the
ranked value, the applied edit, and the saturation signal can never disagree (Invariant 3). At
`compute_cost = 0`, `perturbation_voc > 0` iff `!compression_exhausted`.
"""
function perturbation_voc(g::Grammar, freq_table::SubprogramFrequencyTable; compute_cost::Float64 = 0.0)::Float64
    best = _best_compression_candidate(g, freq_table; compute_cost = compute_cost)
    best === nothing ? 0.0 : best.voc
end

# ═══════════════════════════════════════
# AST structural equality (for subtree matching)
# ═══════════════════════════════════════

function expr_equal(a::GTExpr, b::GTExpr)
    num_equal(a.lhs, b.lhs) && a.threshold == b.threshold
end
function expr_equal(a::LTExpr, b::LTExpr)
    num_equal(a.lhs, b.lhs) && a.threshold == b.threshold
end
function expr_equal(a::AndExpr, b::AndExpr)
    expr_equal(a.left, b.left) && expr_equal(a.right, b.right)
end
function expr_equal(a::OrExpr, b::OrExpr)
    expr_equal(a.left, b.left) && expr_equal(a.right, b.right)
end
function expr_equal(a::NotExpr, b::NotExpr)
    expr_equal(a.child, b.child)
end
function expr_equal(a::NonterminalRef, b::NonterminalRef)
    a.name == b.name
end
function expr_equal(a::PersistsExpr, b::PersistsExpr)
    a.n == b.n && expr_equal(a.child, b.child)
end
function expr_equal(a::ChangedExpr, b::ChangedExpr)
    expr_equal(a.child, b.child)
end
function expr_equal(a::SinceExpr, b::SinceExpr)
    expr_equal(a.p, b.p) && expr_equal(a.q, b.q)
end
function expr_equal(a::ActionExpr, b::ActionExpr)
    a.action == b.action
end
function expr_equal(a::IfExpr, b::IfExpr)
    expr_equal(a.predicate, b.predicate) && expr_equal(a.then_branch, b.then_branch) && expr_equal(a.else_branch, b.else_branch)
end
function expr_equal(::ProgramExpr, ::ProgramExpr) false end

# Threshold-node collection/replacement (`collect_threshold_nodes`, `replace_threshold`) was the
# machinery of the retired `:modify_threshold` op. `:modify_threshold` is generative-change — it
# changes which programs the grammar generates (a likelihood effect over un-entertained programs),
# invisible to depth-one prior-only `net_voc` by construction — so collapse-towers Phase 5 deferred it
# to an EU-priced exploration budget (master plan, named successor) and deleted the dead machinery
# rather than leave it unreachable.

"""
    enumeration.jl — Complexity scoring and bottom-up program enumeration
"""

# ═══════════════════════════════════════
# Complexity scoring
# ═══════════════════════════════════════

# `THRESHOLDS` (the default per-feature grid seed) now lives in types.jl with `Grammar` — it is
# grammar-structural data, and enumeration reads each grammar's own `g.thresholds[feat]` below.

# Numeric-sublayer complexity: a bare feature read costs 1; each arithmetic combinator adds 1.
# expr_complexity(GTExpr) = num_complexity(lhs) keeps the bare-feature atom at cost 1 (the
# threshold- and comparison-free convention of Move 3 Q1(b) — a threshold adds no symbol, the
# comparison is bundled), so every pre-arithmetic program keeps an identical complexity,
# prior, and posterior (pinned by test/fixtures/feature_arithmetic_lift_v1.tsv).
function num_complexity(e::FeatureRef) 1 end
function num_complexity(e::Times) 1 + num_complexity(e.left) + num_complexity(e.right) end
function num_complexity(e::Plus) 1 + num_complexity(e.left) + num_complexity(e.right) end
function num_complexity(e::Minus) 1 + num_complexity(e.left) + num_complexity(e.right) end
function num_complexity(e::Div) 1 + num_complexity(e.left) + num_complexity(e.right) end
function num_complexity(e::AQ) 1 + num_complexity(e.left) + num_complexity(e.right) end
function num_complexity(e::Neg) 1 + num_complexity(e.child) end

function expr_complexity(e::GTExpr) num_complexity(e.lhs) end
function expr_complexity(e::LTExpr) num_complexity(e.lhs) end
function expr_complexity(e::AndExpr) 1 + expr_complexity(e.left) + expr_complexity(e.right) end
function expr_complexity(e::OrExpr) 1 + expr_complexity(e.left) + expr_complexity(e.right) end
function expr_complexity(e::NotExpr) 1 + expr_complexity(e.child) end
function expr_complexity(e::NonterminalRef) 1 end  # nonterminal reference costs 1
function expr_complexity(e::PersistsExpr) 1 + expr_complexity(e.child) end
function expr_complexity(e::ChangedExpr) 1 + expr_complexity(e.child) end
function expr_complexity(e::SinceExpr) 1 + expr_complexity(e.p) + expr_complexity(e.q) end
function expr_complexity(e::ActionExpr) 1 end
function expr_complexity(e::IfExpr) 1 + expr_complexity(e.predicate) + expr_complexity(e.then_branch) + expr_complexity(e.else_branch) end

"""Expanded complexity: cost of expression with nonterminals expanded."""
function expanded_complexity(e::ProgramExpr, rules::Vector{ProductionRule})
    _expanded(e, rules)
end

function _expanded(e::GTExpr, _) num_complexity(e.lhs) end
function _expanded(e::LTExpr, _) num_complexity(e.lhs) end
function _expanded(e::AndExpr, r) 1 + _expanded(e.left, r) + _expanded(e.right, r) end
function _expanded(e::OrExpr, r) 1 + _expanded(e.left, r) + _expanded(e.right, r) end
function _expanded(e::NotExpr, r) 1 + _expanded(e.child, r) end
function _expanded(e::PersistsExpr, r) 1 + _expanded(e.child, r) end
function _expanded(e::ChangedExpr, r) 1 + _expanded(e.child, r) end
function _expanded(e::SinceExpr, r) 1 + _expanded(e.p, r) + _expanded(e.q, r) end
function _expanded(e::NonterminalRef, rules)
    idx = findfirst(r -> r.name == e.name, rules)
    idx === nothing && return 1  # undefined nonterminal
    _expanded(rules[idx].body, rules)
end
function _expanded(e::ActionExpr, _) 1 end
function _expanded(e::IfExpr, r) 1 + _expanded(e.predicate, r) + _expanded(e.then_branch, r) + _expanded(e.else_branch, r) end

# ═══════════════════════════════════════
# Program enumeration (bottom-up)
# ═══════════════════════════════════════

"""
    enumerate_programs(grammar, max_depth; ...) → Vector{Program}

Bottom-up enumeration of programs as expression trees that evaluate to actions.

Depth semantics (overall program tree depth, not just predicate depth):
- Depth 1: ActionExpr(a) — constant action programs
- Depth 2: IfExpr(depth-1-pred, a1, a2) — single predicates with branching
- Depth 3: IfExpr(depth-2-pred, a1, a2) — compound predicates with branching,
           plus nested IfExpr in branches

Predicate depth = program depth - 1. To get AND/OR/NOT predicates (old depth 2),
use max_depth=3.
"""
# The enumeration's default arithmetic combinator roster (design §5 Q2): AQ is the default
# quotient (total, smooth, artifact-free); protected Div is constructible/compilable/priced
# but enters enumeration only when a caller passes it in `num_ops` (poles opt-in, the x/0
# artifact documented at compile_num). All entries must be decision-free combinators
# (precedent decision-free-combinator).
const NUM_OPS_DEFAULT = (Times, Plus, Minus, AQ, Neg)

"""
    _num_exprs_by_depth(g, max_num_depth, num_ops) → Vector{Vector{NumExpr}}

The depth-bounded numeric-expression set. Depth 1 is `FeatureRef` per grammar feature
(sorted — the behaviour-preserving floor: `max_num_depth = 1` yields exactly the lifted
image of the pre-arithmetic atoms). Depth d ≥ 2: the combinators over lower-depth
expressions where at least one argument sits at depth d−1. Commutative ops (Times, Plus)
enumerate unordered pairs; non-commutative ops (Minus, Div, AQ) enumerate ordered pairs
excluding identical arguments (x−x, x/x are constants — degenerate predicates). Order is
deterministic throughout.
"""
function _num_exprs_by_depth(g::Grammar, max_num_depth::Int, num_ops)::Vector{Vector{NumExpr}}
    by_depth = Vector{NumExpr}[]
    push!(by_depth, NumExpr[FeatureRef(f) for f in sort(collect(g.feature_set))])
    for d in 2:max_num_depth
        lower = NumExpr[]
        for dd in 1:(d - 1)
            append!(lower, by_depth[dd])
        end
        depth_of = Dict{Int, Int}()
        i = 0
        for dd in 1:(d - 1), _ in by_depth[dd]
            i += 1
            depth_of[i] = dd
        end
        fresh = NumExpr[]
        for op in num_ops
            if op === Neg
                for a in by_depth[d - 1]
                    push!(fresh, Neg(a))
                end
            elseif op === Times || op === Plus
                for i in eachindex(lower), j in i:length(lower)
                    max(depth_of[i], depth_of[j]) == d - 1 || continue
                    push!(fresh, op(lower[i], lower[j]))
                end
            else  # Minus, Div, AQ — ordered, no identical arguments
                for i in eachindex(lower), j in eachindex(lower)
                    i == j && continue
                    max(depth_of[i], depth_of[j]) == d - 1 || continue
                    push!(fresh, op(lower[i], lower[j]))
                end
            end
        end
        push!(by_depth, fresh)
    end
    by_depth
end

"""Threshold grid for a numeric expression: a bare feature read uses the grammar's
per-feature grid (Move 3's refinement target); a compound expression uses the default seed
grid until the explore-path generalisation attaches per-NumExpr observed-value grids
(design §5 Q4 — the floor choice; the grid is not charged, fineness-Occam rides the
marginal likelihood)."""
_num_threshold_grid(g::Grammar, e::FeatureRef) = g.thresholds[e.feature]
_num_threshold_grid(::Grammar, ::NumExpr) = THRESHOLDS

function enumerate_programs(g::Grammar, max_depth::Int;
                            include_temporal::Bool=false,
                            min_log_prior::Float64=-20.0,
                            action_space::Vector{Symbol}=Symbol[:classify],
                            max_num_depth::Int=1,
                            num_ops=NUM_OPS_DEFAULT)::Vector{Program}
    max_complexity = -min_log_prior - g.complexity  # early pruning threshold

    # ── Phase 1: enumerate predicate expressions by depth ──
    # Predicates are Boolean-valued (GT, LT, AND, OR, NOT, temporal, nonterminal).
    # The comparison atoms threshold a NumExpr; max_num_depth = 1 (the default) restricts
    # the numeric sublayer to bare FeatureRefs — the behaviour-preserving floor. Raising it
    # is a meta-decision, not a host setting (the escalate-arithmetic-depth meta-action is
    # the named successor); the kwarg is the enumeration lever that meta-action drives.

    atoms = ProgramExpr[]
    for (d, nums) in enumerate(_num_exprs_by_depth(g, max_num_depth, num_ops))
        for nexpr in nums
            for t in _num_threshold_grid(g, nexpr)
                push!(atoms, GTExpr(nexpr, t))
                push!(atoms, LTExpr(nexpr, t))
            end
        end
    end
    for r in g.rules
        push!(atoms, NonterminalRef(r.name))
    end

    pred_by_depth = Vector{ProgramExpr}[]
    push!(pred_by_depth, atoms)

    # Build higher-depth predicates up to max_depth-1 (program depth = pred depth + 1)
    max_pred_depth = max_depth - 1
    for d in 2:max_pred_depth
        new_preds = ProgramExpr[]

        # AND, OR: pair depth d-1 with atoms
        for a in pred_by_depth[d-1]
            for b in atoms
                push!(new_preds, AndExpr(a, b))
                push!(new_preds, OrExpr(a, b))
            end
        end

        # NOT of depth d-1
        for a in pred_by_depth[d-1]
            push!(new_preds, NotExpr(a))
        end

        # Temporal operators (only if enabled)
        if include_temporal
            for a in pred_by_depth[d-1]
                push!(new_preds, ChangedExpr(a))
                for n in [2, 3, 5]
                    push!(new_preds, PersistsExpr(a, n))
                end
            end
        end

        push!(pred_by_depth, new_preds)
    end

    # ── Phase 2: build programs from predicates + action_space ──

    programs = Program[]

    # Depth 1: constant action programs (ActionExpr)
    for a in action_space
        c = 1  # expr_complexity(ActionExpr) = 1
        c <= max_complexity || continue
        push!(programs, Program(ActionExpr(a), c, g.id))
    end

    # Depth 2..max_depth: IfExpr with depth-(d-1) predicates and flat action branches
    for pred_depth in 1:max_pred_depth
        pred_depth <= length(pred_by_depth) || continue
        for pred in pred_by_depth[pred_depth]
            pred_c = expr_complexity(pred)
            # Early pruning: IfExpr adds 1 + pred_c + 1 + 1 = 3 + pred_c minimum
            3 + pred_c > max_complexity && continue
            for a1 in action_space, a2 in action_space
                a1 == a2 && continue  # skip tautologies: (if pred a a) ≡ (a)
                expr = IfExpr(pred, ActionExpr(a1), ActionExpr(a2))
                c = 3 + pred_c  # 1 (if) + pred_c + 1 (a1) + 1 (a2)
                push!(programs, Program(expr, c, g.id))
            end
        end
    end

    programs
end

"""
    enumerate_programs_as_measure(grammar, max_depth; ...) → EnumerationMeasure{Program}

Typed-carrier wrapper around `enumerate_programs`. Returns an
`EnumerationMeasure{Program}` whose `carrier` field is the program
vector, `prevision` is a `CategoricalPrevision` with log-weights from
the complexity prior `-grammar.complexity * log(2) - p.complexity *
log(2)` (matching the convention in `add_programs_to_state!`), and
`space` is `Finite(programs)`.

Stratum-2 tolerance per `precedents.md` §4: `==` under deterministic
iteration order. Enumeration order is grammar-fixed (sorted feature
set, fixed threshold list, deterministic depth-wise expansion),
so the returned `EnumerationMeasure` is reproducible bit-for-bit
across invocations with the same grammar / max_depth / action_space.
"""
function enumerate_programs_as_measure(g::Grammar, max_depth::Int;
                                       include_temporal::Bool=false,
                                       min_log_prior::Float64=-20.0,
                                       action_space::Vector{Symbol}=Symbol[:classify],
                                       max_num_depth::Int=1,
                                       num_ops=NUM_OPS_DEFAULT)
    programs = enumerate_programs(g, max_depth;
                                   include_temporal=include_temporal,
                                   min_log_prior=min_log_prior,
                                   action_space=action_space,
                                   max_num_depth=max_num_depth,
                                   num_ops=num_ops)
    # Program node-count prior: the SPEC §1.3 complexity log-prior (`complexity.jl`) as the
    # two-part MDL code — `g.complexity` defines the dictionary, `p.complexity` describes the
    # program given it (each nonterminal ref costs 1). λ = log(2), pinned by §1.3. The two-call
    # sum is bit-identical to `-g.complexity*log(2) - p.complexity*log(2)` (test_complexity.jl).
    log_weights = Float64[complexity_logprior(g.complexity; λ = log(2)) +
                          complexity_logprior(p.complexity; λ = log(2)) for p in programs]
    Ontology.EnumerationMeasure{Program}(CategoricalPrevision(log_weights), programs, Finite(programs))
end

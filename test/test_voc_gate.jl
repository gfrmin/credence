# test_voc_gate.jl — the Value-of-Computation gate on perturb_grammar (collapse-towers Phase 5).
#
# net_voc is the structural twin of net_voi, in LOG-PRIOR currency (R1): perturb_grammar sees only
# (g, freq_table, available_features), so the affordable depth-one value-proxy is the change in the
# program-space complexity prior — net_value(Δcomplexity_logprior, compute_cost). It governs the
# COMPRESSION class only (R2): :add_rule, whose value is exactly propose_nonterminal's net_payoff
# scaled to nats by λ=log(2). The generative-change ops (modify_threshold, add/remove_feature) are
# excluded and deferred — they change the likelihood over un-entertained programs (escape-mass), which
# depth-one prior-only VOC cannot see. The rand-based selection is GONE: which-perturbation is now a
# deterministic argmax of net_voc.
#
# Asserts: (0) net_voc's arithmetic; (1) degenerate reduction — compute_cost=0 reproduces today's
# :add_rule branch bit-for-bit; (2) directional — compute_cost > net_payoff·log(2) suppresses to a
# no-op, the flip relative to the oracle, no magic number; (3) determinism — two runs on identical
# inputs give the same grammar; (4) no `rand(` survives in the perturbation source (the breach is
# closed at the source level, not just behaviourally).
#
# Run from repo root:
#     julia test/test_voc_gate.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, SubprogramFrequencyTable, Program,
                GTExpr, LTExpr, AndExpr, NotExpr, IfExpr, ActionExpr, NonterminalRef, ProgramExpr,
                perturb_grammar, propose_nonterminal, analyse_posterior_subtrees,
                expr_complexity, show_expr, net_voc

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

# Canonical structural readout of a grammar's rules (id-independent, name+body).
rules_str(g::Grammar) = sort([string(r.name) * "=" * show_expr(r.body) for r in g.rules])

# A freq_table with a known compressing subtree s = And(GT(:red,0.7), LT(:green,0.3)) (expr_complexity
# 3) shared by `n` distinct programs ⇒ net_payoff = n·(3−1) − (1+3) = 2n − 4.
function shared_subtree_table(n::Int)
    s = AndExpr(GTExpr(:red, 0.7), LTExpr(:green, 0.3))
    progs = Program[IfExpr(s, ActionExpr(:a), ActionExpr(:b)) |> e -> Program(e, 6, 1) for _ in 1:n]
    w = fill(1.0 / n, n)
    (analyse_posterior_subtrees(progs, w; min_frequency = 0.0, min_complexity = 2), s)
end

# The independent oracle for net_payoff (symbols) of a freq_table's best subtree.
function oracle_net_payoff(ft::SubprogramFrequencyTable)   # credence-lint: allow — precedent:test-oracle — independent net_payoff for the gate
    best = argmax(ft.weighted_frequency)
    expr_c = expr_complexity(ft.subtrees[best])
    n_sources = length(ft.source_programs[best])
    n_sources * (expr_c - 1) - (1 + expr_c)
end

println("="^64)
println("net_voc — the VOC gate on perturb_grammar (Phase 5)")
println("="^64)

# ── (0) net_voc arithmetic: net_value(Δcomplexity_logprior, compute_cost), Δ = net_payoff·log(2) ──
check("net_voc(4, 0.0) == 4·log(2)", net_voc(4, 0.0) ≈ 4 * log(2), "got $(net_voc(4,0.0))")
check("net_voc(4, 1.0) == 4·log(2) − 1", net_voc(4, 1.0) ≈ 4 * log(2) - 1.0, "got $(net_voc(4,1.0))")
check("net_voc(0, 0.0) == 0", net_voc(0, 0.0) ≈ 0.0, "got $(net_voc(0,0.0))")
check("net_voc folds compute_cost linearly (no clamp)", net_voc(2, 100.0) ≈ 2 * log(2) - 100.0)

# ── (1) degenerate reduction: compute_cost = 0 ≡ today's deterministic :add_rule branch ──
let
    g = Grammar(Set([:red, :green]), ProductionRule[], 1)
    ft, s = shared_subtree_table(3)                       # net_payoff = 2·3 − 4 = 2 > 0; oracle subtree s
    proposed = propose_nonterminal(ft)
    @assert proposed !== nothing
    expected = Grammar(g.feature_set, [g.rules; proposed], 999)   # the :add_rule branch's construction
    got = perturb_grammar(g, ft; compute_cost = 0.0)
    check("compute_cost=0 reproduces :add_rule branch (rules)", rules_str(got) == rules_str(expected),
          "got=$(rules_str(got)) expected=$(rules_str(expected))")
    check("compute_cost=0 preserves feature_set", got.feature_set == g.feature_set)
    # Pin the SELECTED body to the independent oracle s (not just a count / run-equality): a
    # mis-selection of a different subtree would pass the count checks but fail this.
    check("the added rule's body == the oracle subtree s",
          length(got.rules) == 1 && show_expr(got.rules[end].body) == show_expr(s),
          "added=$(rules_str(got)) oracle=$(show_expr(s))")
end

# ── (1b) no compressing subtree (net_payoff ≤ 0) ⇒ deterministic no-op ──
let
    g = Grammar(Set([:red, :green]), ProductionRule[], 1)
    empty_ft = SubprogramFrequencyTable(ProgramExpr[], Float64[], Vector{Int}[])
    got = perturb_grammar(g, empty_ft; compute_cost = 0.0)
    check("empty table ⇒ no-op (no rule added)", isempty(got.rules), "rules=$(rules_str(got))")

    ft1, _ = shared_subtree_table(2)                      # net_payoff = 2·2 − 4 = 0, not > 0
    got1 = perturb_grammar(g, ft1; compute_cost = 0.0)
    check("net_payoff = 0 ⇒ no-op (gate is strict >)", isempty(got1.rules), "rules=$(rules_str(got1))")
end

# ── (1c) name-collision no-op (path b): a proposed rule whose name already exists ⇒ no-op, and the
# existing body is never overwritten. This is the idempotence guard that keeps the :add_rule path
# monotonic — re-perturbing a grammar that already holds the compression does not re-add or clobber it.
# Move-1 note: the colliding rule must be REFERENCED by the support, else it is a dead rule and
# :remove_rule (correctly) removes it — a different code path. Adding one program that references the
# nonterminal makes it live, isolating the add-side idempotence guard (this test's sole purpose). ──
let
    s = AndExpr(GTExpr(:red, 0.7), LTExpr(:green, 0.3))   # the compressing subtree (expr_complexity 3)
    collide = Symbol("NT_", hash(show_expr(s)))           # the exact (full-hash) name propose_nonterminal generates for s
    existing_body = GTExpr(:red, 0.9)                     # a DIFFERENT body under that same name
    # Four programs share s (⇒ :add_rule proposes a rule named `collide`, net_payoff 4 > 0), plus one
    # program that REFERENCES `collide` (⇒ it is live, not a :remove_rule candidate).
    progs = Program[Program(IfExpr(s, ActionExpr(:a), ActionExpr(:b)), 6, 1) for _ in 1:4]
    push!(progs, Program(IfExpr(NonterminalRef(collide), ActionExpr(:a), ActionExpr(:b)), 3, 1))
    w = fill(1.0 / 5, 5)
    ft = analyse_posterior_subtrees(progs, w; min_frequency = 0.0, min_complexity = 2)
    g = Grammar(Set([:red, :green]), [ProductionRule(collide, existing_body)], 1)
    got = perturb_grammar(g, ft; compute_cost = 0.0)
    check("name collision ⇒ no-op (no second rule added)", length(got.rules) == 1, "rules=$(rules_str(got))")
    check("name collision never overwrites the existing body",
          show_expr(got.rules[1].body) == show_expr(existing_body), "body=$(show_expr(got.rules[1].body))")
end

# ── (2) directional: compute_cost flips the gate exactly at net_payoff·log(2) ──
let
    g = Grammar(Set([:red, :green]), ProductionRule[], 1)
    ft, _ = shared_subtree_table(4)                       # net_payoff = 2·4 − 4 = 4 > 0
    payoff = oracle_net_payoff(ft)                        # credence-lint: allow — precedent:test-oracle — the gate flips at payoff·log(2)
    threshold = payoff * log(2)
    below = perturb_grammar(g, ft; compute_cost = threshold - 1e-6)
    above = perturb_grammar(g, ft; compute_cost = threshold + 1e-6)
    check("compute_cost just below payoff·log(2) ⇒ rule added", length(below.rules) == 1,
          "rules=$(rules_str(below))")
    check("compute_cost just above payoff·log(2) ⇒ no-op (forward compute priced out)",
          isempty(above.rules), "rules=$(rules_str(above))")
end

# ── (3) determinism: two runs on identical (g, ft) ⇒ structurally identical grammar (no rand) ──
let
    g = Grammar(Set([:red, :green]), ProductionRule[], 1)
    ft, _ = shared_subtree_table(3)
    a = perturb_grammar(g, ft)
    b = perturb_grammar(g, ft)
    check("two runs ⇒ identical rules", rules_str(a) == rules_str(b), "a=$(rules_str(a)) b=$(rules_str(b))")
    check("two runs ⇒ identical feature_set", a.feature_set == b.feature_set)
end

# ── (4) source-level guard: the breach is closed — NO rng entry point on the perturbation path.
# Widened beyond a bare `rand(` substring: (a) the regex also catches randn/randperm/randsubseq/
# shuffle/rand!; (b) line comments are stripped so a docstring mention can't false-trip it; (c) it
# also scans agent_state.jl, where `next_grammar_id` lives — a random id THERE would evade the
# determinism tests, which compare structure and deliberately exclude the grammar id. ──
let
    rng = r"\b(rand|randn|randperm|randsubseq|shuffle|rand!)\s*\("
    for f in ("program_space/perturbation.jl", "program_space/agent_state.jl")
        src = read(joinpath(@__DIR__, "..", "src", f), String)
        code = join((first(split(line, '#')) for line in split(src, '\n')), '\n')  # strip line comments
        check("$f: no rng entry point on the perturbation path", !occursin(rng, code),
              "an rng call survives in $f")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# (5) :remove_rule — dictionary hygiene, the symmetric MDL partner of :add_rule
#     (exploration-budget Move 1). A rule no posterior-support program references is dead weight:
#     removing it raises the complexity prior at ZERO fit cost (no support program changes), so it is
#     prior-only and net_voc-rankable. "Referenced" is counted by a FULL-depth AST walk over the
#     support (w > 1e-15), INDEPENDENT of extract_subtrees' min_complexity filter (the Scope-B fix).
# ═══════════════════════════════════════════════════════════════════════════════

# Independent oracle for "the support references `name`": the rule-name token appears in some support
# program's show_expr. Deliberately NOT the production collect_nonterminal_refs — this oracle pins the
# prior-only claim (belief untouched) without circular reference to the function under test.
references_name(progs, name) =                  # credence-lint: allow — precedent:test-oracle — independent reference check for the prior-only pin
    any(p -> occursin(string(name), show_expr(p.expr)), progs)

# ── (5a) remove a dead rule; keep a live one; prior-only (belief untouched) pin ──
let
    # :LIVE is referenced by the support (a bare NonterminalRef predicate); :DEAD by nobody.
    g = Grammar(Set([:red, :blue]),
                [ProductionRule(:LIVE, GTExpr(:red, 0.7)), ProductionRule(:DEAD, GTExpr(:blue, 0.5))], 1)
    progs = Program[Program(IfExpr(NonterminalRef(:LIVE), ActionExpr(:a), ActionExpr(:b)), 3, 1) for _ in 1:3]
    w = fill(1.0 / 3, 3)
    ft = analyse_posterior_subtrees(progs, w; min_frequency = 0.0, min_complexity = 2)

    got = perturb_grammar(g, ft; compute_cost = 0.0)
    names = Set(r.name for r in got.rules)
    check("dead rule :DEAD is removed", :DEAD ∉ names, "rules=$(rules_str(got))")
    check("live rule :LIVE is kept", :LIVE ∈ names, "rules=$(rules_str(got))")
    check("exactly one rule removed", length(got.rules) == 1, "rules=$(rules_str(got))")

    # The reference pass populated the new field: :LIVE referenced (depth-1), :DEAD not.
    check("reference pass: referenced == {:LIVE}", ft.referenced_nonterminals == Set([:LIVE]),
          "got $(ft.referenced_nonterminals)")
    # net_voc of the removal == log(2)·(1 + expr_complexity(body)) == log(2)·2.
    check("removal net_voc == log(2)·(1 + complexity(body))",
          net_voc(1 + expr_complexity(GTExpr(:blue, 0.5)), 0.0) ≈ 2 * log(2))

    # Prior-only / belief-untouched pin (independent oracle): :DEAD is referenced by ZERO support
    # programs, so the support set — hence the posterior on re-conditioning the same data — is
    # bit-identical across the removal; :LIVE remains referenced.
    check("prior-only: removed :DEAD referenced by no support program (belief untouched)",
          !references_name(progs, :DEAD), "a support program referenced :DEAD")
    check("prior-only: surviving :LIVE still referenced by the support", references_name(progs, :LIVE))
end

# ── (5b) depth-1 reference is seen (R1) — the Scope-B unsoundness guard ──
let
    # :BARE is referenced ONLY by a bare NonterminalRef (a complexity-1 predicate) — exactly the
    # depth-1 reference the old extract_subtrees(min_complexity=2) count dropped. :TDEAD: referenced
    # nowhere. A count routed through extract_subtrees would drop the bare ref and wrongly remove :BARE.
    g = Grammar(Set([:green, :blue]),
                [ProductionRule(:BARE, GTExpr(:green, 0.2)), ProductionRule(:TDEAD, GTExpr(:blue, 0.9))], 1)
    progs = Program[Program(IfExpr(NonterminalRef(:BARE), ActionExpr(:a), ActionExpr(:b)), 3, 1) for _ in 1:2]
    w = fill(0.5, 2)
    ft = analyse_posterior_subtrees(progs, w; min_frequency = 0.0, min_complexity = 2)

    check("depth-1 reference is collected (:BARE ∈ referenced)", :BARE ∈ ft.referenced_nonterminals,
          "got $(ft.referenced_nonterminals)")
    got = perturb_grammar(g, ft; compute_cost = 0.0)
    names = Set(r.name for r in got.rules)
    check("depth-1-referenced :BARE is NOT removed (sound full-depth count)", :BARE ∈ names,
          "rules=$(rules_str(got))")
    check("truly-dead :TDEAD is removed", :TDEAD ∉ names, "rules=$(rules_str(got))")
end

# ── (5c) determinism + tiebreak: add ≡ remove net_voc tie resolves remove-first ──
let
    # Exact tie: an :add_rule candidate of payoff 2 (subtree s shared by 3 programs, expr_complexity 3
    # ⇒ 3·(3−1) − (1+3) = 2) AND a :remove_rule candidate of payoff 2 (dead :D, body complexity 1 ⇒
    # 1 + 1 = 2). Tiebreak = remove-first (hygiene: shrink before grow): :D removed, s NOT added.
    s = AndExpr(GTExpr(:red, 0.7), LTExpr(:green, 0.3))
    progs = Program[Program(IfExpr(s, ActionExpr(:a), ActionExpr(:b)), 6, 1) for _ in 1:3]   # share s; no ref to :D
    w = fill(1.0 / 3, 3)
    ft = analyse_posterior_subtrees(progs, w; min_frequency = 0.0, min_complexity = 2)
    g = Grammar(Set([:red, :green]), [ProductionRule(:D, GTExpr(:blue, 0.5))], 1)

    add_payoff = oracle_net_payoff(ft)          # credence-lint: allow — precedent:test-oracle — confirm the exact tie
    check("tie precondition: add payoff == remove payoff == 2",
          add_payoff == 2 && (1 + expr_complexity(GTExpr(:blue, 0.5))) == 2, "add_payoff=$add_payoff")

    got = perturb_grammar(g, ft; compute_cost = 0.0)
    check("net_voc tie resolves remove-first: :D removed", :D ∉ Set(r.name for r in got.rules),
          "rules=$(rules_str(got))")
    check("net_voc tie resolves remove-first: subtree NOT added (remove won)", isempty(got.rules),
          "rules=$(rules_str(got))")
    a = perturb_grammar(g, ft; compute_cost = 0.0)
    b = perturb_grammar(g, ft; compute_cost = 0.0)
    check("tiebreak is deterministic across runs", rules_str(a) == rules_str(b),
          "a=$(rules_str(a)) b=$(rules_str(b))")
end

# ── (5d) saturation no-op — the Move-2 prior-saturation signal ──
let
    # Every rule referenced AND no compressing subtree ⇒ no add, no remove ⇒ structural no-op: the
    # INPUT grammar returned unchanged (same id). This no-op is the prior-saturation signal Move 2 reads.
    g = Grammar(Set([:red]), [ProductionRule(:ONLY, GTExpr(:red, 0.4))], 7)
    progs = Program[Program(IfExpr(NonterminalRef(:ONLY), ActionExpr(:a), ActionExpr(:b)), 3, 7) for _ in 1:2]
    w = fill(0.5, 2)
    ft = analyse_posterior_subtrees(progs, w; min_frequency = 0.0, min_complexity = 2)
    got = perturb_grammar(g, ft; compute_cost = 0.0)
    check("saturation no-op: same grammar id (Move-2 signal)", got.id == g.id, "id=$(got.id)")
    check("saturation no-op: rules unchanged", Set(r.name for r in got.rules) == Set([:ONLY]),
          "rules=$(rules_str(got))")
end

println("="^64)
println("ALL CHECKS PASSED — net_voc / VOC gate")
println("="^64)

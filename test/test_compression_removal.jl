# test_compression_removal.jl — the compression-class removal cluster (#174).
#
# PR 1 (this file): the :remove_rule TRANSITIVE soundness fix. Enumerated programs hold NonterminalRefs
# UNEXPANDED (a rule's body lives in grammar.rules, resolved at compile time), so a rule referenced ONLY
# inside another rule's body is invisible to analyse_posterior_subtrees' program-AST walk →
# _removal_payoff flags it dead → perturb_grammar removes it → compile_expr hits the now-dangling ref and
# crashes ("Undefined nonterminal"). The fix unions the program-direct referenced_nonterminals with
# collect_nonterminal_refs! over ALL rule bodies (Q3). The fail-closed `nothing` default does NOT catch
# this — that default guards the *un-analysed* table; this is *analysed-but-incomplete* (the set is
# populated, it just under-reports).
#
# Sections:
#   §1  the transitive repro: NT_B referenced only inside NT_A's body — not a removal candidate; the
#       support program compiles (the averted crash).
#   §2  no over-protection: a rule dead AND unreferenced by any live body is still removable.
#   §3  dead-chain reclaim cadence (Q3 union-ALL ⇒ one link per pass).
#   §4  leaf-body sentinel: the fix is a no-op for non-nested grammars (the §3-of-doc capture claim).
#
# PR 3 will extend this file with :remove_feature (collect_feature_refs!, _feature_removal_payoff, the
# unified PerturbationCandidate argmax).
#
# Run: julia test/test_compression_removal.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, Program,
                GTExpr, AndExpr, IfExpr, ActionExpr, NonterminalRef,
                perturb_grammar, analyse_posterior_subtrees, compile_kernel, CompiledKernel

function check(name, cond, detail = "")
    cond ? println("PASSED: $name") : (println("FAILED: $name — $detail"); error("fail: $name"))
end

# Names flagged removable by _removal_payoff (the function under fix). Qualified, not imported.
removal_names(g, ft) = Set(r.name for (r, _) in Credence._removal_payoff(g, ft))
rule_names(g) = Set(r.name for r in g.rules)

println("="^64)
println("compression-removal — :remove_rule transitive soundness (#174 PR 1)")
println("="^64)

# ── §1  the transitive repro: NT_B referenced ONLY inside NT_A's body ──
#
# g holds NT_B = (gt :x 0.3) and NT_A = AND(ref(:NT_B), (gt :y 0.5)). One support program references
# :NT_A directly; :NT_B only transitively (through NT_A's body). analyse_posterior_subtrees walks the
# PROGRAM AST, which never enters NT_A's body, so referenced_nonterminals = {:NT_A}. Pre-fix
# _removal_payoff flags :NT_B dead and removes it, leaving NT_A's body with a dangling ref → crash.
let
    nt_b = ProductionRule(:NT_B, GTExpr(:x, 0.3))
    nt_a = ProductionRule(:NT_A, AndExpr(NonterminalRef(:NT_B), GTExpr(:y, 0.5)))
    g = Grammar(Set([:x, :y]), [nt_b, nt_a], 1)
    prog = Program(IfExpr(NonterminalRef(:NT_A), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    ft = analyse_posterior_subtrees([prog], [1.0]; min_frequency = 0.0, min_complexity = 2)

    # The root of the bug: the program walk sees :NT_A only — :NT_B is invisible to it.
    check("§1 program walk: referenced_nonterminals == {:NT_A} (NT_B invisible to it)",
          ft.referenced_nonterminals == Set([:NT_A]), "got $(ft.referenced_nonterminals)")

    # THE FIX (root cause): _removal_payoff unions rule-body refs, so :NT_B is NOT a candidate (NT_A's
    # body references it) and :NT_A is not either (the program references it). No removal candidates.
    check("§1 _removal_payoff flags neither rule (NT_B protected by NT_A's body)",
          removal_names(g, ft) == Set{Symbol}(), "got removal candidates $(removal_names(g, ft))")

    # Behavioural: perturb_grammar keeps :NT_B (a saturation no-op — nothing to compress, nothing dead).
    got = perturb_grammar(g, ft; compute_cost = 0.0)
    check("§1 perturb_grammar keeps :NT_B (transitive ref honoured)",
          :NT_B in rule_names(got), "rules=$(sort(string.(collect(rule_names(got)))))")

    # End-to-end: the support program compiles against the perturbed grammar with NO dangling ref.
    # Pre-fix (NT_B removed) compile_kernel throws error("Undefined nonterminal: NT_B").
    check("§1 the support program compiles — no dangling NonterminalRef (the averted crash)",
          compile_kernel(prog, got, 1) isa CompiledKernel, "compile_kernel did not return a CompiledKernel")
end

# ── §2  no over-protection: a rule dead AND unreferenced by any live body is still removable ──
#
# NT_LEAF referenced only by NT_MID's body; NT_MID referenced by the program; NT_ORPHAN referenced by
# NOBODY. The fix keeps NT_LEAF (NT_MID's body protects it) but NT_ORPHAN stays dead and removable.
let
    g = Grammar(Set([:x, :y, :z]),
                [ProductionRule(:NT_LEAF, GTExpr(:x, 0.3)),
                 ProductionRule(:NT_MID, AndExpr(NonterminalRef(:NT_LEAF), GTExpr(:y, 0.5))),
                 ProductionRule(:NT_ORPHAN, GTExpr(:z, 0.1))], 1)
    prog = Program(IfExpr(NonterminalRef(:NT_MID), ActionExpr(:a), ActionExpr(:b)), 3, 1)
    ft = analyse_posterior_subtrees([prog], [1.0]; min_frequency = 0.0, min_complexity = 2)

    check("§2 only :NT_ORPHAN is a removal candidate (dead, unreferenced anywhere)",
          removal_names(g, ft) == Set([:NT_ORPHAN]), "got $(removal_names(g, ft))")
    got = perturb_grammar(g, ft; compute_cost = 0.0)
    check("§2 :NT_ORPHAN removed", :NT_ORPHAN ∉ rule_names(got), "rules=$(sort(string.(collect(rule_names(got)))))")
    check("§2 :NT_LEAF and :NT_MID kept (live chain)", :NT_LEAF in rule_names(got) && :NT_MID in rule_names(got),
          "rules=$(sort(string.(collect(rule_names(got)))))")
end

# ── §3  dead-chain reclaim cadence (Q3 union-ALL ⇒ one link per pass) ──
#
# A dead chain NT_X → NT_Y (NT_X's body references NT_Y; the program references neither). Union-ALL keeps
# NT_Y alive while dead NT_X's body names it: pass 1 removes NT_X (referenced by nothing), pass 2 reclaims
# the orphaned NT_Y. Sound, and matched to one-perturbation-per-call. (referenced_nonterminals depends on
# the programs only, not the grammar, so the same ft re-scores each pass.)
let
    g0 = Grammar(Set([:x, :y]),
                 [ProductionRule(:NT_Y, GTExpr(:x, 0.3)),
                  ProductionRule(:NT_X, AndExpr(NonterminalRef(:NT_Y), GTExpr(:y, 0.5)))], 1)
    prog = Program(IfExpr(GTExpr(:x, 0.2), ActionExpr(:a), ActionExpr(:b)), 3, 1)  # references no nonterminal
    ft = analyse_posterior_subtrees([prog], [1.0]; min_frequency = 0.0, min_complexity = 2)

    check("§3 referenced_nonterminals == {} (program references no rule)",
          ft.referenced_nonterminals == Set{Symbol}(), "got $(ft.referenced_nonterminals)")
    check("§3 pass 1: only :NT_X removable (NT_Y protected by NT_X's body)",
          removal_names(g0, ft) == Set([:NT_X]), "got $(removal_names(g0, ft))")

    g1 = perturb_grammar(g0, ft; compute_cost = 0.0)   # removes NT_X (the one candidate)
    check("§3 pass 1 removed :NT_X", :NT_X ∉ rule_names(g1), "rules=$(sort(string.(collect(rule_names(g1)))))")
    check("§3 pass 2: :NT_Y now removable (orphaned by NT_X's removal)",
          removal_names(g1, ft) == Set([:NT_Y]), "got $(removal_names(g1, ft))")
end

# ── §4  leaf-body sentinel: the fix is a no-op for non-nested grammars ──
#
# Mirrors test_voc_gate §5a: leaf bodies, no NonterminalRef in any rule body, so the rule-body union adds
# nothing (collect_nonterminal_refs! over a leaf body is empty). A dead leaf rule is removable exactly as
# before. The existing suites (test_voc_gate, test_saturation, test_program_space, test_threshold_explore)
# are the full capture pins; this is the in-file sentinel.
let
    g = Grammar(Set([:red, :blue]),
                [ProductionRule(:LIVE, GTExpr(:red, 0.7)), ProductionRule(:DEAD, GTExpr(:blue, 0.5))], 1)
    progs = Program[Program(IfExpr(NonterminalRef(:LIVE), ActionExpr(:a), ActionExpr(:b)), 3, 1) for _ in 1:3]
    ft = analyse_posterior_subtrees(progs, fill(1 / 3, 3); min_frequency = 0.0, min_complexity = 2)
    check("§4 leaf-body grammar: only :DEAD removable (fix is a no-op here)",
          removal_names(g, ft) == Set([:DEAD]), "got $(removal_names(g, ft))")
end

println("="^64)
println("ALL CHECKS PASSED — compression-removal :remove_rule transitive soundness")
println("="^64)

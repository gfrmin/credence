"""
    terminals.jl — Email terminal alphabet and seed grammars

Email-specific seed grammars. GTExpr/LTExpr reference named features
directly (e.g., :urgency, :sender_is_manager).
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Credence: Grammar, ProductionRule
using Credence: GTExpr, LTExpr, AndExpr, OrExpr, NotExpr
using Credence: next_grammar_id, reset_grammar_counter!

# ═══════════════════════════════════════
# Seed grammars
# ═══════════════════════════════════════

"""
    generate_email_seed_grammars() → Vector{Grammar}

Generate 14 seed grammars for the email domain.
"""
function generate_email_seed_grammars()::Vector{Grammar}
    reset_grammar_counter!()
    grammars = Grammar[]

    # 1. Minimal: sender_freq, is_manager, urgency
    push!(grammars, Grammar(
        Set([:sender_frequency, :sender_is_manager, :urgency]),
        ProductionRule[], next_grammar_id()))

    # 2. Sender-focused: sender_freq, is_manager, is_dr, is_external
    push!(grammars, Grammar(
        Set([:sender_frequency, :sender_is_manager, :sender_is_direct_report, :sender_is_external]),
        ProductionRule[], next_grammar_id()))

    # 3. Topic-focused: topic_finance, topic_scheduling, topic_marketing
    push!(grammars, Grammar(
        Set([:topic_finance, :topic_scheduling, :topic_marketing]),
        ProductionRule[], next_grammar_id()))

    # 4. Action-focused: urgency, requires_action
    push!(grammars, Grammar(
        Set([:urgency, :requires_action]),
        ProductionRule[], next_grammar_id()))

    # 5. Content: urgency, email_length, has_attachment
    push!(grammars, Grammar(
        Set([:urgency, :email_length, :has_attachment]),
        ProductionRule[], next_grammar_id()))

    # 6. Metadata: time_of_day, thread_depth
    push!(grammars, Grammar(
        Set([:time_of_day, :thread_depth]),
        ProductionRule[], next_grammar_id()))

    # 7. Sender + urgency: sender_freq, is_manager, is_dr, is_external, urgency
    push!(grammars, Grammar(
        Set([:sender_frequency, :sender_is_manager, :sender_is_direct_report, :sender_is_external, :urgency]),
        ProductionRule[], next_grammar_id()))

    # 8. Full: all 13 content features
    push!(grammars, Grammar(
        ALL_EMAIL_FEATURES,
        ProductionRule[], next_grammar_id()))

    # 9. Minimal with FROM_BOSS nonterminal
    #    FROM_BOSS = GT(:sender_is_manager, 0.5)
    push!(grammars, Grammar(
        Set([:sender_frequency, :sender_is_manager, :urgency]),
        [ProductionRule(:FROM_BOSS, GTExpr(:sender_is_manager, 0.5))],
        next_grammar_id()))

    # 10. Action-focused with NEEDS_ATTENTION nonterminal
    #     NEEDS_ATTENTION = AND(GT(:requires_action, 0.5), GT(:urgency, 0.7))
    push!(grammars, Grammar(
        Set([:urgency, :requires_action]),
        [ProductionRule(:NEEDS_ATTENTION, AndExpr(GTExpr(:requires_action, 0.5), GTExpr(:urgency, 0.7)))],
        next_grammar_id()))

    # 11. Topic + urgency with ROUTINE nonterminal
    #     ROUTINE = AND(GT(:topic_marketing, 0.5), NOT(GT(:urgency, 0.7)))
    push!(grammars, Grammar(
        Set([:urgency, :topic_marketing]),
        [ProductionRule(:ROUTINE, AndExpr(GTExpr(:topic_marketing, 0.5), NotExpr(GTExpr(:urgency, 0.7))))],
        next_grammar_id()))

    # 12. Sender + urgency with FROM_BOSS + NEEDS_ATTENTION
    push!(grammars, Grammar(
        Set([:sender_frequency, :sender_is_manager, :urgency, :requires_action]),
        [ProductionRule(:FROM_BOSS, GTExpr(:sender_is_manager, 0.5)),
         ProductionRule(:NEEDS_ATTENTION, AndExpr(GTExpr(:requires_action, 0.5), GTExpr(:urgency, 0.7)))],
        next_grammar_id()))

    # 13. Sender + topic with ROUTINE + FROM_BOSS
    push!(grammars, Grammar(
        Set([:sender_frequency, :sender_is_manager, :urgency, :topic_marketing]),
        [ProductionRule(:ROUTINE, AndExpr(GTExpr(:topic_marketing, 0.5), NotExpr(GTExpr(:urgency, 0.7)))),
         ProductionRule(:FROM_BOSS, GTExpr(:sender_is_manager, 0.5))],
        next_grammar_id()))

    # 14. Sender with TRUSTED nonterminal
    #     TRUSTED = AND(GT(:sender_frequency, 0.5), NOT(GT(:sender_is_external, 0.5)))
    push!(grammars, Grammar(
        Set([:sender_frequency, :sender_is_manager, :sender_is_direct_report, :sender_is_external]),
        [ProductionRule(:TRUSTED, AndExpr(GTExpr(:sender_frequency, 0.5), NotExpr(GTExpr(:sender_is_external, 0.5))))],
        next_grammar_id()))

    grammars
end

"""
    generate_email_seed_grammars_extended() → Vector{Grammar}

Generate seed grammars for multi-step episodes. Includes the original 14
content-only grammars plus 4 processing-state-aware grammars.
"""
function generate_email_seed_grammars_extended()::Vector{Grammar}
    grammars = generate_email_seed_grammars()

    # 15. Triage: urgency, has_label_urgent, is_in_priority, user_notified
    #     NOT_YET_LABELLED = NOT(GT(:has_label_urgent, 0.5))
    push!(grammars, Grammar(
        Set([:urgency, :has_label_urgent, :is_in_priority, :user_notified]),
        [ProductionRule(:NOT_YET_LABELLED, NotExpr(GTExpr(:has_label_urgent, 0.5)))],
        next_grammar_id()))

    # 16. Archive: topic_marketing, is_in_archive, is_read
    push!(grammars, Grammar(
        Set([:topic_marketing, :is_in_archive, :is_read]),
        ProductionRule[], next_grammar_id()))

    # 17. Delegate: is_manager, has_label_delegated, is_assigned
    push!(grammars, Grammar(
        Set([:sender_is_manager, :has_label_delegated, :is_assigned]),
        ProductionRule[], next_grammar_id()))

    # 18. Full extended: all 22 features
    push!(grammars, Grammar(
        ALL_EMAIL_FEATURES_EXTENDED,
        ProductionRule[], next_grammar_id()))

    grammars
end

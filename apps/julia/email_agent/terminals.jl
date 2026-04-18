"""
    terminals.jl — Email terminal alphabet and seed grammars

Email-specific seed grammars. GTExpr/LTExpr reference raw observable
features directly (e.g., :subject_has_you, :sender_is_bulk_domain).
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
using Credence
using Credence: Grammar, ProductionRule
using Credence: GTExpr, LTExpr, AndExpr, OrExpr, NotExpr
using Credence: next_grammar_id, reset_grammar_counter!

# ═══════════════════════════════════════
# Seed grammars
# ═══════════════════════════════════════

"""
    generate_email_seed_grammars() → Vector{Grammar}

Generate seed grammars for the email domain using raw observable features.
"""
function generate_email_seed_grammars()::Vector{Grammar}
    reset_grammar_counter!()
    grammars = Grammar[]

    # 1. Personalization: "your"/"you" + confirmed + new event
    push!(grammars, Grammar(
        Set([:subject_has_you, :subject_has_confirmed, :subject_has_new_event]),
        ProductionRule[], next_grammar_id()))

    # 2. Actionability: failed + action_kw + money
    push!(grammars, Grammar(
        Set([:subject_has_failed, :subject_has_action_kw, :subject_has_money_kw]),
        ProductionRule[], next_grammar_id()))

    # 3. Newsletter signals: news sender + large + click + unsubscribe
    push!(grammars, Grammar(
        Set([:sender_has_news_kw, :is_large_html, :preview_has_click, :preview_has_unsubscribe]),
        ProductionRule[], next_grammar_id()))

    # 4. Sender type: noreply + bulk + news + frequency
    push!(grammars, Grammar(
        Set([:sender_is_noreply, :sender_is_bulk_domain, :sender_has_news_kw, :sender_frequency]),
        ProductionRule[], next_grammar_id()))

    # 5. Sender identity: hash bits for sender fingerprinting
    push!(grammars, Grammar(
        Set([:sender_h0, :sender_h1, :sender_h2, :sender_h3]),
        ProductionRule[], next_grammar_id()))

    # 6. Subject content: urgent + action + reply + fwd
    push!(grammars, Grammar(
        Set([:subject_has_urgent_kw, :subject_has_action_kw, :subject_is_reply, :subject_is_fwd]),
        ProductionRule[], next_grammar_id()))

    # 7. Mixed signals: personalized + large + attachment
    push!(grammars, Grammar(
        Set([:subject_has_you, :is_large_html, :has_attachment]),
        ProductionRule[], next_grammar_id()))

    # 8. Cross-cutting: personalization + sender type + size
    push!(grammars, Grammar(
        Set([:subject_has_you, :sender_is_noreply, :sender_has_news_kw,
             :is_large_html, :subject_has_new_event, :subject_has_confirmed]),
        ProductionRule[], next_grammar_id()))

    # 9. PERSONAL = AND(subject_has_you, NOT(sender_has_news_kw))
    push!(grammars, Grammar(
        Set([:subject_has_you, :sender_has_news_kw, :is_large_html]),
        [ProductionRule(:PERSONAL, AndExpr(GTExpr(:subject_has_you, 0.5), NotExpr(GTExpr(:sender_has_news_kw, 0.5))))],
        next_grammar_id()))

    # 10. NEWSLETTER = AND(is_large_html, OR(sender_has_news_kw, preview_has_click))
    push!(grammars, Grammar(
        Set([:is_large_html, :sender_has_news_kw, :preview_has_click]),
        [ProductionRule(:NEWSLETTER, AndExpr(GTExpr(:is_large_html, 0.5), OrExpr(GTExpr(:sender_has_news_kw, 0.5), GTExpr(:preview_has_click, 0.5))))],
        next_grammar_id()))

    # 11. NEEDS_ACTION = OR(subject_has_failed, subject_has_action_kw)
    push!(grammars, Grammar(
        Set([:subject_has_failed, :subject_has_action_kw, :subject_has_you]),
        [ProductionRule(:NEEDS_ACTION, OrExpr(GTExpr(:subject_has_failed, 0.5), GTExpr(:subject_has_action_kw, 0.5)))],
        next_grammar_id()))

    # 12. Sender hash + noreply with KNOWN_SENDER
    push!(grammars, Grammar(
        Set([:sender_h0, :sender_h1, :sender_h2, :sender_h3, :sender_is_noreply]),
        [ProductionRule(:KNOWN_SENDER, AndExpr(GTExpr(:sender_h0, 0.5), GTExpr(:sender_h2, 0.5)))],
        next_grammar_id()))

    # 13. Metadata: time + thread + length + question
    push!(grammars, Grammar(
        Set([:time_of_day, :thread_depth, :email_length, :preview_has_question]),
        ProductionRule[], next_grammar_id()))

    # 14. Money + attachment with INVOICE
    push!(grammars, Grammar(
        Set([:subject_has_money_kw, :has_attachment, :subject_has_confirmed]),
        [ProductionRule(:INVOICE, AndExpr(GTExpr(:subject_has_money_kw, 0.5), GTExpr(:has_attachment, 0.5)))],
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

    # 15. Triage state
    push!(grammars, Grammar(
        Set([:subject_has_urgent_kw, :has_label_urgent, :is_in_priority, :user_notified]),
        [ProductionRule(:NOT_YET_LABELLED, NotExpr(GTExpr(:has_label_urgent, 0.5)))],
        next_grammar_id()))

    # 16. Archive state
    push!(grammars, Grammar(
        Set([:preview_has_unsubscribe, :is_in_archive, :is_read]),
        ProductionRule[], next_grammar_id()))

    # 17. Delegate state
    push!(grammars, Grammar(
        Set([:sender_frequency, :has_label_delegated, :is_assigned]),
        ProductionRule[], next_grammar_id()))

    # 18. Cross-cutting extended
    push!(grammars, Grammar(
        Set([:subject_has_you, :sender_has_news_kw, :subject_has_failed,
             :has_label_urgent, :is_in_archive, :is_read]),
        ProductionRule[], next_grammar_id()))

    grammars
end

"""
    terminals.jl — Email terminal alphabet and seed grammars

Email-specific seed grammars. GTExpr channel indices are position-relative
(index into the grammar's sensor vector, 0-based), NOT global source indices.
compile_expr does ch+1 for Julia 1-based indexing.

Source index mapping (for _email_sensor_config):
    0 = sender_frequency       7 = topic_marketing
    1 = sender_is_manager      8 = requires_action
    2 = sender_is_direct_report 9 = email_length
    3 = sender_is_external     10 = has_attachment
    4 = urgency                11 = time_of_day
    5 = topic_finance          12 = thread_depth
    6 = topic_scheduling
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using Credence
using Credence: Grammar, ProductionRule, SensorChannel, SensorConfig
using Credence: GTExpr, LTExpr, AndExpr, OrExpr, NotExpr
using Credence: next_grammar_id, reset_grammar_counter!

# ═══════════════════════════════════════
# Sensor config constructor
# ═══════════════════════════════════════

function _email_sensor_config(source_indices::Vector{Int})
    SensorConfig([SensorChannel(si, :identity, 0.05, 1.0) for si in source_indices])
end

# ═══════════════════════════════════════
# Seed grammars
# ═══════════════════════════════════════

"""
    generate_email_seed_grammars() → Vector{Grammar}

Generate 14 seed grammars for the email domain. GTExpr channel indices
are positions in the grammar's sensor vector, not global source indices.
"""
function generate_email_seed_grammars()::Vector{Grammar}
    reset_grammar_counter!()
    grammars = Grammar[]

    # 1. Minimal: sender_freq(pos0), is_manager(pos1), urgency(pos2)
    push!(grammars, Grammar(
        _email_sensor_config([0, 1, 4]),
        ProductionRule[], next_grammar_id()))

    # 2. Sender-focused: sender_freq(pos0), is_manager(pos1), is_dr(pos2), is_external(pos3)
    push!(grammars, Grammar(
        _email_sensor_config([0, 1, 2, 3]),
        ProductionRule[], next_grammar_id()))

    # 3. Topic-focused: topic_finance(pos0), topic_scheduling(pos1), topic_marketing(pos2)
    push!(grammars, Grammar(
        _email_sensor_config([5, 6, 7]),
        ProductionRule[], next_grammar_id()))

    # 4. Action-focused: urgency(pos0), requires_action(pos1)
    push!(grammars, Grammar(
        _email_sensor_config([4, 8]),
        ProductionRule[], next_grammar_id()))

    # 5. Content: urgency(pos0), email_length(pos1), has_attachment(pos2)
    push!(grammars, Grammar(
        _email_sensor_config([4, 9, 10]),
        ProductionRule[], next_grammar_id()))

    # 6. Metadata: time_of_day(pos0), thread_depth(pos1)
    push!(grammars, Grammar(
        _email_sensor_config([11, 12]),
        ProductionRule[], next_grammar_id()))

    # 7. Sender + urgency: sender_freq(pos0), is_manager(pos1), is_dr(pos2), is_external(pos3), urgency(pos4)
    push!(grammars, Grammar(
        _email_sensor_config([0, 1, 2, 3, 4]),
        ProductionRule[], next_grammar_id()))

    # 8. Full: all 13 channels (pos0..pos12 = source0..source12)
    push!(grammars, Grammar(
        _email_sensor_config(collect(0:12)),
        ProductionRule[], next_grammar_id()))

    # 9. Minimal with FROM_BOSS nonterminal
    #    Channels: sender_freq(pos0), is_manager(pos1), urgency(pos2)
    #    FROM_BOSS = GT(pos1, 0.5) = is_manager
    push!(grammars, Grammar(
        _email_sensor_config([0, 1, 4]),
        [ProductionRule(:FROM_BOSS, GTExpr(1, 0.5))],
        next_grammar_id()))

    # 10. Action-focused with NEEDS_ATTENTION nonterminal
    #     Channels: urgency(pos0), requires_action(pos1)
    #     NEEDS_ATTENTION = AND(GT(pos1, 0.5), GT(pos0, 0.7))
    push!(grammars, Grammar(
        _email_sensor_config([4, 8]),
        [ProductionRule(:NEEDS_ATTENTION, AndExpr(GTExpr(1, 0.5), GTExpr(0, 0.7)))],
        next_grammar_id()))

    # 11. Topic + urgency with ROUTINE nonterminal
    #     Channels: urgency(pos0), topic_marketing(pos1)
    #     ROUTINE = AND(GT(pos1, 0.5), NOT(GT(pos0, 0.7)))
    push!(grammars, Grammar(
        _email_sensor_config([4, 7]),
        [ProductionRule(:ROUTINE, AndExpr(GTExpr(1, 0.5), NotExpr(GTExpr(0, 0.7))))],
        next_grammar_id()))

    # 12. Sender + urgency with FROM_BOSS + NEEDS_ATTENTION
    #     Channels: sender_freq(pos0), is_manager(pos1), urgency(pos2), requires_action(pos3)
    push!(grammars, Grammar(
        _email_sensor_config([0, 1, 4, 8]),
        [ProductionRule(:FROM_BOSS, GTExpr(1, 0.5)),
         ProductionRule(:NEEDS_ATTENTION, AndExpr(GTExpr(3, 0.5), GTExpr(2, 0.7)))],
        next_grammar_id()))

    # 13. Sender + topic with ROUTINE + FROM_BOSS
    #     Channels: sender_freq(pos0), is_manager(pos1), urgency(pos2), topic_marketing(pos3)
    push!(grammars, Grammar(
        _email_sensor_config([0, 1, 4, 7]),
        [ProductionRule(:ROUTINE, AndExpr(GTExpr(3, 0.5), NotExpr(GTExpr(2, 0.7)))),
         ProductionRule(:FROM_BOSS, GTExpr(1, 0.5))],
        next_grammar_id()))

    # 14. Sender with TRUSTED nonterminal
    #     Channels: sender_freq(pos0), is_manager(pos1), is_dr(pos2), is_external(pos3)
    #     TRUSTED = AND(GT(pos0, 0.5), NOT(GT(pos3, 0.5)))
    push!(grammars, Grammar(
        _email_sensor_config([0, 1, 2, 3]),
        [ProductionRule(:TRUSTED, AndExpr(GTExpr(0, 0.5), NotExpr(GTExpr(3, 0.5))))],
        next_grammar_id()))

    grammars
end

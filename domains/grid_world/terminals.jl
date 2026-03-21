"""
    terminals.jl — Grid-world terminal alphabet and seed grammars

Grid-world-specific seed grammars. These use source indices 0-7 which
map to RGB, position, speed, and wall distance channels.
"""

"""
    generate_seed_grammars() → Vector{Grammar}

Generate the grid-world seed grammar pool. These grammars use
sensor configs specific to the grid-world observation space.
"""
function generate_seed_grammars()::Vector{Grammar}
    reset_grammar_counter!()
    grammars = Grammar[]

    # 1. Empty grammar — minimal 2-channel (red + speed), no rules
    push!(grammars, Grammar(minimal_sensor_config(), ProductionRule[], next_grammar_id()))

    # 2. Colour-only grammar — r, g, b channels, no rules
    push!(grammars, Grammar(colour_sensor_config(), ProductionRule[], next_grammar_id()))

    # 3. Motion grammar — speed + wall_dist, no rules
    push!(grammars, Grammar(motion_sensor_config(), ProductionRule[], next_grammar_id()))

    # 4. Full sensor grammar — all 8 channels, no rules
    push!(grammars, Grammar(full_sensor_config(), ProductionRule[], next_grammar_id()))

    # 5. Colour + speed grammar — r, g, b, speed
    push!(grammars, Grammar(colour_speed_sensor_config(), ProductionRule[], next_grammar_id()))

    # 6. Colour grammar with RED nonterminal
    red_body = AndExpr(GTExpr(0, 0.7), AndExpr(LTExpr(1, 0.3), LTExpr(2, 0.3)))
    push!(grammars, Grammar(
        colour_sensor_config(),
        [ProductionRule(:RED, red_body)],
        next_grammar_id()))

    # 7. Colour grammar with BLUE nonterminal
    blue_body = AndExpr(LTExpr(0, 0.3), AndExpr(LTExpr(1, 0.3), GTExpr(2, 0.7)))
    push!(grammars, Grammar(
        colour_sensor_config(),
        [ProductionRule(:BLUE, blue_body)],
        next_grammar_id()))

    # 8. Motion grammar with MOVING nonterminal
    push!(grammars, Grammar(
        motion_sensor_config(),
        [ProductionRule(:MOVING, GTExpr(0, 0.3))],  # channel 0 in motion config = speed
        next_grammar_id()))

    # 9. Motion grammar with NEAR_WALL nonterminal
    push!(grammars, Grammar(
        motion_sensor_config(),
        [ProductionRule(:NEAR_WALL, LTExpr(1, 0.3))],  # channel 1 = wall_dist
        next_grammar_id()))

    # 10. Colour+speed with RED and MOVING nonterminals
    red_cs = AndExpr(GTExpr(0, 0.7), AndExpr(LTExpr(1, 0.3), LTExpr(2, 0.3)))
    moving_cs = GTExpr(3, 0.3)  # channel 3 = speed in colour+speed config
    push!(grammars, Grammar(
        colour_speed_sensor_config(),
        [ProductionRule(:RED, red_cs), ProductionRule(:MOVING, moving_cs)],
        next_grammar_id()))

    # 11. Colour+speed with RED_AND_MOVING compound nonterminal
    ram = AndExpr(
        AndExpr(GTExpr(0, 0.7), AndExpr(LTExpr(1, 0.3), LTExpr(2, 0.3))),
        GTExpr(3, 0.3))
    push!(grammars, Grammar(
        colour_speed_sensor_config(),
        [ProductionRule(:RED_AND_MOVING, ram)],
        next_grammar_id()))

    # 12. Full sensor with colour nonterminals
    push!(grammars, Grammar(
        full_sensor_config(),
        [ProductionRule(:RED, AndExpr(GTExpr(0, 0.7), AndExpr(LTExpr(1, 0.3), LTExpr(2, 0.3)))),
         ProductionRule(:BLUE, AndExpr(LTExpr(0, 0.3), AndExpr(LTExpr(1, 0.3), GTExpr(2, 0.7))))],
        next_grammar_id()))

    grammars
end

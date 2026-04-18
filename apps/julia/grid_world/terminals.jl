"""
    terminals.jl — Grid-world terminal alphabet and seed grammars

Grid-world-specific seed grammars. GTExpr/LTExpr reference named features
(:red, :green, :blue, :x_norm, :y_norm, :speed, :wall_dist, :agent_dist).
"""

const ALL_GW_FEATURES = Set([:red, :green, :blue, :x_norm, :y_norm, :speed, :wall_dist, :agent_dist])

"""
    generate_seed_grammars() → Vector{Grammar}

Generate the grid-world seed grammar pool.
"""
function generate_seed_grammars()::Vector{Grammar}
    reset_grammar_counter!()
    grammars = Grammar[]

    # 1. Minimal: red + speed, no rules
    push!(grammars, Grammar(Set([:red, :speed]), ProductionRule[], next_grammar_id()))

    # 2. Colour-only: r, g, b
    push!(grammars, Grammar(Set([:red, :green, :blue]), ProductionRule[], next_grammar_id()))

    # 3. Motion: speed + wall_dist
    push!(grammars, Grammar(Set([:speed, :wall_dist]), ProductionRule[], next_grammar_id()))

    # 4. Full sensor: all 8 features
    push!(grammars, Grammar(ALL_GW_FEATURES, ProductionRule[], next_grammar_id()))

    # 5. Colour + speed: r, g, b, speed
    push!(grammars, Grammar(Set([:red, :green, :blue, :speed]), ProductionRule[], next_grammar_id()))

    # 6. Colour grammar with RED nonterminal
    red_body = AndExpr(GTExpr(:red, 0.7), AndExpr(LTExpr(:green, 0.3), LTExpr(:blue, 0.3)))
    push!(grammars, Grammar(
        Set([:red, :green, :blue]),
        [ProductionRule(:RED, red_body)],
        next_grammar_id()))

    # 7. Colour grammar with BLUE nonterminal
    blue_body = AndExpr(LTExpr(:red, 0.3), AndExpr(LTExpr(:green, 0.3), GTExpr(:blue, 0.7)))
    push!(grammars, Grammar(
        Set([:red, :green, :blue]),
        [ProductionRule(:BLUE, blue_body)],
        next_grammar_id()))

    # 8. Motion grammar with MOVING nonterminal
    push!(grammars, Grammar(
        Set([:speed, :wall_dist]),
        [ProductionRule(:MOVING, GTExpr(:speed, 0.3))],
        next_grammar_id()))

    # 9. Motion grammar with NEAR_WALL nonterminal
    push!(grammars, Grammar(
        Set([:speed, :wall_dist]),
        [ProductionRule(:NEAR_WALL, LTExpr(:wall_dist, 0.3))],
        next_grammar_id()))

    # 10. Colour+speed with RED and MOVING nonterminals
    red_cs = AndExpr(GTExpr(:red, 0.7), AndExpr(LTExpr(:green, 0.3), LTExpr(:blue, 0.3)))
    moving_cs = GTExpr(:speed, 0.3)
    push!(grammars, Grammar(
        Set([:red, :green, :blue, :speed]),
        [ProductionRule(:RED, red_cs), ProductionRule(:MOVING, moving_cs)],
        next_grammar_id()))

    # 11. Colour+speed with RED_AND_MOVING compound nonterminal
    ram = AndExpr(
        AndExpr(GTExpr(:red, 0.7), AndExpr(LTExpr(:green, 0.3), LTExpr(:blue, 0.3))),
        GTExpr(:speed, 0.3))
    push!(grammars, Grammar(
        Set([:red, :green, :blue, :speed]),
        [ProductionRule(:RED_AND_MOVING, ram)],
        next_grammar_id()))

    # 12. Full sensor with colour nonterminals
    push!(grammars, Grammar(
        ALL_GW_FEATURES,
        [ProductionRule(:RED, AndExpr(GTExpr(:red, 0.7), AndExpr(LTExpr(:green, 0.3), LTExpr(:blue, 0.3)))),
         ProductionRule(:BLUE, AndExpr(LTExpr(:red, 0.3), AndExpr(LTExpr(:green, 0.3), GTExpr(:blue, 0.7))))],
        next_grammar_id()))

    grammars
end

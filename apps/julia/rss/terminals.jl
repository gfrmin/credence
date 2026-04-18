"""
    terminals.jl — RSS seed grammars for preference learning.

    Grammars are built dynamically from the feature registry since
    feed names and tag names vary per installation.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "src"))
using Credence
using Credence: Grammar, ProductionRule
using Credence: GTExpr, LTExpr, AndExpr, OrExpr, NotExpr
using Credence: next_grammar_id, reset_grammar_counter!

"""
    generate_rss_seed_grammars(reg::FeatureRegistry) → Vector{Grammar}

Generate seed grammars for the RSS domain. Dynamic: adapts to the
user's actual feeds, categories, and LLM-generated tags.
"""
function generate_rss_seed_grammars(reg::FeatureRegistry)::Vector{Grammar}
    reset_grammar_counter!()
    grammars = Grammar[]

    # 1. Content structure
    push!(grammars, Grammar(
        Set([:log_word_count, :has_code, :has_images, :heading_count_norm]),
        ProductionRule[], next_grammar_id()))

    # 2. Temporal patterns
    push!(grammars, Grammar(
        Set([:hour_sin, :hour_cos, :dow_sin, :dow_cos, :is_weekend]),
        ProductionRule[], next_grammar_id()))

    # 3. Priority + freshness
    push!(grammars, Grammar(
        Set([:feed_priority, :log_age_hours]),
        ProductionRule[], next_grammar_id()))

    # 4. Priority + content (cross-cutting)
    push!(grammars, Grammar(
        Set([:feed_priority, :has_code, :log_word_count, :log_age_hours]),
        ProductionRule[], next_grammar_id()))

    # 5. Content + temporal (weekend long reads, weekday short)
    push!(grammars, Grammar(
        Set([:log_word_count, :is_weekend, :has_audio]),
        ProductionRule[], next_grammar_id()))

    # 6. Per-feed grammars (up to 5 feeds, highest priority first)
    feed_syms = collect(values(reg.feed_map))
    for sym in first(feed_syms, 5)
        push!(grammars, Grammar(
            Set([sym, :feed_priority, :log_word_count]),
            ProductionRule[], next_grammar_id()))
    end

    # 7. Per-category grammars
    for sym in values(reg.cat_map)
        push!(grammars, Grammar(
            Set([sym, :feed_priority]),
            ProductionRule[], next_grammar_id()))
    end

    # 8. Tag combination grammars (groups of 3-4 tags)
    tag_syms = collect(values(reg.tag_map))
    for chunk in Iterators.partition(tag_syms, 4)
        length(chunk) >= 2 || continue
        push!(grammars, Grammar(
            Set(collect(chunk)),
            ProductionRule[], next_grammar_id()))
    end

    # 9. Tag + priority (cross-cutting)
    if !isempty(tag_syms)
        top_tags = first(tag_syms, 3)
        push!(grammars, Grammar(
            Set(vcat(top_tags, [:feed_priority])),
            ProductionRule[], next_grammar_id()))
    end

    # 10. Nonterminal: TECHNICAL = has_code AND word_count > medium
    push!(grammars, Grammar(
        Set([:has_code, :log_word_count, :feed_priority]),
        [ProductionRule(:TECHNICAL, AndExpr(GTExpr(:has_code, 0.5), GTExpr(:log_word_count, 0.3)))],
        next_grammar_id()))

    grammars
end

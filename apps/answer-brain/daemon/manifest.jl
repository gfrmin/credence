# Role: brain
"""
    manifest.jl — serve the body's vocabulary from the bdsl, through the ONE library parser.

The de-dup cut (move-4-design §0, §5 Q1): credence-pi parses its manifest a *second* time in
TypeScript (`extension/src/manifest.ts`, 159 lines). The library already owns the single BDSL
parser (`src/parse.jl`); so the answer-brain body has **no** parser — the daemon reads
`bdsl/capabilities.bdsl` + `bdsl/features.bdsl` via `Credence.parse_all` and serves the effector /
feature vocabulary as JSON (`GET /manifest`). The body fetches it and verifies every declared
effector / feature has a registered impl.

The served shape mirrors the `EffectorDecl` / `FeatureDecl` interfaces the TS body consumes:
    {effectors:[{name, parameters:[{name,type}]}], features:[{name, spaceName}]}

Read-only and stateless — parsing two small files per request; the body fetches once at startup.
"""
module Manifest

using Credence: parse_all
using Credence.Parse: SExpr, Atom, SList

export manifest_dict

# Walk the parsed forms, collecting every (head …) list. Like credence-pi's
# manifest.ts `collect`: descend through nested lists, but do NOT recurse into a
# form once its head matches (an (effector …) has no nested (effector …)).
function _collect(exprs::Vector{SExpr}, head::Symbol)::Vector{SList}
    out = SList[]
    function visit(e::SExpr)
        e isa SList || return
        if !isempty(e.items) && e.items[1] isa Atom && (e.items[1]::Atom).value === head
            push!(out, e)
        else
            for child in e.items
                visit(child)
            end
        end
    end
    for e in exprs
        visit(e)
    end
    out
end

_sym(e::SExpr)::String =
    e isa Atom ? string((e::Atom).value) : error("manifest: expected an atom, got $(e)")

# (effector NAME (parameters (p type) …)) → {name, parameters:[{name,type}]}
function _effector(form::SList)::Dict{String, Any}
    length(form.items) >= 2 || error("manifest: effector form missing name in $(form)")
    name = _sym(form.items[2])
    params = Dict{String, String}[]
    for i in 3:length(form.items)
        clause = form.items[i]
        if !(clause isa SList && !isempty(clause.items) &&
             clause.items[1] isa Atom && (clause.items[1]::Atom).value === :parameters)
            error("manifest: effector $(name): expected (parameters …), got $(clause)")
        end
        for j in 2:length(clause.items)
            p = clause.items[j]
            (p isa SList && length((p::SList).items) == 2) ||
                error("manifest: effector $(name): parameter must be (name type), got $(p)")
            push!(params, Dict("name" => _sym((p::SList).items[1]),
                               "type" => _sym((p::SList).items[2])))
        end
    end
    Dict{String, Any}("name" => name, "parameters" => params)
end

# (feature NAME SPACE) → {name, spaceName}
function _feature(form::SList)::Dict{String, String}
    length(form.items) == 3 ||
        error("manifest: feature form must be (feature NAME SPACE), got $(form)")
    Dict("name" => _sym(form.items[2]), "spaceName" => _sym(form.items[3]))
end

# Scope features to the `(define features (list …))` form — a sibling
# `(define safety-features …)` declares a different body's vocabulary and is
# intentionally not served here (parity with manifest.ts's `parseFeatures`).
function _features_scope(exprs::Vector{SExpr})::Vector{SExpr}
    for e in exprs
        if e isa SList && length((e::SList).items) >= 2 &&
           (e::SList).items[1] isa Atom && ((e::SList).items[1]::Atom).value === :define &&
           (e::SList).items[2] isa Atom && ((e::SList).items[2]::Atom).value === :features
            return SExpr[e]
        end
    end
    exprs
end

"""
    manifest_dict(; capabilities_path, features_path) -> Dict

Parse the two bdsl files through the library parser and return the served manifest. Throws (→ HTTP
500 at the route) on a malformed manifest; the body's `verify` is the second gate (a declared
effector with no impl fails the body closed).
"""
function manifest_dict(; capabilities_path::AbstractString,
                         features_path::AbstractString)::Dict{String, Any}
    caps = parse_all(read(capabilities_path, String))
    feats = parse_all(read(features_path, String))
    Dict{String, Any}(
        "effectors" => [_effector(f) for f in _collect(caps, :effector)],
        "features"  => [_feature(f) for f in _collect(_features_scope(feats), :feature)],
    )
end

end # module Manifest

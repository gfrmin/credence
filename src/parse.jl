"""
    parse.jl — S-expression parser. Identical to v1.

Grammar:  expr = atom | '(' expr* ')'
Atom:     number | symbol | string
"""
module Parse

export SExpr, Atom, SList, parse_sexpr, parse_all, complexity

abstract type SExpr end

struct Atom <: SExpr
    value::Union{Symbol, Float64, Int, Bool, String}
end

struct SList <: SExpr
    items::Vector{SExpr}
end

Base.show(io::IO, a::Atom) = print(io, a.value)
function Base.show(io::IO, s::SList)
    print(io, "(")
    join(io, s.items, " ")
    print(io, ")")
end

complexity(::Atom) = 1
complexity(s::SList) = 1 + sum(complexity, s.items; init=0)

function tokenise(src::String)
    tokens = String[]
    chars = collect(src)
    i = 1
    while i <= length(chars)
        c = chars[i]
        if c == '(' || c == ')'
            push!(tokens, string(c))
            i += 1
        elseif c == ';'
            while i <= length(chars) && chars[i] != '\n'; i += 1; end
        elseif c == '"'
            j = i + 1
            while j <= length(chars) && chars[j] != '"'; j += 1; end
            push!(tokens, String(chars[i:j]))
            i = j + 1
        elseif isspace(c)
            i += 1
        else
            j = i
            while j <= length(chars) && !isspace(chars[j]) && chars[j] != '(' && chars[j] != ')'
                j += 1
            end
            push!(tokens, String(chars[i:j-1]))
            i = j
        end
    end
    tokens
end

function parse_atom(token::String)
    startswith(token, '"') && endswith(token, '"') && return Atom(token[2:end-1])
    token == "Inf" && return Atom(Inf)
    token == "-Inf" && return Atom(-Inf)
    v = tryparse(Int, token)
    v !== nothing && return Atom(v)
    v = tryparse(Float64, token)
    v !== nothing && return Atom(v)
    token == "true" && return Atom(true)
    token == "false" && return Atom(false)
    # :keyword syntax — strip colon, create symbol
    startswith(token, ':') && length(token) > 1 && return Atom(Symbol(token[2:end]))
    Atom(Symbol(token))
end

function parse_expr(tokens::Vector{String}, pos::Int)
    pos > length(tokens) && error("unexpected end of input")
    if tokens[pos] == "("
        pos += 1
        items = SExpr[]
        while pos <= length(tokens) && tokens[pos] != ")"
            expr, pos = parse_expr(tokens, pos)
            push!(items, expr)
        end
        pos > length(tokens) && error("missing closing ')'")
        return SList(items), pos + 1
    elseif tokens[pos] == ")"
        error("unexpected ')'")
    else
        return parse_atom(tokens[pos]), pos + 1
    end
end

function parse_sexpr(src::String)
    tokens = tokenise(src)
    expr, _ = parse_expr(tokens, 1)
    expr
end

function parse_all(src::String)
    tokens = tokenise(src)
    exprs = SExpr[]
    pos = 1
    while pos <= length(tokens)
        expr, pos = parse_expr(tokens, pos)
        push!(exprs, expr)
    end
    exprs
end

end # module Parse

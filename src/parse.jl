"""
    parse.jl — S-expression parser. The entire front-end.

Grammar:  expr = atom | '(' expr* ')'
Atom:     number | symbol
That's it. There is no other syntax.
"""
module Parse

export SExpr, Atom, SList, parse_sexpr, parse_all

# AST: an expression is either an atom or a list of expressions
abstract type SExpr end

struct Atom <: SExpr
    value::Union{Symbol, Float64, Int, Bool}
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

# Count nodes — proxy for Kolmogorov complexity
complexity(::Atom) = 1
complexity(s::SList) = 1 + sum(complexity, s.items; init=0)

# Tokeniser: split on parens and whitespace
# Uses character iteration to handle Unicode safely
function tokenise(src::String)
    tokens = String[]
    chars = collect(src)
    i = 1
    while i <= length(chars)
        c = chars[i]
        if c == '(' || c == ')'
            push!(tokens, string(c))
            i += 1
        elseif c == ';'  # line comment
            while i <= length(chars) && chars[i] != '\n'
                i += 1
            end
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

# Parse atom: try number, then bool, then symbol
function parse_atom(token::String)
    # Try integer
    v = tryparse(Int, token)
    v !== nothing && return Atom(v)
    # Try float
    v = tryparse(Float64, token)
    v !== nothing && return Atom(v)
    # Booleans
    token == "true" && return Atom(true)
    token == "false" && return Atom(false)
    # Symbol
    Atom(Symbol(token))
end

# Recursive descent — returns (expr, next_position)
function parse_expr(tokens::Vector{String}, pos::Int)
    pos > length(tokens) && error("unexpected end of input")

    if tokens[pos] == "("
        pos += 1  # consume '('
        items = SExpr[]
        while pos <= length(tokens) && tokens[pos] != ")"
            expr, pos = parse_expr(tokens, pos)
            push!(items, expr)
        end
        pos > length(tokens) && error("missing closing ')'")
        pos += 1  # consume ')'
        return SList(items), pos
    elseif tokens[pos] == ")"
        error("unexpected ')'")
    else
        return parse_atom(tokens[pos]), pos + 1
    end
end

"""Parse a single S-expression from a string."""
function parse_sexpr(src::String)
    tokens = tokenise(src)
    isempty(tokens) && error("empty input")
    expr, pos = parse_expr(tokens, 1)
    expr
end

"""Parse all S-expressions from a string."""
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

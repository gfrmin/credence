"""
    prevision.jl — the de Finettian primitive type and its test function hierarchy.

Declared here (Move 1 of the Posture 3 reconstruction):
  - `Prevision` (abstract) — coherent linear functional on a declared test
    function space, in the sense of de Finetti (1974) and Whittle (1992).
  - `TestFunction` (abstract) and concrete shells: `Identity`, `Projection`,
    `NestedProjection`, `Tabular`, `LinearCombination`, `OpaqueClosure`,
    `Indicator`.
  - `apply` — abstract evaluator `apply(f::TestFunction, s) → ℝ`. Methods
    land in Move 2.

Not declared here:
  - `expect` dispatch (Move 2)
  - `ConjugatePrevision`, `MixturePrevision`, `ExchangeablePrevision`, etc.
    (Moves 4-5)
  - `ParticlePrevision`, `QuadraturePrevision` (Move 6)
  - `condition(p::Prevision, e::Event)` primary form (Move 7)
  - `TestFunctionSpace` container — deferred; see
    `docs/posture-3/move-1-design.md` §5.1 for the three-option analysis
    that settled on option (c).

Loaded before `ontology.jl` so that Move 3's Measure-as-view refactor can
wrap `Prevision` subtypes in `Measure` fields. At Move 1 there are no
`Prevision` subtypes yet and no consumers; the ordering is forward-looking.

See `docs/posture-3/master-plan.md` for the full branch plan.
"""
module Previsions

export Prevision, TestFunction, Indicator, apply, expect
export Identity, Projection, NestedProjection, Tabular, LinearCombination, OpaqueClosure
export BetaPrevision
# At Move 2, `Ontology`'s `Functional` hierarchy is aliased onto these
# types (`const Functional = TestFunction` plus `import ..Previsions:
# Identity, …`), so both modules export the same bindings (they resolve
# to the same objects via `===`). `Credence` brings them into scope via
# both `using .Previsions` and `using .Ontology` without ambiguity.

# ── Prevision ─────────────────────────────────────────────────────────────

"""
    Prevision

Abstract type for a coherent linear functional on a declared test function
space. Coherence is the single rationality axiom (Dutch-book equivalent;
de Finetti 1937, 1974; Walley 1991):

1. Non-negativity: `p(f) ≥ 0` when `f ≥ 0` everywhere.
2. Normalisation: `p(𝟙) = 1`.
3. Linearity: `p(αf + βg) = α·p(f) + β·p(g)`.
4. (Optional) σ-continuity: `p(inf_n f_n) = inf_n p(f_n)` when `f_n ↓` —
   Whittle's A5. A technical strengthening for the bounded-measurable
   setting; Credence's default `Prevision` is finitely coherent, with
   σ-continuous subclasses declared explicitly where needed.

Concrete `Prevision` subtypes land in subsequent moves:
  - Move 3: `BetaPrevision`, `CategoricalPrevision`, `GaussianPrevision`,
    `GammaPrevision`, `DirichletPrevision`, `NormalGammaPrevision`, etc.
  - Move 4: `ConjugatePrevision{Prior, Likelihood}`
  - Move 5: `MixturePrevision`, `ExchangeablePrevision`
  - Move 6: `ParticlePrevision`, `QuadraturePrevision`

Move 1 declares only the abstract type — no dispatch, no consumers.
"""
abstract type Prevision end

# ── TestFunction hierarchy ────────────────────────────────────────────────

"""
    TestFunction

Abstract type for test functions — the arguments to a Prevision's action.
A `TestFunction` `f` is any element of the declared test function space
over a base `Space`; the Prevision's action `expect(p, f)` evaluates `f`
at the (implicit) representing measure.

Concrete subtypes declared here are *shells*: no dispatch yet. Move 2
wires the `expect` dispatch through these types, aliasing the existing
`Functional` hierarchy in `ontology.jl` onto them.
"""
abstract type TestFunction end

"""
    Identity() <: TestFunction

The identity test function: `apply(Identity(), s) = s` for scalar `s`.
Used to compute the mean of a scalar-valued prevision.
"""
struct Identity <: TestFunction end

"""
    Projection(index::Int) <: TestFunction

Selects the `index`-th factor of a product-structured point (e.g. an atom
from a `ProductMeasure` over a `ProductSpace`). `index` must be >= 1.
"""
struct Projection <: TestFunction
    index::Int  # 1-based index into ProductMeasure factors
    function Projection(index::Int)
        index >= 1 || throw(ArgumentError("Projection.index must be >= 1, got $index"))
        new(index)
    end
end

"""
    NestedProjection(indices::Vector{Int}) <: TestFunction

Selects a nested factor through a path of indices — walks successive
`ProductSpace` levels, terminating at a scalar leaf. `indices` must be
non-empty and each index must be >= 1.
"""
struct NestedProjection <: TestFunction
    indices::Vector{Int}  # 1-based
    function NestedProjection(indices::Vector{Int})
        isempty(indices) && throw(ArgumentError("NestedProjection requires at least one index"))
        all(i -> i >= 1, indices) ||
            throw(ArgumentError("NestedProjection.indices must all be >= 1, got $indices"))
        new(indices)
    end
end

"""
    Tabular(values::Vector{Float64}) <: TestFunction

A tabulated test function over a finite space: value at atom `i` is
`values[i]`. `values` must be non-empty. Length must match the space's
atom count at dispatch time (enforced by the `expect(::CategoricalMeasure,
::Tabular)` method).
"""
struct Tabular <: TestFunction
    values::Vector{Float64}
    function Tabular(values::Vector{Float64})
        isempty(values) && throw(ArgumentError("Tabular requires non-empty values"))
        new(values)
    end
end

"""
    LinearCombination(terms, offset=0.0) <: TestFunction

A linear combination of test functions: `sum(c * f for (c, f) in terms)
+ offset`. Composition is part of structure — each sub-functional
navigates its own shape via its own `apply` method.
"""
struct LinearCombination <: TestFunction
    terms::Vector{Tuple{Float64, TestFunction}}
    offset::Float64

    LinearCombination(terms::Vector{Tuple{Float64, TestFunction}}, offset::Float64 = 0.0) =
        new(terms, offset)
end

"""
    OpaqueClosure(f::Function) <: TestFunction

Escape hatch: a test function wrapped around an opaque Julia closure.
Use this only when no declared `TestFunction` subtype fits; forfeits the
type-structural dispatch the other subtypes enable, falling back to
whatever default integration strategy the prevision implements.
"""
struct OpaqueClosure <: TestFunction
    f::Function
end

"""
    Indicator{E}(event::E) <: TestFunction

The indicator of an event: `apply(Indicator(e), s) = 1` if `s ∈ e`, else
`0`. The bridge between previsions and probabilities —
`probability(μ, e) = expect(μ.prevision, Indicator(e))` (Move 3).

Parametric on `E` because the `Event` hierarchy lives in `ontology.jl`,
which loads *after* `prevision.jl`. The structural constraint `E <:
Ontology.Event` is enforced at method dispatch (Move 7's
`condition(p, e::Event)` wraps `e` into `Indicator(e)` only for `Event`
subtypes), not at the field level. See `docs/posture-3/move-1-design.md`
§5.3 for the placement discussion.
"""
struct Indicator{E} <: TestFunction
    event::E
end

# ── Concrete Prevision subtypes (Move 3) ──────────────────────────────────

"""
    BetaPrevision(alpha::Float64, beta::Float64) <: Prevision

Prevision whose representing measure is Beta(α, β) on [0, 1]. The
`BetaMeasure` view wraps a `BetaPrevision` and forwards `m.alpha`,
`m.beta` reads through its `getproperty` shield (see `src/ontology.jl`).

Move 3 declares this as a shell; Move 4 adds it as the Prior-type in
`ConjugatePrevision{BetaPrevision, Bernoulli}` for the conjugate registry.
"""
struct BetaPrevision <: Prevision
    alpha::Float64
    beta::Float64

    function BetaPrevision(alpha::Float64, beta::Float64)
        alpha > 0 && beta > 0 || error("alpha and beta must be positive")
        new(alpha, beta)
    end
end

# ── apply — abstract evaluator ────────────────────────────────────────────

"""
    apply(f::TestFunction, s) → ℝ

Evaluate a test function `f` at a point `s`. Methods land in Move 2 —
one per `(Prevision, TestFunction)` pair that requires a distinct
implementation. This declaration exists so that Move 2's additions are
method-level (on an already-declared function), not type-level.
"""
function apply end

# ── expect — the definitional operator ────────────────────────────────────

"""
    expect(p::Prevision, f::TestFunction) → ℝ
    expect(m::Measure, f::TestFunction) → ℝ

Integration against a prevision / measure. In the de Finettian view,
`expect` is not a derived operation but the *definition* of what a
prevision is: a prevision is the operator that assigns a real number to
each test function in its declared space. `expect(p, f) = p(f)`.

The declaration here exists so methods in `ontology.jl` (Measure
subtype × TestFunction subtype dispatches, plus the `Function`-argument
quadrature and Monte Carlo fallbacks) extend a single generic. Move 2
adds the import in `ontology.jl` (`import ..Previsions: expect`) so the
existing 16 dispatch methods attach to this declaration.

Subsequent moves (3-6) add new Prevision subtypes with their own `expect`
methods; Move 7 inverts the derivation (`expect(p, f)` becomes the
primitive, `condition(p, k, obs)` derived via `Indicator(ObservationEvent(k, obs))`).
"""
function expect end

end # module Previsions

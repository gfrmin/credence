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
export BetaPrevision, TaggedBetaPrevision, GaussianPrevision, GammaPrevision, CategoricalPrevision, DirichletPrevision, NormalGammaPrevision, ProductPrevision, MixturePrevision
export ExchangeablePrevision, decompose
export ParticlePrevision, QuadraturePrevision, EnumerationPrevision
export ConditionalPrevision
export ConjugatePrevision, maybe_conjugate, update, _dispatch_path
export push_component!, replace_component!
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

"""
    TaggedBetaPrevision(tag::Int, beta) <: Prevision

Prevision for a tag-bearing Beta component in a mixture. Holds a tag and
the underlying Beta. The `beta` field holds a `BetaMeasure` (not a raw
`BetaPrevision`) so that consumer code accessing `m.beta.alpha` via the
`TaggedBetaMeasure` shield receives a BetaMeasure and its shield handles
the subsequent `.alpha` access. This preserves the existing consumer
idiom without reconstructing a BetaMeasure wrapper on each access.

(Note: holding a Measure inside a Prevision is a minor semantic
impurity — the de Finettian purist view would wrap a BetaPrevision
here. The pragmatic choice preserves 0 consumer edits and allocates
nothing on access; a future cleanup pass can replace with BetaPrevision
+ view-reconstruction if the performance profile justifies.)
"""
struct TaggedBetaPrevision <: Prevision
    tag::Int
    beta::Any  # BetaMeasure — forward declaration before BetaMeasure loads in ontology.jl
end

"""
    GaussianPrevision(mu::Float64, sigma::Float64) <: Prevision

Prevision whose representing measure is N(μ, σ²) on Euclidean space.
The `GaussianMeasure` view wraps a `GaussianPrevision` and forwards
`m.mu`, `m.sigma` reads through its `getproperty` shield.
"""
struct GaussianPrevision <: Prevision
    mu::Float64
    sigma::Float64
end

"""
    GammaPrevision(alpha::Float64, beta::Float64) <: Prevision

Prevision whose representing measure is Gamma(α, β) on positive reals
(shape α, rate β). The `GammaMeasure` view wraps a `GammaPrevision` and
forwards `m.alpha`, `m.beta` reads through its `getproperty` shield.
"""
struct GammaPrevision <: Prevision
    alpha::Float64
    beta::Float64

    function GammaPrevision(alpha::Float64, beta::Float64)
        alpha > 0 && beta > 0 || error("alpha and beta must be positive")
        new(alpha, beta)
    end
end

"""
    CategoricalPrevision(log_weights::Vector{Float64}) <: Prevision

Prevision whose representing measure is categorical over a finite space,
with log-weights `log_weights` normalised at construction time. The
`CategoricalMeasure{T}` view wraps a `CategoricalPrevision` and forwards
`m.logw` reads through its `getproperty` shield (Move 5 retires the shield
with `CategoricalMeasure`; the external `m.logw` surface persists until
then for Moves 2–4 consumer compatibility).

Field-name unification (Posture 4 Move 1): the five log-mass carriers
(`CategoricalPrevision`, `ParticlePrevision`, `QuadraturePrevision`,
`EnumerationPrevision`, `MixturePrevision`) all store their normalised
log-weight vector under the name `log_weights`. Prior to Move 1,
`CategoricalPrevision.logw` was the one outlier; this rename aligns
the surface.

The `log_weights` field is returned by reference through the shield — see
the shared-reference contract in docs/posture-3/move-3-design.md §3. In
practice `log_weights` is never mutated post-construction (the normalisation
is one-shot in the constructor), but the contract is maintained for
consistency with MixturePrevision's `components` / `log_weights` where
in-place mutation is a real consumer pattern.

Not parametric on the atom type — the type connection lives at the
Measure level (`CategoricalMeasure{T}.space::Finite{T}`). `log_weights`
values stand alone as a probability vector.
"""
struct CategoricalPrevision <: Prevision
    log_weights::Vector{Float64}

    function CategoricalPrevision(log_weights::Vector{Float64})
        if all(lw -> lw == -Inf, log_weights)
            error("measure has zero total mass — all hypotheses impossible")
        end
        max_lw = maximum(log_weights)
        log_total = max_lw + log(sum(exp.(log_weights .- max_lw)))
        new(log_weights .- log_total)
    end
end

"""
    DirichletPrevision(alpha::Vector{Float64}) <: Prevision

Prevision whose representing measure is Dirichlet(α) on the simplex Δ^(k-1).
The `DirichletMeasure` view wraps a `DirichletPrevision` and forwards
`m.alpha` reads through its `getproperty` shield. The alpha vector is
returned by reference (shared-reference contract).
"""
struct DirichletPrevision <: Prevision
    alpha::Vector{Float64}

    function DirichletPrevision(alpha::Vector{Float64})
        all(a -> a > 0, alpha) || error("all alpha must be positive")
        new(alpha)
    end
end

"""
    NormalGammaPrevision(κ, μ, α, β) <: Prevision

Prevision whose representing measure is the Normal-Gamma conjugate prior
for a Normal with unknown mean and variance. Carries the four
hyperparameters (κ pseudo-obs count, μ mean location, α shape, β rate).
The `NormalGammaMeasure` view wraps it and forwards `m.κ`, `m.μ`, `m.α`,
`m.β` through the `getproperty` shield.
"""
struct NormalGammaPrevision <: Prevision
    κ::Float64
    μ::Float64
    α::Float64
    β::Float64

    function NormalGammaPrevision(κ::Float64, μ::Float64, α::Float64, β::Float64)
        κ > 0 || error("κ must be positive")
        α > 0 || error("α must be positive")
        β > 0 || error("β must be positive")
        new(κ, μ, α, β)
    end
end

"""
    ProductPrevision(factors::Vector) <: Prevision

Prevision for an independent joint over a product space. Holds the factors
as a Vector of Measure (not Vector of Prevision) — consumers access
`m.factors[i]` expecting a Measure; the shield forwards the vector by
reference, preserving shared-reference semantics.

Holding Measures-in-a-Prevision is the same pragmatic choice made for
TaggedBetaPrevision — allocates nothing on access; a future cleanup
can replace with Previsions + view reconstruction if perf justifies.
"""
struct ProductPrevision <: Prevision
    factors::Vector
end

"""
    MixturePrevision(components::Vector, log_weights::Vector{Float64}) <: Prevision

Prevision for a coherent convex combination of component previsions.
Holds components (as Vector of Measure — same pragmatic impurity as
ProductPrevision) and log-weights (normalised at construction).

**Shared-reference contract (load-bearing).** Both `components` and
`log_weights` are returned by reference through the `MixtureMeasure`
shield. Consumer code that does
`push!(state.belief.components, new_comp)` at
`apps/skin/server.jl:549,552` (and `push!(state.belief.log_weights,
new_lw)`) depends on this; breaking the contract by defensively copying
silently corrupts state (push! succeeds on the copy, original is
unchanged, no test-site error). The contract test in
`test/test_prevision_unit.jl` (`test_shared_reference_contract`)
asserts the invariant directly. See docs/posture-3/move-3-design.md §3
and R4 for the full rationale; future Moves 5/6 inherit this pattern
for MixturePrevision's own component updates and ParticlePrevision's
sample arrays.
"""
struct MixturePrevision <: Prevision
    components::Vector
    log_weights::Vector{Float64}

    function MixturePrevision(components::Vector, log_weights::Vector{Float64})
        length(components) == length(log_weights) || error("components and weights must match")
        length(components) > 0 || error("mixture must have at least one component")
        if all(lw -> lw == -Inf, log_weights)
            error("mixture has zero total mass — all components impossible")
        end
        max_lw = maximum(log_weights)
        log_total = max_lw + log(sum(exp.(log_weights .- max_lw)))
        new(components, log_weights .- log_total)
    end
end

"""
    ExchangeablePrevision(component_space::Space, prior_on_components::Prevision) <: Prevision

Declares exchangeability as a first-class type. An
`ExchangeablePrevision` represents an exchangeable sequence of
observations in `component_space` generated i.i.d. under a latent
parameter whose prior is `prior_on_components`. De Finetti's
representation theorem guarantees such a prevision decomposes as a
mixture of i.i.d. ergodic components; `decompose` computes that
mixture.

Move 5 ships this type with known-limited test coverage (simplest
representation-theorem case — Dirichlet prior over a Finite component
space, per §5.1 R3 scoping). Hierarchical / multi-component
correctness is the email-agent migration follow-up's burden, not
Move 5's.
"""
struct ExchangeablePrevision <: Prevision
    component_space  # Space
    prior_on_components::Prevision
end

"""
    decompose(p::ExchangeablePrevision) :: MixturePrevision

Apply the representation theorem: return the `MixturePrevision`
corresponding to the exchangeable `p`. Methods attach in `ontology.jl`
where the concrete component Measure types are in scope.
"""
function decompose end

"""
    ParticlePrevision(samples::Vector, log_weights::Vector{Float64}, seed::Int) <: Prevision

Move 6: typed carrier for importance-sampling posteriors. `samples`
holds the drawn hypotheses (one per particle; type depends on the
sampled Measure — Float64, tuples, Vectors, Symbols, etc.).
`log_weights` is the per-particle log importance weight, normalised at
construction via logsumexp. `seed` records the RNG seed that produced
the samples, for reproducibility auditing — not used by any computation,
but load-bearing for the seeded-MC `==` precedent (see
`docs/posture-3/precedents.md` §4).

**Shared-reference contract inherited from Move 3** (precedent #2):
`samples` and `log_weights` are returned by reference through the
`CategoricalMeasure` wrapper's `getproperty` shield; defensive copying
breaks the shared-reference semantics that downstream Measure-surface
readers depend on. See `test/test_prevision_particle.jl` — contract
tests confirm the shield.

At Move 6 Phase 2 the struct is declared additively: it exists as a
type but no code path constructs it yet. Phase 3 wires
`_condition_particle` to construct a `ParticlePrevision` wrapped by
`CategoricalMeasure(Finite(samples), log_weights)` for consumer-
surface compatibility. Phase 0's canonical-bit-invariance test is the
tripwire across the refactor.
"""
struct ParticlePrevision <: Prevision
    samples::Vector
    log_weights::Vector{Float64}
    seed::Int

    function ParticlePrevision(samples::Vector, log_weights::Vector{Float64}, seed::Int)
        length(samples) == length(log_weights) || error("particles and weights must match")
        length(samples) > 0 || error("particle set must be non-empty")
        if all(lw -> lw == -Inf, log_weights)
            error("particle set has zero total mass — all particles impossible")
        end
        max_lw = maximum(log_weights)
        log_total = max_lw + log(sum(exp.(log_weights .- max_lw)))
        new(samples, log_weights .- log_total, seed)
    end
end

"""
    QuadraturePrevision(grid::Vector{Float64}, log_weights::Vector{Float64}) <: Prevision

Move 6: typed carrier for deterministic grid-quadrature posteriors.
`grid` holds the deterministic quadrature points; `log_weights` the
per-point log weights (typically `log_density_at(prior, x) +
log_density(kernel, x, obs)`), normalised at construction via logsumexp.

No `seed` field — quadrature is deterministic (range over an interval
with fixed endpoints and fixed n), and reproducibility doesn't depend
on RNG state. Stratum-2 tolerance per `precedents.md` §4 is
`atol=1e-14`; the grid is bit-identical across refactors because the
range is a deterministic `collect(range(lo, hi; length=n))`.
"""
struct QuadraturePrevision <: Prevision
    grid::Vector{Float64}
    log_weights::Vector{Float64}

    function QuadraturePrevision(grid::Vector{Float64}, log_weights::Vector{Float64})
        length(grid) == length(log_weights) || error("grid and weights must match")
        length(grid) > 0 || error("quadrature grid must be non-empty")
        if all(lw -> lw == -Inf, log_weights)
            error("quadrature posterior has zero total mass")
        end
        max_lw = maximum(log_weights)
        log_total = max_lw + log(sum(exp.(log_weights .- max_lw)))
        new(grid, log_weights .- log_total)
    end
end

"""
    EnumerationPrevision(enumerated::Vector, log_weights::Vector{Float64}) <: Prevision

Move 6: typed carrier for exhaustive-enumeration posteriors (primarily
program-space enumeration per `src/program_space/enumeration.jl`).
`enumerated` holds the enumerated items (programs, AST shapes, etc.);
`log_weights` the per-item log weights, normalised at construction via
logsumexp.

Enumeration is deterministic under fixed iteration order; Stratum-2
tolerance per `precedents.md` §4 is `==`.
"""
struct EnumerationPrevision <: Prevision
    enumerated::Vector
    log_weights::Vector{Float64}

    function EnumerationPrevision(enumerated::Vector, log_weights::Vector{Float64})
        length(enumerated) == length(log_weights) || error("items and weights must match")
        length(enumerated) > 0 || error("enumeration set must be non-empty")
        if all(lw -> lw == -Inf, log_weights)
            error("enumeration posterior has zero total mass")
        end
        max_lw = maximum(log_weights)
        log_total = max_lw + log(sum(exp.(log_weights .- max_lw)))
        new(enumerated, log_weights .- log_total)
    end
end

"""
    ConditionalPrevision{E}(base::Prevision, event::E, mass::Float64) <: Prevision

Lazy wrapper representing `base | event` under de Finetti / Whittle's
conditional-prevision semantics (Move 7 introduced; Posture 4 Move 1
Invariant-2-tightened). Used as the §5.3 Option B fallback when no
closed-form event-restriction exists at the concrete (Prevision, Event)
pair.

Fields:
  - `base::Prevision` — the prior, unconditioned.
  - `event::E` — the conditioning event, carried as a declared Event
    subtype (TagSet, FeatureEquals, FeatureInterval, Conjunction,
    Disjunction, Complement). Posture 4 Move 1 replaced the prior
    `event_predicate::Function` opaque-closure representation with
    this typed field, closing the last Invariant-2 discipline hole in
    the Prevision module.
  - `mass::Float64` — cached `p(1_e) = expect(base, Indicator(e))`.
    Positive by construction; the constructing `condition(p, e::Event)`
    method guards against measure-zero events.

Evaluation: `expect(cp, f) = expect(base, f · 1_e) / mass`, where
`1_e = Indicator(cp.event)` — a declared TestFunction, not an opaque
closure. The product `f · 1_e` is evaluated via the TestFunction
hierarchy's composition (typically `LinearCombination` or a direct
indicator-gated evaluator).

Type-parameter discipline (Move 1 §5.1 Option A): `E` is unconstrained
at the struct level, matching `Indicator{E}`'s existing pattern. The
`<: Event` bound is enforced at the constructor signature of methods
in `ontology.jl` (where the Event hierarchy is in scope); runtime
construction via `ConditionalPrevision{NotAnEvent}(...)` type-checks at
the struct level but fails constructor dispatch. This is a theoretical
hole no working Julia code steps into — Indicator{E} has run through
Posture 3 Moves 7–8 with the same pattern without a single issue.

Type-stability caveat (move-7-design.md §5.3): `condition(p::Prevision,
e::Event)` returns either a specialised closed-form Prevision
(MixturePrevision restricted to firing tags, etc.) or a
`ConditionalPrevision` wrapper. Callers requiring a specific concrete
subtype must dispatch accordingly; generic-Prevision callers via
`expect` are uniform across both return shapes.
"""
struct ConditionalPrevision{E} <: Prevision
    base::Prevision
    event::E
    mass::Float64

    function ConditionalPrevision{E}(base::Prevision, event::E, mass::Float64) where E
        mass > 0 || error("ConditionalPrevision requires positive event mass; got $mass")
        new{E}(base, event, mass)
    end
end

# Outer constructor — lets callers write `ConditionalPrevision(base, event, mass)`
# and have the parameter `E` inferred from `event`'s type.
ConditionalPrevision(base::Prevision, event::E, mass::Float64) where E =
    ConditionalPrevision{E}(base, event, mass)

"""
    condition(p::Prevision, e) → Prevision

Event-form conditional prevision. Methods attach in `ontology.jl` where the
`Event` hierarchy is in scope. Move 7 §5.1 committed to Option B: this
Prevision-level form and the parametric `condition(p, k::Kernel, obs)` form
are peer primary primitives; neither derives from the other.

Closed-form specialisations exist for common pairs (e.g. `(MixturePrevision,
TagSet)`); the generic case returns a `ConditionalPrevision` lazy wrapper.
"""
function condition end

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
methods; Move 7 elevates `condition` to the Prevision level as a peer
primary primitive alongside the parametric `condition(p, k::Kernel, obs)`
form. See `docs/posture-3/move-7-design.md` §5.1 for the Option B
commitment: both forms are peer primaries at the Prevision level;
event-form and parametric-form are equivalent on deterministic events
(Di Lavore–Román–Sobociński "Partial Markov Categories", Proposition
4.9, arXiv:2502.03477); continuous-kernel general equivalence requires
disintegration (out of scope per master plan).
"""
function expect end

# ── ConjugatePrevision registry (Move 4) ──────────────────────────────────

"""
    ConjugatePrevision{Prior, Likelihood}(prior::Prior, likelihood::Likelihood)

Parametric wrapper representing a (Prior, Likelihood) pair for which a
closed-form posterior update exists. Moves the case-analytic conjugate
dispatch in `condition` into a type-structural registry: adding a pair
is adding one `update` method, not editing a branch in a central
function.

Per Move 4 design doc §5.1, the registry is method-based (Julia's
method table *is* the registry). `maybe_conjugate(p, k)` is the lookup
function; methods specialised on (Prior, Likelihood) pairs return a
`ConjugatePrevision` when the pair matches, `nothing` otherwise. `update`
is dispatched on the parametric type to apply the conjugate update in
closed form.

The `Likelihood` type parameter is the kernel's `likelihood_family`
type or a pair-specific marker (for pairs where the existing kernel
dispatches on `params` rather than `likelihood_family`).
"""
struct ConjugatePrevision{Prior, Likelihood} <: Prevision
    prior::Prior
    likelihood::Likelihood
end

"""
    maybe_conjugate(p::Prevision, k::Kernel) → Union{ConjugatePrevision, Nothing}

Registry lookup. Returns a `ConjugatePrevision{Prior, Likelihood}` if
the (prior, kernel) pair has a registered closed-form update; `nothing`
otherwise (caller falls through to particle/grid).

Default method returns `nothing`; specific methods on (Prior, Likelihood)
type pairs are declared in `ontology.jl` where the `Prior` and
`Likelihood` types are in scope. A `nothing` return is NOT an error —
it's the standard "not conjugate; use particle" path.

The TaggedBetaPrevision case specifically returns `nothing` as
transitional scaffolding until Move 5's `MixturePrevision` takes over
per-component routing. See `docs/posture-3/move-4-design.md` §3 and R3.
"""
maybe_conjugate(p::Prevision, k) = nothing

"""
    update(cp::ConjugatePrevision{Prior, Likelihood}, obs) → ConjugatePrevision

Apply the closed-form conjugate update for the (Prior, Likelihood) pair.
Returns a new `ConjugatePrevision` with the updated prior; caller
typically takes `.prior` to extract the posterior Prevision.

Specialised methods attach per (Prior, Likelihood) pair in `ontology.jl`.
No default method — calling `update` on an unregistered pair is a
MethodError, which is the expected behaviour (the registry should not
have returned a ConjugatePrevision without a matching update method).
"""
function update end

"""
    _dispatch_path(p::Prevision, k::Kernel) → Symbol

Observability hook (Move 4 §5.2; retained unchanged through Posture 4
Move 1 per move-1-design.md §5.3). Returns `:conjugate` if
`maybe_conjugate(p, k)` matches, `:particle` otherwise. Underscore-
prefixed per the repo conventions (see `CLAUDE.md` §Repo conventions §
Scope boundary): test-only surface, not a production API. The
Symbol-return discipline is load-bearing for the stratum-2 tests that
pin dispatch-path decisions — promoting to an `@enum DispatchPath`
would add structural type-work without discriminatory gain, and those
tests break loudly on any new return value regardless of enum-ness.

Used in Stratum-2 tests to pin dispatch-path decisions explicitly —
without it, a silent registry miss would fall through to particle and
produce the correct value for the wrong reason.
"""
_dispatch_path(p::Prevision, k) = maybe_conjugate(p, k) === nothing ? :particle : :conjugate

# ── Prevision-level mutation APIs (Move 2 Phase 1) ────────────────────────
#
# Per docs/posture-4/move-2-design.md §2. Land ahead of Move 7's skin
# rewrite as surface-ready migration targets. Not consumed by `src/`
# internally today. The Move 5-7 rewrite of apps/skin/server.jl:611-614
# migrates `push!(state.belief.components, ...)` to `push_component!`
# and `push!(state.belief.log_weights, ...)` is subsumed by
# push_component!'s atomic component+log_weight append.

"""
    push_component!(p::MixturePrevision, c::Prevision, log_weight::Float64)

Append a component to a `MixturePrevision` atomically — appends `c` to
`p.components` and `log_weight` to `p.log_weights`, then re-normalises
`p.log_weights` in place via log-sum-exp. Mutates `p` (both fields);
returns `p`.

This is the canonical mutation API for Move 2+; direct `push!`-through-
shield patterns on `m.components` fail loudly via `FrozenVectorView`
guards at the Measure-level shield (see `src/ontology.jl`).

Note: mutates `p.components` and `p.log_weights` in place. The prior's
normalisation is recomputed from scratch after the append (not
incrementally), so callers get a fresh normalised log-weight vector.
"""
function push_component!(p::MixturePrevision, c::Prevision, log_weight::Float64)
    push!(getfield(p, :components), c)
    push!(getfield(p, :log_weights), log_weight)
    # Re-normalise in place
    lws = getfield(p, :log_weights)
    if all(lw -> lw == -Inf, lws)
        error("mixture has zero total mass after push_component! — all components impossible")
    end
    max_lw = maximum(lws)
    log_total = max_lw + log(sum(exp.(lws .- max_lw)))
    for i in eachindex(lws)
        lws[i] -= log_total
    end
    return p
end

"""
    replace_component!(p::MixturePrevision, i::Int, c::Prevision)

Replace the `i`-th component of `p` in place with `c`. Mutates `p.components[i]`;
does not change `p.log_weights`. Returns `p`.

Use this for Move 7's skin-server component-update paths where the
existing pattern was `state.belief.components[i] = new_comp` — the
`FrozenVectorView` guard rejects `setindex!` on the reconstructed shield
vector, and this API is the migration target.
"""
function replace_component!(p::MixturePrevision, i::Int, c::Prevision)
    components = getfield(p, :components)
    1 <= i <= length(components) || error("replace_component!: index $i out of range [1, $(length(components))]")
    components[i] = c
    return p
end

end # module Previsions

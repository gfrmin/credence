# RSS Preference Learning: Design Document

## 1. The Problem

A single user reads RSS feeds through a Miniflux + sidecar application. The sidecar already collects implicit signals:

- **Read events**: (entry_id, feed_id, read_at) — the user opened an article
- **Stars**: the user bookmarked an article (strong positive)
- **Shares**: the user generated a share link (very strong positive)
- **Exposures**: which articles appeared in the user's entry list (once tracked)
- **Skips**: articles exposed but not read within a window (implicit negative)

The sidecar also enriches articles with:

- **LLM tags**: 1–5 topic labels per article (e.g., "ai", "linux", "privacy")
- **Embeddings**: dense vectors from `nomic-embed-text` (stored in `article_embeddings`)
- **Summaries**: 2–3 sentence LLM summaries
- **Content structure**: word count, headings, images, code blocks, media
- **Feed metadata**: priority (1–3, user-set), category, update frequency, error rate

The goal: learn the user's article preferences from these signals, rank the entry list accordingly, and use pairwise comparisons as an active elicitation channel to accelerate learning.

---

## 2. Framing in Credence's Ontology

### 2.1 This is a CIRL problem

The user has a latent preference parameter θ. The agent (preference service) is uncertain about θ. The user's behaviour — reading, starring, skipping, comparing — is the signal. The agent's utility equals the user's utility (SPEC §1.4): rank articles so the user sees what they want.

The observation model P(behaviour | θ) is itself uncertain — "user read article" might mean genuine interest, boredom scrolling, or accidental click. The model should discover which signals are informative (SPEC §4.2), not hardcode their meaning.

### 2.2 The hypothesis space

Each hypothesis θ is a **preference program** — a function from article features to a real-valued score. These compete under the complexity prior P(θ) = 2^(-|θ|).

Simple programs:
```scheme
;; "I read everything from this feed"
(gt :feed_is_hackernews 0.5)

;; "I like AI articles"
(gt :tag_ai 0.5)
```

Medium programs:
```scheme
;; "I like short AI articles from specific feeds"
(and (gt :tag_ai 0.5)
     (lt :log_word_count 0.6)
     (gt :feed_read_rate 0.3))
```

Complex programs:
```scheme
;; "Tech news in the morning, long-form on weekends"
(if (and (gt :hour_sin 0.5) (lt :is_weekend 0.5))
    (and (gt :tag_technology 0.5) (lt :log_word_count 0.5))
    (gt :log_word_count 0.7))
```

Each program's score for an article determines its prediction for the user's behaviour. Programs that predict well gain posterior weight; those that don't lose weight. The complexity prior penalises long programs, implementing Occam's razor.

**MAUT as a program family**: A multi-attribute utility decomposition U(x) = Σ w_i · u_i(x_i) is one structural hypothesis in the mixture. It's a moderately complex program that references all attribute dimensions. If the user's preferences are actually decomposable, MAUT programs will dominate the mixture. If not, simpler or richer programs will win. This is what the SPEC means by "discovered not assumed" (§3.4, §10.2).

### 2.3 The named feature dictionary

Following SPEC §6.7, features are a `Dict{Symbol, Float64}`, not a positional vector. The RSS domain provides:

**Source features** (per-article, from feed identity):
```
:feed_is_{feed_name}     — one-hot per feed (e.g., :feed_is_hackernews)
:category_is_{cat_name}  — one-hot per category
:feed_priority           — user-set priority (normalised: 1→1.0, 2→0.5, 3→0.0)
:feed_read_rate          — fraction of this feed's articles user has historically read
:feed_star_rate          — fraction of this feed's articles user has starred
```

**Content features** (per-article, from LLM enrichment):
```
:tag_{tag_name}          — binary, 1.0 if article has this tag (e.g., :tag_ai, :tag_linux)
:pca_{0..k}             — PCA-projected embedding components (top-k principal directions
                           computed across all article embeddings; captures semantic content
                           without requiring a feature per embedding dimension)
```

**Temporal features**:
```
:hour_sin, :hour_cos     — cyclical encoding of publication hour
:dow_sin, :dow_cos       — cyclical encoding of day-of-week
:is_weekend              — binary
:log_age_hours           — log(hours since publication), normalised
:log_feed_gap_hours      — log(hours since last read from same feed), normalised
```

**Meta features** (per-article, from content structure):
```
:log_word_count          — normalised log word count
:log_reading_time        — normalised log estimated reading time
:has_images              — binary
:has_audio               — binary (podcast/audio enclosure)
:has_code                — binary (code blocks in content)
:heading_count_norm      — normalised heading count (structure depth)
:n_tags_norm             — normalised number of LLM tags (topic breadth)
:version_count           — number of content revisions (>1 = article was updated)
```

**Interaction features** (per-article, require computation against user history):
```
:max_sim_recent          — max cosine similarity to embeddings of recently-read articles
:mean_tag_affinity       — mean historical read rate across this article's tags
:articles_read_today     — normalised count of articles read today (saturation signal)
```

All features are normalised to [0, 1] where possible so that `(gt :feature threshold)` predicates have consistent semantics across features.

### 2.4 The observation model (kernel)

Observations come through multiple channels. Each channel has a kernel encoding P(observation | θ, article_features).

**Channel 1: Pairwise comparisons (Bradley-Terry)**

The user is shown two articles (A, B) and chooses which they'd prefer to read. The likelihood under preference program θ:

$$P(A \succ B \mid \theta) = \sigma(\text{score}_\theta(A) - \text{score}_\theta(B))$$

where score_θ(·) is the program's output on the article's feature dictionary and σ is the logistic sigmoid. This is the standard Bradley-Terry model (SPEC §3.3).

In Credence:
```julia
obs_space = Finite([1.0, 0.0])  # A wins, B wins

comparison_kernel = Kernel(
    hypothesis_space,
    obs_space,
    _ -> error("generate not used"),
    (θ, o) -> begin
        score_a = evaluate_program(θ, features_a)
        score_b = evaluate_program(θ, features_b)
        diff = score_a - score_b
        o == 1.0 ? log_sigmoid(diff) : log_sigmoid(-diff)
    end,
    nothing, nothing
)
```

**Channel 2: Read events (Bernoulli)**

The user read an article. This is a softer signal than a pairwise comparison — it could mean genuine interest or habitual scrolling.

$$P(\text{read} \mid \theta, \text{features}) = \sigma(\text{score}_\theta(\text{features}) - \mu_{\text{read}})$$

where μ_read is a threshold parameter (also learned). Articles with high scores under θ are more likely to be read.

**Channel 3: Stars (weighted Bernoulli)**

Same structure as read events but with higher Fisher information — starring is a deliberate positive act. Equivalent to multiple read events in likelihood contribution.

**Channel 4: Skips (negative Bernoulli)**

Article was exposed in the entry list but not read. This is a weak negative signal, confounded by many factors (user didn't scroll that far, was busy, etc.). The likelihood is tempered:

$$P(\text{skip} \mid \theta, \text{features}) = [1 - \sigma(\text{score}_\theta(\text{features}) - \mu_{\text{read}})]^{w_{\text{skip}}}$$

where w_skip < 1 tempers the signal (SPEC §4.1 discusses tempered likelihoods for unreliable channels).

**Fisher information ordering**: The channels are naturally ordered by informativeness:

comparison > star > read > skip

Pairwise comparisons have the highest Fisher information because they directly constrain the preference ordering. Stars are unambiguous positive signals. Read events are moderately informative. Skips are weakly informative and confounded.

Critically, these weights are NOT manually tuned — they are properties of the likelihood functions. A comparison between articles A and B updates every program that makes different predictions for A vs B. A read event only updates programs relative to the read threshold. The mathematics of conditioning (SPEC §4.1) handles calibration automatically for correctly specified likelihoods.

### 2.5 Actions and VOI

The agent's action space for the comparison interface includes:

```
:show_pair(A, B)    — present this pair for comparison
:rank_and_stop      — stop asking, show the ranked entry list
```

There is no separate VOI computation (SPEC §5.2). `:show_pair(A, B)` is an action whose EU includes the expected information gain minus the user's attention cost. The agent asks when asking is more valuable than stopping:

$$EU(\text{show\_pair}(A, B)) = \text{EVOI}(A, B) - c_{\text{attention}}$$
$$EU(\text{rank\_and\_stop}) = \text{value}(\text{current\_belief})$$

The agent shows pairs as long as the information gain exceeds the attention cost, then stops. This means the number of comparisons per session is **not a fixed parameter** — it emerges from the EU comparison. Early on (high posterior entropy), many pairs are valuable. After convergence, few or none are.

---

## 3. What Credence Needs

### 3.1 No axiom changes

Everything described above uses existing Credence primitives:
- Spaces: `Finite`, `Interval`, `ProductSpace`, `Simplex`
- Measures: `CategoricalMeasure`, `BetaMeasure`, `DirichletMeasure`, `GaussianMeasure`, `ProductMeasure`, `MixtureMeasure`
- Functions: `condition`, `expect`, `optimise`, `value`, `voi`, `draw`

The GaussianMeasure is scalar-only, but this is handled by ProductMeasure of scalar Gaussians (the same pattern used throughout Credence for per-category reliability tracking via ProductMeasure of Betas).

### 3.2 New code needed in Credence

**A. RSS feature extraction and connection definition**

A new "connection" (SPEC §6.3) for the RSS domain that defines the feature dictionary and action vocabulary. This lives outside the frozen layer.

```julia
struct RSSConnection
    feature_set::Set{Symbol}       # all named features
    feed_names::Vector{Symbol}     # feed identity features
    tag_names::Vector{Symbol}      # tag features
    pca_components::Matrix{Float64} # for embedding projection
    pca_mean::Vector{Float64}
end

function extract_features(conn::RSSConnection, article::Dict) :: Dict{Symbol, Float64}
    # Assembles the full named-feature dictionary from article data
end
```

**B. Bradley-Terry kernel constructor**

A helper that constructs a BT kernel given two feature dictionaries. Not a new axiom-constrained function — just a convenience for building kernels.

```julia
function bt_kernel(hypothesis_space, features_a::Dict, features_b::Dict)
    Kernel(hypothesis_space, Finite([1.0, 0.0]),
        _ -> error("generate not used"),
        (θ, o) -> begin
            diff = evaluate(θ, features_a) - evaluate(θ, features_b)
            o == 1.0 ? log_sigmoid(diff) : log_sigmoid(-diff)
        end,
        nothing, nothing)
end
```

**C. JSON serialization for all Measure types**

The HTTP service needs to send model state as JSON. Round-trip encode/decode for: `CategoricalMeasure`, `BetaMeasure`, `GaussianMeasure`, `DirichletMeasure`, `ProductMeasure`, `MixtureMeasure`, and all Space types.

**D. HTTP service wrapper**

A Julia HTTP server that exposes Credence operations over REST. Endpoints:

| Endpoint | Input | Output | Credence operation |
|----------|-------|--------|--------------------|
| `POST /init` | feed_ids, tags, pca_matrix | model state | Construct initial prior |
| `POST /condition/comparison` | winner_features, loser_features | updated model state | `condition(belief, bt_kernel, obs)` |
| `POST /condition/implicit` | features, signal_type, weight | updated model state | `condition(belief, implicit_kernel, obs)` |
| `POST /rank` | list of article features | scored + sorted articles | `draw(belief)` + evaluate |
| `POST /select-pairs` | candidate article features, budget | pairs ordered by EU | `voi` / EU comparison |
| `GET /diagnostics` | — | top programs, weights, entropy | `weights(belief)`, posterior analysis |
| `POST /save` | — | — | Serialize to PostgreSQL |
| `POST /load` | — | model state | Deserialize from PostgreSQL |

**E. PCA utility** (trivial — LinearAlgebra.svd)

Compute PCA projection matrix from article embeddings. Called at init time and periodically refreshed.

### 3.3 Approach selection: program space vs parametric MAUT

There are two approaches, and the design should support both.

**Approach 1: Tier 2 program-space inference**

Use Credence's existing program enumeration (Grammar → enumerate_programs → compile_kernel) with RSS-specific features. Each program is a preference hypothesis. The CategoricalMeasure over programs IS the belief state. Conditioning updates program weights. Grammar evolution discovers useful abstractions.

Pros:
- Most principled — programs compete under the complexity prior
- MAUT decomposition is discovered, not assumed
- Handles non-linear, non-decomposable preferences
- Grammar evolution discovers reusable concepts ("urgent tech news")

Cons:
- Enumeration at useful depth may be expensive with ~50+ features
- Existing program-space code is oriented toward the email domain (feature names, action vocabulary)
- Requires careful feature normalisation so thresholds are meaningful

**Approach 2: Parametric MAUT with conjugate priors**

Hardcode the MAUT structure: U(x) = Σ w_k · u_k(x_k). Use Dirichlet for attribute weights, ProductMeasure of Betas/Gaussians for per-attribute parameters. Conditioning uses conjugate fast paths where possible.

Pros:
- Simpler to implement and debug
- Conjugate updates are fast and exact
- Interpretable (can show "you prefer content 40%, source 30%, ...")
- Fewer features per attribute = tractable enumeration within each attribute

Cons:
- Assumes mutual preferential independence (may not hold)
- Cannot discover non-linear interactions across attributes
- Hardcoded structure is the antithesis of "discovered not assumed"

**Recommended: Start with Approach 2, migrate to Approach 1**

Approach 2 is a legitimate first hypothesis — it's the simplest model consistent with the MAUT assumption, and the SPEC acknowledges that MAUT is often a good structural constraint (§3.4). It gets a working system up quickly and provides a concrete baseline.

The migration path: once Approach 2 is running, wrap each MAUT configuration as a **single program** in a Tier 2 mixture. Then add non-MAUT programs alongside it. The complexity prior determines whether the MAUT structure or simpler/richer programs win. This is exactly how the SPEC envisions it.

### 3.4 Conditioning strategy for Approach 2

The MAUT model is a ProductMeasure:

```julia
belief = ProductMeasure([
    DirichletMeasure(Simplex(5), attributes, [2,2,1,1,1]),  # attribute weights
    content_belief,      # ProductMeasure of scalar Gaussians (PCA dims)
    source_belief,       # ProductMeasure of per-feed Betas
    temporal_belief,     # ProductMeasure of scalar Gaussians
    meta_belief,         # ProductMeasure of scalar Gaussians
    interaction_belief,  # ProductMeasure of scalar Gaussians
])
```

**The conditioning problem**: A pairwise comparison updates ALL components simultaneously. The BT likelihood depends on the full MAUT score, which is a function of all attribute weights and per-attribute parameters. There is no conjugate relationship between this likelihood and the ProductMeasure prior.

**Available conditioning strategies** (all are valid implementations of Bayes' rule — they differ in computational accuracy, not mathematical correctness):

**Strategy A: Full Monte Carlo conditioning**

Use the existing fallback `condition` for non-conjugate cases. Draw N particles from the ProductMeasure prior, weight each by the BT likelihood, normalise.

```julia
posterior = condition(belief, bt_kernel, observation)
# → CategoricalMeasure over N particles
```

Pro: Exact up to sampling error. No approximations.
Con: Loses the conjugate structure. After one conditioning, the belief is a CategoricalMeasure, not a ProductMeasure. Subsequent updates require particle methods. Particle depletion over many updates.

**Strategy B: Factored variational conditioning**

Approximate the joint posterior as a product of independent marginals (mean-field variational Bayes). After observing A ≻ B:

1. Compute the score difference under current means: Δ_k = E[u_k(A)] - E[u_k(B)] for each attribute k
2. Compute total predicted difference: Δ = Σ E[w_k] · Δ_k
3. Compute the Bradley-Terry prediction error: e = observation - sigmoid(Δ)
4. For each attribute k, compute its "credit": c_k = E[w_k] · |Δ_k| / Σ_j E[w_j] · |Δ_j|
5. Update each attribute's parameters proportionally to its credit:
   - Source Betas: if feed A won, condition A's feed Beta on success, B's on failure
   - Gaussian weights: shift mean toward the sign of Δ_k for the winning side
   - Dirichlet: increment the concentration of the attribute with the largest |Δ_k| for the winner

This is a mean-field approximation. It's not exact Bayesian conditioning, but it preserves conjugate structure and avoids particle depletion. It's the approach used in host_helpers.jl for per-category reliability updates (where the categorical selects which Beta to update, and the MixtureMeasure tracks the posterior over which category was active).

Pro: Preserves ProductMeasure structure. O(1) per update. No particle depletion.
Con: Mean-field approximation ignores correlations between attributes. Credit assignment is heuristic.

**Strategy C: Gibbs-style sequential conditioning**

Condition one factor at a time, cycling through the ProductMeasure:

1. Fix all parameters except attribute weights. Condition Dirichlet on "which attribute explained this comparison?" (treating it as a categorical observation).
2. Fix attribute weights. For each attribute k, condition its parameters on the comparison weighted by w_k.
3. Repeat until convergence (or just one pass per observation for online updating).

This is a valid inference strategy (Gibbs sampling converges to the true posterior). Each step uses conjugate conditioning. The FactorSelector pattern (ontology.jl:538-581) already handles "condition one factor while fixing others."

Pro: Each step is exact conjugate conditioning. Preserves structure. Well-motivated by Gibbs.
Con: Multiple conditioning steps per observation. Convergence not guaranteed in single-pass mode.

**Recommended: Strategy C** for the initial implementation. It's the most principled option that preserves conjugate structure. Each step is a genuine `condition` call — no approximations, no second learning mechanism. The FactorSelector pattern already exists for exactly this purpose.

### 3.5 Thompson sampling for ranking

Thompson sampling is `draw` in the host + evaluation in ordinary computation (per the rejected-pattern in CLAUDE.md — "Construct the posterior in the DSL, call draw() in the host"):

```julia
# Host code (server.jl):
θ_sample = draw(belief)              # sample from posterior ProductMeasure
scores = Dict{Int, Float64}()
for article in articles
    features = extract_features(conn, article)
    scores[article["id"]] = evaluate_maut(θ_sample, features)
end
ranked = sort(collect(scores), by=x -> -x[2])
```

Each call to `draw(ProductMeasure)` independently samples each factor:
- DirichletMeasure → sample weight vector on simplex
- BetaMeasure → sample from Beta
- GaussianMeasure → sample from Normal

This naturally produces ranking variation (exploration). Articles from uncertain feeds or with uncertain tag affinity sometimes get lucky draws, generating the data needed to reduce uncertainty.

### 3.6 Active pair selection via VOI

The existing `voi` in stdlib.bdsl works for finite observation spaces. For pair selection:

```julia
# For each candidate pair (A, B):
pair_kernel = bt_kernel(belief.space, features_a, features_b)
pair_voi = voi(belief, pair_kernel, ranking_actions, ranking_pref, [1.0, 0.0])

# Show the pair with highest VOI - attention_cost
```

Since VOI computation requires `expect` over the belief (which uses MC for ProductMeasure with n=1000 samples), and we may have O(N²) candidate pairs, a fast proxy is needed:

**Fast EVOI proxy**: For each pair, compute:
1. p = sigmoid(E[U(A)] - E[U(B)]) using posterior means
2. EVOI_proxy = -p·log(p) - (1-p)·log(1-p) (binary entropy)

Then compute full VOI only for the top-20 pairs by this proxy. This is not a second decision mechanism — it's a computational optimisation for the VOI calculation, analogous to how Credence uses grid quadrature vs MC sampling based on measure type.

---

## 4. New Files

All new code lives outside the frozen layer. No changes to `ontology.jl`, `eval.jl`, or `stdlib.bdsl`.

### In the Credence repository (`~/git/bayesian-stuff/credence/`)

**`src/json_serialization.jl`** (~150 lines)

JSON encode/decode for all Space and Measure types. Uses JSON3.jl. Required for the HTTP service to communicate with the Python sidecar.

Functions:
- `to_json(s::Space) → Dict` / `from_json_space(d::Dict) → Space`
- `to_json(m::Measure) → Dict` / `from_json_measure(d::Dict) → Measure`
- Round-trip property: `from_json_measure(to_json(m))` reconstructs m exactly

**`src/pca.jl`** (~30 lines)

PCA via LinearAlgebra.svd. Two functions:
- `compute_pca(embeddings::Matrix{Float64}, k::Int) → (components, mean_vec)`
- `project(embedding::Vector{Float64}, components, mean_vec) → Vector{Float64}`

**`src/Credence.jl`** (modify — add includes and exports)

Add `include("json_serialization.jl")`, `include("pca.jl")`.
Export: `to_json`, `from_json_space`, `from_json_measure`, `compute_pca`, `project`.

### In the RSS feed repository (`~/git/rssfeed/`)

**`credence-service/`** — new directory

**`credence-service/rss_connection.jl`** (~120 lines)

The RSS "connection" (SPEC §6.3). Defines:
- `RSSConnection` struct: feature_set, feed_names, tag_names, pca_components, pca_mean
- `extract_features(conn, article_data) → Dict{Symbol, Float64}`
- Feature normalisation (all features to [0,1])

**`credence-service/preference.jl`** (~300 lines)

The preference model, built from Credence Tier 1 primitives:

```julia
module RSSPreference

using Credence

struct PreferenceState
    belief::ProductMeasure           # the full MAUT posterior
    conn::RSSConnection              # feature extraction config
    read_threshold::GaussianMeasure  # μ_read for implicit signals
    n_comparisons::Int
    n_implicit::Int
end

# Construction
function create_initial(feed_ids, category_map, tag_names, pca_components, pca_mean)
    → PreferenceState with uniform priors

# Conditioning (Gibbs-style sequential — Strategy C)
function condition_comparison(state, winner_features, loser_features)
    → new PreferenceState
    # 1. Condition Dirichlet on which attribute "explained" the comparison
    # 2. For each attribute, condition its parameters via FactorSelector
    # Both steps are genuine condition() calls

function condition_implicit(state, features, signal, weight)
    → new PreferenceState
    # Condition source Betas on read/skip (conjugate)
    # Condition read_threshold on observation (conjugate)

# Ranking
function rank(state, articles; thompson=true)
    → [(entry_id, score, contributions)]
    # If thompson: draw(state.belief) + evaluate
    # Else: use posterior means

# Pair selection
function select_pairs(state, candidates, n, attention_cost)
    → [(left, right, eu)]
    # Fast proxy for EVOI, then full VOI for top candidates

# Diagnostics
function diagnostics(state)
    → Dict with attribute weights, top features, posterior entropy, counts

end
```

**`credence-service/server.jl`** (~200 lines)

HTTP server using HTTP.jl. Routes map to `RSSPreference` functions:

```
POST /init                     → create_initial
POST /condition/comparison     → condition_comparison
POST /condition/implicit       → condition_implicit
POST /rank                     → rank
POST /select-pairs             → select_pairs
GET  /diagnostics              → diagnostics
POST /save                     → serialize state to PostgreSQL
POST /load                     → deserialize state from PostgreSQL
```

State is held in-memory (single user, single server). Persisted to PostgreSQL on /save (triggered by the sidecar after each conditioning call, or periodically).

**`credence-service/Dockerfile`**

```dockerfile
FROM julia:1.11-bookworm
WORKDIR /app

# Install dependencies
RUN julia -e 'using Pkg; Pkg.add(["HTTP", "JSON3", "LibPQ"])'

# Copy Credence source (mounted or copied from ~/git/bayesian-stuff/credence/src/)
COPY credence-src/ credence/src/

# Copy service code
COPY *.jl ./

# Precompile
RUN julia --project=. -e 'push!(LOAD_PATH, "credence/src"); using Credence; using HTTP; using JSON3'

EXPOSE 8081
CMD ["julia", "--project=.", "server.jl"]
```

Note: The Credence source needs to be available at build time. Options:
1. Git submodule pointing to `~/git/bayesian-stuff/credence/src/`
2. Volume mount in docker-compose (for development)
3. Copy at build time (for production)

**`docker-compose.yml`** (modify — add credence service)

```yaml
  credence:
    build:
      context: ./credence-service
    environment:
      - DATABASE_URL=postgres://miniflux:${POSTGRES_PASSWORD:-miniflux}@db/miniflux?sslmode=disable
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped
```

The sidecar gets `CREDENCE_URL=http://credence:8081`.

---

## 5. Open Questions

### 5.1 Feature set size and enumeration depth

With ~50+ named features, even depth-2 enumeration produces O(50²) = 2500 programs. Depth-3 produces O(50³) = 125,000. The existing program-space inference handles this via grammar-based enumeration (only features in the grammar's feature_set are used), but the initial feature_set determination for a new domain needs thought.

Possible approach: start with a small feature set (feed identity + top-10 tags + basic meta features ≈ 25 features) and expand via grammar evolution as the agent discovers which features are informative.

### 5.2 PCA stability

PCA components change as new articles arrive. If the projection matrix shifts, previously computed feature vectors become inconsistent. Options:
- Recompute PCA rarely (weekly) and accept small inconsistencies
- Incremental PCA that adjusts smoothly
- Fixed PCA computed once on a large initial corpus

### 5.3 Cold start

Before any comparisons, the model has only implicit signals (read_events, stars). These bootstrap the source Betas (per-feed read rates) well, but provide weak signal for content/temporal/meta/interaction weights.

The VOI framework handles this naturally: early on, the EU of showing comparisons is high because posterior entropy is high. The agent should aggressively request comparisons until the preference model converges.

### 5.4 Attention cost calibration

The attention cost c_attention determines how many comparisons the agent requests. Too low → the agent asks too many questions. Too high → it never asks. This parameter should itself be learned from user behaviour (do they complete the comparison session or abandon it?), but initially it needs a reasonable default.

### 5.5 Migration to Tier 2 program space

The Approach 2 → Approach 1 migration involves:
1. Wrapping each MAUT configuration as a Program in the grammar
2. Adding non-MAUT programs alongside
3. Using CategoricalMeasure over programs instead of the parametric ProductMeasure
4. Using grammar evolution to discover useful abstractions

This is a significant architectural change but doesn't require any new axiom-constrained functions. The existing Tier 2 machinery (grammar, enumerate, compile, AgentState) should work with RSS features after adapting the feature names and action vocabulary.

---

## 6. Verification

### Unit tests (in `~/git/bayesian-stuff/credence/test/`)

**`test_json_serialization.jl`**: Round-trip all Measure types through JSON.

**`test_rss_preference.jl`**:
- Create initial state with 5 feeds and 10 tags
- Condition on 20 synthetic comparisons where Feed A articles always win
- Verify: Feed A's Beta α >> β, Dirichlet weight on source attribute has increased
- Condition on 20 synthetic comparisons where tag "ai" articles always win
- Verify: content PCA weights have shifted (if ai-tagged articles cluster in PCA space)
- Verify: ranking places Feed A + tag "ai" articles at top

**`test_bt_kernel.jl`**:
- Verify log_sigmoid(Δ) for known inputs
- Verify `condition(uniform_prior, bt_kernel, 1.0)` shifts weights toward hypothesis that predicts A > B

### Integration tests (require running service)

- Start service, POST /init, POST 10 comparisons, GET /diagnostics → verify posterior has shifted from prior
- POST /rank with 20 articles → verify ordering reflects comparison data
- POST /select-pairs → verify returned pairs have high EVOI (not random)
- POST /save, restart service, POST /load → verify state restored exactly

### Smoke test against real data

- Load actual read_events from PostgreSQL
- POST /condition/implicit for each read event
- POST /rank for 50 recent unread articles
- Manual inspection: does the ranking look sensible given reading history?

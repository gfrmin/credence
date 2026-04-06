# RSS Credence Service — Integration Instructions

These instructions describe how to connect the Credence RSS ranking service
(Julia, in this repo) with the rssfeed sidecar (Python, at `~/git/rssfeed`).

## Architecture

```
┌──────────┐     ┌──────────┐     ┌──────────────┐
│ Miniflux │────▶│ Sidecar  │────▶│  Credence    │
│  :9144   │     │  :9145   │     │  RSS Service │
└────┬─────┘     └────┬─────┘     │  :8081       │
     │                │           └──────┬───────┘
     └────────┬───────┘                  │
              ▼                          │
         ┌─────────┐                     │
         │ Postgres │◀────────────────────┘
         │   :5432  │  (read-only queries)
         └─────────┘
```

All three services share the same PostgreSQL database. The credence service
reads from Miniflux tables (entries, feeds, categories) and sidecar tables
(read_events, article_tags, article_snapshots, feed_config) to learn
preferences and rank articles.

## Changes needed in `~/git/rssfeed`

### 1. Add `CREDENCE_URL` to sidecar config

**File:** `sidecar/app/config.py`

```python
CREDENCE_URL = os.environ.get("CREDENCE_URL", "http://credence:8081")
```

### 2. Modify entry list to use credence ranking

**File:** `sidecar/app/routes/entries.py`, in the `entries_list` function (~line 109-126)

Replace the priority-based sorting with a call to the credence service. Fall
back to priority sort if the service is unavailable.

```python
import httpx
from app.config import CREDENCE_URL

async def _get_credence_ranking(n: int = 200) -> dict[int, float] | None:
    """Fetch ranking scores from credence service. Returns {entry_id: score} or None."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{CREDENCE_URL}/rank", params={"n": n})
            if resp.status_code == 200:
                data = resp.json()
                return {item["entry_id"]: item["score"] for item in data["ranking"]}
    except Exception:
        pass
    return None
```

In the entry list handler, after fetching entries:

```python
# Try credence ranking, fall back to priority sort
scores = await _get_credence_ranking(n=len(all_entries))
if scores:
    all_entries.sort(key=lambda e: -scores.get(e["id"], 0.0))
else:
    # Existing priority-based sort (keep as fallback)
    ...
```

### 3. Notify credence service on read events

**File:** `sidecar/app/routes/entries.py`, in `entry_detail` (~line 186)

After inserting the read_event, fire-and-forget a train request:

```python
# After: await conn.execute("INSERT INTO read_events ...")
try:
    async with httpx.AsyncClient(timeout=2.0) as client:
        await client.post(f"{CREDENCE_URL}/train")
except Exception:
    pass  # Non-critical — credence will catch up on next train
```

### 4. Notify credence service on dismiss (mark-as-read)

**File:** `sidecar/app/routes/entries.py`, in `mark_read` (~line 408)

```python
# After: await miniflux_client.update_entry_status([entry_id], "read")
try:
    async with httpx.AsyncClient(timeout=2.0) as client:
        await client.post(f"{CREDENCE_URL}/train")
except Exception:
    pass
```

### 5. Add credence service to docker-compose

**File:** `docker-compose.yml`

```yaml
  credence:
    build:
      context: ./credence-service
    environment:
      DATABASE_URL: postgres://miniflux:${POSTGRES_PASSWORD:-miniflux}@db/miniflux?sslmode=disable
      CREDENCE_PORT: "8081"
    depends_on:
      db:
        condition: service_healthy
    ports:
      - "9146:8081"
    restart: unless-stopped
```

Add to sidecar service environment:
```yaml
  sidecar:
    environment:
      CREDENCE_URL: http://credence:8081
      # ... existing env vars ...
```

### 6. Create credence-service directory

**Directory:** `credence-service/`

**`credence-service/Dockerfile`:**
```dockerfile
FROM julia:1.11-bookworm
WORKDIR /app

# Install Julia dependencies
RUN julia --project=. -e 'using Pkg; Pkg.add(["HTTP", "JSON3", "LibPQ"]); Pkg.precompile()'

# Copy Credence DSL source
COPY credence/src/ /app/src/

# Copy RSS domain
COPY credence/domains/rss/ /app/domains/rss/

# Precompile
RUN julia -e 'push!(LOAD_PATH, "src"); include("domains/rss/server.jl")'

EXPOSE 8081
CMD ["julia", "domains/rss/server.jl"]
```

**Build context:** The Dockerfile expects the Credence source to be available.
For development, use volume mounts instead of COPY:

```yaml
  credence:
    image: julia:1.11-bookworm
    working_dir: /app
    command: julia domains/rss/server.jl
    volumes:
      - /path/to/bayesian-stuff/credence/src:/app/src:ro
      - /path/to/bayesian-stuff/credence/domains/rss:/app/domains/rss:ro
    environment:
      DATABASE_URL: postgres://miniflux:${POSTGRES_PASSWORD:-miniflux}@db/miniflux?sslmode=disable
    depends_on:
      db:
        condition: service_healthy
    ports:
      - "9146:8081"
```

### 7. Add httpx to sidecar dependencies

The sidecar needs `httpx` for async HTTP calls to the credence service.
Add to `sidecar/requirements.txt` (or `pyproject.toml`):

```
httpx>=0.27
```

## API Reference

### POST /init
Initialize the agent from database. Called automatically on startup.

**Response:** `{"status": "initialized", "n_programs": 1234, "n_features": 25, "n_feeds": 10, "n_tags": 50}`

### POST /train
Condition on recent read/dismiss events since last training.

**Optional body:** `{"since": "2026-04-01T00:00:00"}`

**Response:** `{"n_reads": 5, "n_dismissals": 2, "n_sessions": 1, "n_components": 1500}`

### GET /rank?n=50
Rank unread articles by predicted interest.

**Response:** `{"ranking": [{"entry_id": 123, "score": 0.75}, ...], "total_unread": 200}`

### GET /health
Service status and diagnostics.

**Response:** `{"status": "ok", "n_components": 1500, "top_grammar_id": 3, "last_trained": "2026-04-06T12:00:00"}`

## How it works

1. **Programs are predicates** that fire on "interesting" articles (e.g., `has_code > 0.5 AND feed_is_hn > 0.5`)
2. **Each reading event is a Plackett-Luce observation**: "user chose article A from all unread articles"
3. **Dismissals** (mark-as-read without opening) are negative signals
4. **Conditioning** updates which programs predict well — programs that fire on articles read early gain weight
5. **Ranking** = sort by posterior expected score: articles that many high-weight programs fire on rank highest
6. The **log scoring rule** (Plackett-Luce log-likelihood) is both the utility function and the conditioning kernel — learning IS utility maximisation

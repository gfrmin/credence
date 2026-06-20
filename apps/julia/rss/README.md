# rss domain — REFERENCE-ONLY

> **Status: reference-only (superseded 2026-06-20).** This program-space /
> Plackett-Luce ranker learns user preference from **read-order**, which is noise
> for the actual rssfeed user (a swipe-reader who opens everything). It is kept as
> a worked program-space example; it is **not** the rssfeed integration path.

The decoupled rssfeed integration is a parametric **MAUT** preference model that
lives in the **rssfeed repo** as a BDSL program and runs entirely over the skin
wire — per-feature weight beliefs, engagement signals (`star`/`thumb`/`dwell`)
conditioning them via `:family` kernels, score `Σ mean(wᵢ)·featureᵢ`. See:

- `docs/decouple/master-plan.md` and `docs/decouple/move-2-design.md` — the design.
- `examples/maut_demo.bdsl` — the domain-neutral pure model to mirror.
- `examples/maut_demo_wire.py` — the end-to-end wire example.

Do not build new rssfeed work on this directory.

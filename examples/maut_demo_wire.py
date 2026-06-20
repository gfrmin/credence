#!/usr/bin/env python
"""End-to-end MAUT demo over the skin wire (decouple Move 2).

Drives examples/maut_demo.bdsl purely over JSON-RPC. Beliefs cross the wire as
declarative {type, params} specs — the skin holds NO state; the consuming app
(here, this script) is the only stateful layer. This is the pattern an external
app (e.g. rssfeed) mirrors from Python.

Run from the repo root (uses the local server.jl for a no-Docker demo):

    uv run python examples/maut_demo_wire.py

In production, a consuming app spawns the pinned engine IMAGE instead — same wire,
no Julia toolchain on the app side:

    SkinClient(command=["docker", "run", "--rm", "-i",
                        "ghcr.io/gfrmin/credence-skin:<tag>"])

The /load /observe /score /health HTTP contract a consuming app exposes maps onto
these skin calls:  /load → initialize(dsl_sources);  /observe → call_dsl(observe);
/score → call_dsl(score-batch);  /health → a cheap call_dsl/initialize ping.
"""

from pathlib import Path

from credence_skin_client import SkinClient

REPO = Path(__file__).resolve().parent.parent
MODEL = (REPO / "examples" / "maut_demo.bdsl").read_text()


def main() -> None:
    skin = SkinClient(
        server_path=REPO / "apps" / "skin" / "server.jl",
        project=REPO,
    )
    try:
        info = skin.initialize(dsl_sources={"model": MODEL})
        print(f"engine {info['version']}, wire protocol {info['protocol']}")

        # Two signed-weight priors, as declarative belief-specs (the app's
        # durable store would hold these as plain JSON in a database).
        w0 = {"type": "gaussian", "mu": 0.0, "sigma": 1.0}
        w1 = {"type": "gaussian", "mu": 0.0, "sigma": 1.0}

        # /observe: a positive engagement signal on weight 0. Belief-spec in ->
        # conditioned belief-spec out. No state_id — the skin held nothing.
        w0 = skin.call_dsl("model", "observe", [w0, 0.8])
        assert "state_id" not in w0, f"belief leaked as state: {w0}"
        print(f"w0 posterior: {w0}")

        # /score: rank two articles over the (updated) weight vector. Returns a
        # numeric vector in input order.
        scores = skin.call_dsl(
            "model", "score-batch",
            [[w0, w1], [[1.0, 0.0], [0.0, 1.0]]],
        )
        print(f"article scores: {scores}")
        # Article 0 loads onto the reinforced weight 0; article 1 onto the
        # untouched weight 1 (prior mean 0).
        assert scores[0] > scores[1], scores
        print("OK: pure belief round-trip over the wire, no skin state")
    finally:
        skin.shutdown()


if __name__ == "__main__":
    main()

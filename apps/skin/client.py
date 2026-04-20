# Role: skin
"""Credence skin client — spawns Julia skin process, talks JSON-RPC over stdio.

Usage:
    skin = SkinClient()
    skin.initialize(dsl_files={"router": "examples/router.bdsl"})
    state_id = skin.create_state(type="beta", alpha=1.0, beta=1.0)
    skin.condition(state_id, kernel={"type": "bernoulli"}, observation=1.0)
    w = skin.weights(state_id)
    skin.shutdown()
"""

from __future__ import annotations

import json
import logging
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class SkinError(Exception):
    """Error from the skin process."""

    def __init__(self, code: int, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


class SkinClient:
    """Spawns a Julia skin subprocess, communicates via JSON-RPC 2.0 over stdio."""

    def __init__(
        self,
        julia: str = "julia",
        server_path: str | Path | None = None,
        startup_timeout: float = 120.0,
        project: str | Path | None = None,
    ):
        if server_path is None:
            server_path = Path(__file__).parent / "server.jl"
        self._julia = julia
        self._server_path = str(Path(server_path).resolve())
        self._startup_timeout = startup_timeout
        self._project = str(Path(project).resolve()) if project else None
        self._process: subprocess.Popen | None = None
        self._request_id = 0

    def _ensure_started(self):
        if self._process is not None and self._process.poll() is None:
            return
        argv = [self._julia]
        if self._project:
            argv.append(f"--project={self._project}")
        argv.append(self._server_path)
        log.info("Starting skin process: %s", " ".join(argv))
        self._process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        self._wait_for_ready()

    def _wait_for_ready(self):
        """Wait for the Julia skin server's startup-complete sentinel.

        The server emits `{"status": "ready"}\\n` on stdout after all
        module loading finishes, before entering the JSON-RPC accept
        loop (`apps/skin/server.jl::main()`). This method reads stdout
        until that line arrives, with `process.poll()` polled alongside
        to detect early crash, and a generous ceiling
        (`self._startup_timeout`, default 120s) that accommodates
        cold-compile + loaded-runner variance. See issue #22 for the
        class of lifecycle-budget failures this replaces.
        """
        assert self._process is not None
        assert self._process.stdout is not None
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            exit_code = self._process.poll()
            if exit_code is not None:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read()
                raise SkinError(
                    -1,
                    f"Skin process exited with code {exit_code} before "
                    f"ready sentinel. stderr: {stderr}",
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Poll stdout with a short budget; loop re-checks process.poll().
            readable, _, _ = select.select(
                [self._process.stdout.fileno()], [], [], min(remaining, 1.0),
            )
            if not readable:
                continue
            line = self._process.stdout.readline()
            if not line:
                # EOF; process exited. Loop catches it via .poll() next iter.
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Not a JSON line (e.g. stray Julia warning to stdout).
                # Shouldn't happen in practice — stderr captures warnings —
                # but guard defensively.
                continue
            if isinstance(msg, dict) and msg.get("status") == "ready":
                log.info("Skin process ready")
                return
        raise SkinError(
            -1,
            f"Skin process did not emit ready sentinel within "
            f"{self._startup_timeout}s",
        )

    def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC request and return the result."""
        self._ensure_started()
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }

        line = json.dumps(request) + "\n"
        log.debug("→ %s", line.rstrip())
        self._process.stdin.write(line)
        self._process.stdin.flush()

        response_line = self._process.stdout.readline()
        if not response_line:
            # Process died — collect stderr
            stderr = ""
            if self._process.stderr:
                stderr = self._process.stderr.read()
            raise SkinError(-1, f"Skin process died. stderr: {stderr}")

        log.debug("← %s", response_line.rstrip())
        response = json.loads(response_line)

        if "error" in response:
            err = response["error"]
            raise SkinError(err["code"], err["message"])

        return response["result"]

    # ── Lifecycle ──

    def initialize(
        self,
        dsl_files: dict[str, str | Path] | None = None,
        plugins: list[str | Path] | None = None,
    ) -> dict:
        """Initialize the skin. Must be called before other operations."""
        params: dict[str, Any] = {}
        if dsl_files:
            params["dsl_files"] = {
                name: str(Path(path).resolve()) for name, path in dsl_files.items()
            }
        if plugins:
            params["plugins"] = [str(Path(p).resolve()) for p in plugins]
        return self._call("initialize", params)

    def shutdown(self):
        """Bounded-time shutdown.

        Steps:
          1. Send `shutdown` RPC (the server's main loop returns on receipt).
          2. Close stdin to signal EOF (the server's `eachline(stdin)` loop
             exits on EOF even if the RPC path didn't land cleanly).
          3. `wait(timeout=5)` for clean exit.
          4. If still alive: `terminate()` (SIGTERM), `wait(timeout=5)`.
          5. If still alive: `kill()` (SIGKILL), `wait(timeout=2)`.

        Total ceiling ~12s; guaranteed to complete (SIGKILL cannot be
        ignored). Replaces the earlier 30s `wait-after-terminate` pattern
        that flaked under cold-compile + loaded-runner variance (issue #22,
        matching class as #9's teardown-path flake). Per the repo precedent,
        further timeout bumps would be a signal that something in shutdown
        regressed, not a budget inadequacy.
        """
        if self._process is None:
            return

        try:
            self._call("shutdown")
        except (SkinError, BrokenPipeError):
            pass

        # Close stdin so the server's `eachline(stdin)` loop sees EOF and exits,
        # even if the shutdown RPC path didn't land (e.g. broken pipe).
        try:
            if self._process.stdin:
                self._process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.info("Skin process did not exit cleanly; sending SIGTERM")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log.warning("Skin process ignored SIGTERM; sending SIGKILL")
                self._process.kill()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    log.error(
                        "Skin process still alive after SIGKILL+2s; giving up"
                    )

        self._process = None

    def __del__(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()

    # ── State management ──

    def create_state(self, **kwargs) -> str:
        """Create a new measure or agent state. Returns state_id."""
        result = self._call("create_state", kwargs)
        return result["state_id"]

    def destroy_state(self, state_id: str):
        """Release a state object."""
        self._call("destroy_state", {"state_id": state_id})

    def snapshot_state(self, state_id: str) -> str:
        """Serialize state to base64. For persistence."""
        result = self._call("snapshot_state", {"state_id": state_id})
        return result["data"]

    def restore_state(self, data: str) -> str:
        """Restore state from base64 snapshot. Returns new state_id."""
        result = self._call("restore_state", {"data": data})
        return result["state_id"]

    def transfer_beliefs(self, source_id: str, target_config: dict) -> dict:
        """Transfer beliefs to a new state with different embodiment."""
        return self._call("transfer_beliefs", {
            "source_id": source_id,
            "target_config": target_config,
        })

    # ── Core inference (Tier 1) ──

    def condition(
        self,
        state_id: str,
        kernel: dict,
        observation: float | int | str,
    ) -> dict:
        """Bayesian inversion. Updates state in place."""
        return self._call("condition", {
            "state_id": state_id,
            "kernel": kernel,
            "observation": observation,
        })

    def weights(self, state_id: str) -> list[float]:
        """Normalized probability weights."""
        result = self._call("weights", {"state_id": state_id})
        return result["weights"]

    def mean(self, state_id: str) -> float:
        """Mean of a measure."""
        result = self._call("mean", {"state_id": state_id})
        return result["mean"]

    def expect(self, state_id: str, function: dict) -> float:
        """Integration against a measure."""
        result = self._call("expect", {
            "state_id": state_id,
            "function": function,
        })
        return result["value"]

    def optimise(
        self,
        state_id: str,
        actions: dict,
        preference: dict,
    ) -> tuple[Any, float]:
        """EU-maximizing action. Returns (action, eu)."""
        result = self._call("optimise", {
            "state_id": state_id,
            "actions": actions,
            "preference": preference,
        })
        return result["action"], result["eu"]

    def value(self, state_id: str, actions: dict, preference: dict) -> float:
        """Maximum expected utility."""
        result = self._call("value", {
            "state_id": state_id,
            "actions": actions,
            "preference": preference,
        })
        return result["value"]

    def draw(self, state_id: str) -> Any:
        """Sample from the measure."""
        result = self._call("draw", {"state_id": state_id})
        return result["value"]

    # ── ProductMeasure decomposition ──

    def factor(self, state_id: str, index: int) -> str:
        """Extract factor `index` (0-based) of a ProductMeasure as a new state."""
        result = self._call("factor", {"state_id": state_id, "index": index})
        return result["state_id"]

    def replace_factor(self, state_id: str, index: int, new_factor_id: str) -> str:
        """Return a new state with factor `index` replaced by new_factor_id."""
        result = self._call("replace_factor", {
            "state_id": state_id,
            "index": index,
            "new_factor_id": new_factor_id,
        })
        return result["state_id"]

    def n_factors(self, state_id: str) -> int:
        """Number of factors in a ProductMeasure state."""
        result = self._call("n_factors", {"state_id": state_id})
        return result["n_factors"]

    def _dispatch_path(self, state_id: str, kernel: dict) -> str:
        """Dispatch-path observability hook (underscore-prefixed, test-only).

        Returns "conjugate" if the (state, kernel) pair matches a registered
        conjugate entry in the ConjugatePrevision registry; "particle"
        otherwise. Does NOT mutate state. Used by Stratum-2 skin-smoke
        tests to pin registry-dispatch decisions explicitly.

        See docs/posture-3/move-4-design.md §5.2 for the contract.
        """
        result = self._call("_dispatch_path", {
            "state_id": state_id,
            "kernel": kernel,
        })
        return result["path"]

    # ── Program-space operations (Tier 2) ──

    def enumerate(
        self,
        state_id: str,
        grammar: dict,
        max_depth: int,
        action_space: list[str],
    ) -> dict:
        """Enumerate programs from a grammar."""
        return self._call("enumerate", {
            "state_id": state_id,
            "grammar": grammar,
            "max_depth": max_depth,
            "action_space": action_space,
        })

    def perturb_grammar(
        self,
        state_id: str,
        grammar_id: int,
        all_features: list[str],
    ) -> dict:
        """Analyse posterior and create perturbed grammar."""
        return self._call("perturb_grammar", {
            "state_id": state_id,
            "grammar_id": grammar_id,
            "all_features": all_features,
        })

    def add_programs(self, state_id: str, grammar_id: int, max_depth: int) -> dict:
        """Enumerate from existing grammar at new depth."""
        return self._call("add_programs", {
            "state_id": state_id,
            "grammar_id": grammar_id,
            "max_depth": max_depth,
        })

    def sync_prune(self, state_id: str, threshold: float = -30.0) -> int:
        """Remove negligible-weight components. Returns n_remaining."""
        result = self._call("sync_prune", {
            "state_id": state_id,
            "threshold": threshold,
        })
        return result["n_remaining"]

    def sync_truncate(self, state_id: str, max_components: int = 2000) -> int:
        """Keep only top-weighted components. Returns n_remaining."""
        result = self._call("sync_truncate", {
            "state_id": state_id,
            "max_components": max_components,
        })
        return result["n_remaining"]

    def top_grammars(self, state_id: str, k: int = 3) -> list[dict]:
        """Top-k grammars by posterior weight."""
        result = self._call("top_grammars", {
            "state_id": state_id,
            "k": k,
        })
        return result["grammars"]

    # ── Batch operations ──

    def belief_summary(self, state_id: str) -> dict:
        """Full belief summary in one call."""
        return self._call("belief_summary", {"state_id": state_id})

    def condition_and_prune(
        self,
        state_id: str,
        kernel: dict,
        observation: float,
        prune_threshold: float = -30.0,
        max_components: int = 2000,
    ) -> dict:
        """condition + sync_prune + sync_truncate in one call."""
        return self._call("condition_and_prune", {
            "state_id": state_id,
            "kernel": kernel,
            "observation": observation,
            "prune_threshold": prune_threshold,
            "max_components": max_components,
        })

    def eu_interact(
        self,
        state_id: str,
        features: dict[str, float],
        rewards: dict[str, float],
    ) -> dict:
        """EU of interacting for program-space agents."""
        return self._call("eu_interact", {
            "state_id": state_id,
            "features": features,
            "rewards": rewards,
        })

    # ── DSL operations ──

    def call_dsl(
        self,
        env_id: str,
        function: str,
        args: list[Any] | None = None,
    ) -> Any:
        """Call a named function from a loaded DSL environment.

        Use {"ref": state_id} in args to pass state references.
        """
        result = self._call("call_dsl", {
            "env_id": env_id,
            "function": function,
            "args": args or [],
        })
        # Result is either {"value": x} or {"state_id": x} or {"state_ids": [...]}
        if "value" in result:
            return result["value"]
        if "state_id" in result:
            return result["state_id"]
        if "state_ids" in result:
            return result["state_ids"]
        return result

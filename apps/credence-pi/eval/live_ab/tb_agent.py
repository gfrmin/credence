"""Custom Terminal-Bench agent: claude-code with clean stream-json capture.

The stock `claude-code` agent pipes `claude --output-format stream-json` to a
tmux pane, which asciinema double-encodes and the terminal line-wraps — the
per-message `usage` (incl. cache tokens) and the final `result` event
(`num_turns`, `total_cost_usd`) survive only as mangled fragments. We need
those verbatim to price each run cache-accurately and to feed the
Poisson-Gamma turns belief.

This subclass changes ONE thing: it redirects claude's stdout to a JSONL file
in the container's mounted logs dir (`/logs` -> host `sessions/`), so the host
gets clean, parseable JSONL. Everything else (model mapping via ANTHROPIC_MODEL,
allowed tools, install script) is inherited unchanged — so this stays a faithful
native-tool-calling agentic loop, the same class as OpenClaw.

Run via:
    tb run --agent-import-path tb_agent:ClaudeCodeStreamJSONL \
           --model anthropic/claude-haiku-4-5-20251001 ...
with the eval dir on PYTHONPATH.
"""
from __future__ import annotations

import os
import shlex
import tempfile
from pathlib import Path

import terminal_bench.agents.installed_agents.claude_code.claude_code_agent as _cc
from terminal_bench.agents.installed_agents.claude_code.claude_code_agent import (
    ClaudeCodeAgent,
)
from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.utils.template_utils import render_setup_script

# Where claude's clean stream-json lands inside the container; the logs dir is
# bind-mounted to the host trial's sessions/ directory by the harness.
STREAM_PATH = "/logs/claude_stream.jsonl"


class ClaudeCodeStreamJSONL(ClaudeCodeAgent):
    @staticmethod
    def name() -> str:
        return "claude-code-stream-jsonl"

    @property
    def _env(self) -> dict:
        # Install nvm/node/claude under /opt, not $HOME (=/root). One task
        # (extract-safely) mounts /root as a 100MB tmpfs, which overflows during
        # the ~2GB node install → `claude: command not found`. /opt is roomy
        # (overlay, not tmpfs), system-level (outside the task's /app workdir so it
        # can't trip a "no extra files" check), and exists on all the tb base
        # images. nvm installs to $HOME/.nvm and the run shell sources the same, so
        # one HOME override fixes both install and invocation.
        env = super()._env
        env["HOME"] = "/opt"
        return env

    @property
    def _install_agent_script_path(self) -> Path:
        # The parent renders the template relative to its OWN file via
        # inspect.getfile(self.__class__); for a subclass that resolves to this
        # file's dir, which has no .j2. Render the parent package's template instead.
        template_path = Path(_cc.__file__).parent / "claude-code-setup.sh.j2"
        content = render_setup_script(template_path, self._get_template_variables())
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
        tf.write(content)
        tf.close()
        os.chmod(tf.name, 0o755)
        return Path(tf.name)

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        escaped = shlex.quote(instruction)
        tools = " ".join(self.ALLOWED_TOOLS)
        # Redirect stdout (the stream-json) to the mounted logs file; stderr to a
        # sibling. The pane stays quiet — we parse the file on the host, not the cast.
        return [
            TerminalCommand(
                command=(
                    f"claude --verbose --output-format stream-json "
                    f"-p {escaped} --allowedTools {tools} "
                    f"> {STREAM_PATH} 2> /logs/claude_stderr.log"
                ),
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=True,
                append_enter=True,
            ),
        ]

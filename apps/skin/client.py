# Role: body
"""In-repo re-export of the credence-skin-client wire client.

The canonical implementation lives in the `credence-skin-client` package under
`apps/skin/clients/python/`. This shim preserves the in-repo import path
`from client import SkinClient` (used by the skin smoke tests and in-repo
co-released consumers) and defaults `server_path` to the adjacent `server.jl`
for local development. External apps depend on `credence-skin-client` directly,
not on this shim.
"""

from __future__ import annotations

from pathlib import Path

from credence_skin_client import PROTOCOL_MAJOR, SkinError
from credence_skin_client import SkinClient as _SkinClient

__all__ = ["SkinClient", "SkinError", "PROTOCOL_MAJOR"]

_ADJACENT_SERVER = Path(__file__).parent / "server.jl"


class SkinClient(_SkinClient):
    """SkinClient defaulting to the adjacent in-repo ``server.jl`` (dev convenience)."""

    def __init__(
        self,
        julia: str = "julia",
        server_path: str | Path | None = None,
        startup_timeout: float = 120.0,
        project: str | Path | None = None,
        command: list[str] | None = None,
    ):
        if command is None and server_path is None:
            server_path = _ADJACENT_SERVER
        super().__init__(
            julia=julia,
            server_path=server_path,
            startup_timeout=startup_timeout,
            project=project,
            command=command,
        )

# credence-skin-client

The thin JSON-RPC wire client for the Credence skin — the one library external
apps depend on to talk to the engine.

It ships **no engine code** (no `juliacall`, no Julia) and holds **no Measures**:
state stays server-side as opaque IDs. A consumer of this client physically
cannot do probability arithmetic, so it **cannot host a brain**. This is the
boundary that makes the constitutional single-reasoner invariant hold by
construction for any app on the other side of the wire.

## Usage

Launch the engine as a pinned image and talk JSON-RPC over its stdio:

```python
from credence_skin_client import SkinClient

skin = SkinClient(command=[
    "docker", "run", "--rm", "-i", "ghcr.io/gfrmin/credence-skin:<tag>",
])
skin.initialize(dsl_sources={"agent": "(defun choose ...)"})  # inline BDSL, no host paths
sid = skin.create_state(type="beta", alpha=1.0, beta=1.0)
skin.condition(sid, kernel={"type": "bernoulli"}, observation=1.0)
print(skin.weights(sid))
skin.shutdown()
```

The client pins a wire **protocol major** and the handshake rejects an
incompatible engine with error `-32010`. See `apps/skin/protocol.md` for the
full method set and the protocol changelog.

Apps declare their domain as **data** — `dsl_sources` are inline BDSL source
strings; the engine never reads the host filesystem.

# Notes on `apps/credence-governance-sidecar/`

`apps/credence-governance-sidecar/` (the OpenClaw integration) predates
this branch and is not a model for `credence-pi`. Its design — host-side
EU arithmetic with pragma armour, hand-rolled feature partitions,
multi-endpoint HTTP — is precisely what credence-pi was built to avoid.

The sidecar remains in place untouched as an artefact; deprecating or
deleting it is deferred to Pass 2 (see `SPEC.md` § "Decisions deferred
to Pass 2").

For the binding argument behind credence-pi's first principles, see
`SPEC.md` § "Status" and § "First principles".

# Packaging Audit: Credence Governance Plugin

Audited 2026-04-30 against OpenClaw 2026.4.27 plugin documentation
(docs.openclaw.ai/plugins/{building-plugins,manifest,hooks,sdk-overview}).

## Matches documentation

- **Hook names.** `before_tool_call`, `after_tool_call`, `before_compaction`,
  `agent_end` all exist with the same semantics the plugin assumes.
- **Hook registration.** `api.on(name, handler, opts?)` is the documented API.
- **Hook priority.** `{ priority: 100 }` is valid (higher runs first).
- **`before_tool_call` return shape.** `{ block, blockReason, requireApproval }`
  matches the documented `BeforeToolCallResult` type.
- **`requireApproval` fields.** `title`, `description`, `severity`, `timeoutMs`,
  `timeoutBehavior` are all documented.
- **Manifest required fields.** `openclaw.plugin.json` has `id` and `configSchema`
  (both required per docs).
- **`activation.onStartup: true`.** Documented and correct.
- **Fail-open pattern.** No documentation requires fail-closed; fail-open is the
  expected default for non-security plugins.
- **`package.json` `type: "module"`.** Correct for ESM plugins.
- **`openclaw.extensions`.** Array of entry files pointing at `./src/index.ts`.

## Drift identified

### D1. `package.json` missing `openclaw.runtimeExtensions`

When `extensions` points at TypeScript source (`./src/index.ts`), docs require
`runtimeExtensions` containing the built JS peer (`./dist/index.js`). Without
this, the loader may fail to resolve the runtime entry.

**Fix:** Add `"runtimeExtensions": ["./dist/index.js"]` to `openclaw` key.

### D2. `package.json` missing `openclaw.compat`

Docs show `compat.pluginApi` (semver range) and `compat.minGatewayVersion` as
standard metadata. Absence may cause install warnings or `plugins doctor` flags.

**Fix:** Add `compat` with `pluginApi` and `minGatewayVersion` values.

### D3. `package.json` missing `openclaw.build`

Docs show `build.openclawVersion` and `build.pluginSdkVersion` as informational
metadata expected by `plugins doctor`.

**Fix:** Add `build` with current versions.

### D4. Config access: `api.config` should be `api.pluginConfig`

`api.config` is the full OpenClaw config object. Plugin-specific config from
`plugins.entries.<id>.config` is exposed via `api.pluginConfig`. The plugin reads
`api.config.sidecarUrl` which resolves to `undefined` because `sidecarUrl` lives
under the plugin's config subtree, not at the OpenClaw config root.

**Impact:** Functional bug — plugin always falls back to hardcoded defaults
regardless of user configuration.

**Fix:** Replace `api.config` with `api.pluginConfig` on line 66 of `src/index.ts`.

### D5. Logging: `api.log` should be `api.logger`

`api.log` is not in the documented plugin API. The documented logging interface
is `api.logger` with `.debug()`, `.info()`, `.warn()`, `.error()` methods.
The current code uses `api.log?.(level, message)` — optional chaining prevents
a crash, but logging silently no-ops.

**Fix:** Replace `api.log` calls with `api.logger` method calls.

### D6. `RequireApprovalPayload` severity enum mismatch

Docs define `severity?: "info" | "warning" | "critical"`. Plugin type uses
`"error"` instead of `"critical"`. If the sidecar sends severity `"error"`,
OpenClaw may ignore it or fall back to a default.

**Fix:** Change `"error"` to `"critical"` in `RequireApprovalPayload` type.

### D7. Unused `maxRepetitions` config in manifest

`openclaw.plugin.json` declares `maxRepetitions` in `configSchema.properties`
but the plugin code never reads it. Dead config confuses users inspecting the
plugin via `openclaw plugins inspect`.

**Fix:** Remove `maxRepetitions` from `configSchema.properties`.

### D8. Timeout default mismatch

`openclaw.plugin.json` declares default 200ms; `src/index.ts` hardcodes fallback
50ms. These should agree. The manifest value (200ms) is the documented default.

**Fix:** Change the hardcoded fallback in `src/index.ts` from 50 to 200.

### D9. README references YAML config format

OpenClaw config is `~/.openclaw/openclaw.json` (JSON5 format). The README
references `config.yaml` which doesn't exist.

**Fix:** Rewrite README with correct JSON5 config format and path.

### D10. README install command uses wrong flag

README says `openclaw plugins install . --link`; docs show the link flag as
`-l` (short form): `openclaw plugins install -l <path>`.

**Fix:** Correct in README rewrite.

### D11. Entry point pattern: raw export vs `definePluginEntry()`

Plugin uses `export default { id, name, description, register }`. Docs show
`definePluginEntry()` from `"openclaw/plugin-sdk/plugin-entry"`. The shapes are
structurally equivalent, but `definePluginEntry` may add runtime markers the
loader checks.

**Status:** Needs empirical verification during smoke test. If the loader
rejects the raw export, an additional commit switches to `definePluginEntry()`.

### D12. `dist/` not built

No compiled output exists. The plugin needs `npm install && npm run build`
before it can be loaded by OpenClaw.

**Fix:** Build step added to README prerequisites. `dist/` stays gitignored.

## Noted but out of scope

- **`onResolution` callback.** Docs show `requireApproval.onResolution` as the
  way to receive user approval/denial decisions. Plugin currently correlates via
  `event.userApproval` on `after_tool_call`. This may be an older pattern. Not a
  packaging item — changing it would alter hook behavior.
- **`after_tool_call` `event.userApproval` field.** Not explicitly documented in
  the hook event reference. May work via OpenClaw's internal forwarding. Verify
  during deployment, not here.

## Smoke test results

Tested 2026-04-30 against OpenClaw 2026.4.27 (cbc2ba0).

### Build

```
$ npm run build
> credence-governance@0.1.0 build
> tsc
```

No errors. `dist/index.js`, `dist/sidecar-client.js` + declaration files produced.

### Install

```
$ openclaw plugins install -l /home/g/git/credence/apps/openclaw-plugin
Linked plugin path: ~/git/credence/apps/openclaw-plugin
Restart the gateway to load plugins.
```

### Verify

```
$ openclaw plugins list
Credence Governance | credence-governance | openclaw | enabled | 0.1.0

$ openclaw plugins inspect credence-governance
Status: loaded
Format: openclaw
Shape: hook-only
Typed hooks:
  after_tool_call
  before_compaction
  before_tool_call (priority 100)
Diagnostics:
  WARN: typed hook "agent_end" blocked — requires allowConversationAccess=true

$ openclaw plugins doctor
credence-governance is hook-only [info]
```

### Entry point pattern (D11 verification)

**Confirmed: raw `export default { register }` is accepted.** The loader
loaded the plugin without requiring `definePluginEntry()`. No fix needed.

### New finding: `agent_end` hook requires `allowConversationAccess`

OpenClaw classifies `agent_end` as a conversation-access hook. Non-bundled
plugins need `hooks.allowConversationAccess: true` in config for it to fire.
Without this, the plugin's history buffer cleanup doesn't run at agent-turn
boundaries. Impact is minor — the buffer is bounded (50 entries) and stale
entries naturally age out. But the config key should be documented.

**Fix:** Added `allowConversationAccess` to README config example.

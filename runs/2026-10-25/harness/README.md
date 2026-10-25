# Harness evidence

The two operational recovery instructions are retained separately from the
unchanged task prompt. The initial instruction is the benchmark's complete
procedural `PROMPT.md`. `usage.json` contains only token usage, timestamps,
model names, status codes and request hashes; it contains no request bodies,
images, account identifiers or API keys.

`guard-checks.json` verifies that an unknown model, an alternate inference route,
and a purchase route are rejected locally before upstream inference. The four
cache-price regression checks cover cached and uncached input, reasoning tokens
counted once, and invalid cached-input counts.

`scripts/kimi_budget_gateway.py` is the final standard-library budget relay.
It is a harness utility, not part of the HTML page, and never purchases credit.
Its state directory defaults to ignored `.work/kimi-control` and can be selected
with the nonsecret `KIMI_RUN_CONTROL_DIR` environment variable. It receives an
already approved API key once over a user-only Unix socket and keeps it in memory.
OpenCode uses only a dummy local credential. Keys must never be placed in a
command line, file, Git, a prompt, browser JavaScript shipped to viewers, or a log.
Obtain authorization and a secure credential handoff before any future run.

For each request it pins `kimi-k3.1-pro`, uses the official token-estimation API,
reserves the full uncached-input worst case plus maximum output before sending,
and rejects requests exceeding USD 5. Completed requests release the excess
reservation using reported cache hits. Interrupted requests without usage retain
the full reservation across restarts. Provider balance and quota remain separate
authoritative checks. The prices are the published 2026-10-25 values and must be
reverified before reuse. The original overcounted ledger was corrected without
resetting prior usage or discarding unknown-request reservations.

No ongoing inference process is needed by the delivered viewer. The API gateway
and OpenCode generation processes were stopped after the quota block; the HTML
only serves the saved artifacts.

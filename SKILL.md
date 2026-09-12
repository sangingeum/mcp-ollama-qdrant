---
name: mcp-ollama-qdrant
description: Persistent vector memory for AI agents via Ollama embeddings + Qdrant. Use when the agent needs to save, recall, or semantically search notes/scenarios/documents across sessions.
---

# mcp-ollama-qdrant (Agent Vector Memory)

MCP stdio server giving the agent a persistent semantic memory: text is
embedded by Ollama (`qwen3-embedding:8b`, 4096-dim) and stored in Qdrant
(cosine). Fully LAN-local, no cloud.

## When to use

- Persisting durable knowledge: scenario outcomes, decisions, facts, doc
  summaries — for later recall across sessions.
- Recalling past context: search memories by meaning, not keyword.
- Do NOT use for: indexing codebases (that is `mcp-code-indexer`), storing
  secrets/credentials, or exact-match lookup (it is semantic, not a DB).

## Tools (stdio MCP, six)

| Tool | Params | Returns |
|---|---|---|
| `save_memory` | `text: str`, `metadata: str = "{}"` (JSON string), `collection: str = ""` | Confirmation with new point UUID + collection, e.g. `Memory saved (ID: <uuid>, collection: <name>)`. The saved text is also stored in the payload under `text`. |
| `save_memories` | `texts: list[str]`, `metadata: str` (applied to all), `collection: str = ""` | Count + comma-separated UUIDs. Single Ollama batch call, single upsert. |
| `search_memory` | `query: str`, `limit: int = 3`, `filter: str = ""` (payload-filter JSON), `collection: str = ""` | Lines like `- [ID: <uuid>] [score: 0.8421] metadata: {"tags": [...]} | content: <text>`, best first. **Every hit includes the point ID and metadata** — enough to call `delete_memory`/`update_memory` directly from search output. Empty → `No matching memories found.` |
| `update_memory` | `point_id: str`, `text: str \| None = None`, `metadata: str = ""`, `collection: str = ""` | Re-embeds the new text and overwrites the point in place (same ID). Empty `metadata` keeps the existing payload metadata and replaces only the text; a JSON string replaces the whole payload metadata. `text=None` (or blank) with a `metadata` JSON string updates ONLY metadata — the stored text and vector are kept, no re-embedding (no Ollama call). Passing neither text nor metadata → `Error: nothing to update…`. Nonexistent ID → `Update failed: point_id … not found.` (existence verified via retrieve first). |
| `delete_memory` | `point_id: str` (the UUID from save/search), `collection: str = ""` | Confirmation `Memory deleted (ID: …)`. Deletes the point permanently. Nonexistent ID → `Error: point_id … not found` (existence verified via retrieve first, no false success). |
| `list_collections` | — | Collection names (comma-separated). |

All tool text returns are English-language strings; scores are cosine
similarity, 0–1.

## Collections (`collection` param)

Every data tool (`save_memory`, `save_memories`, `search_memory`,
`update_memory`, `delete_memory`) accepts an optional `collection` param.
Empty string (default) uses the server-configured collection
(`--collection` / `COLLECTION_NAME`, default `agent_scenarios`).

- Naming a nonexistent collection on **save** creates it on the fly
  (dimension probed from the current `EMBED_MODEL`, cosine). This is the
  intended escape from the single-collection limitation — no server split
  needed; run one server instance with a different `--collection` only if
  you also want a different default.
- On **search/update/delete** a nonexistent collection errors naturally
  (search/update return the Qdrant error; update checks
  `collection_exists` first).
- Dimension-mismatch fail-fast applies to ad-hoc collections too: saving
  into an existing collection built with a different model/vector size
  exits the server with the dimension-mismatch error, same as the
  default collection at startup.

## Payload filtering (search_memory `filter`)

JSON string `{"field": value}`. List value → MatchAny (field contains any
value); scalar → exact match; multiple fields AND-ed together.
Invalid JSON → filter is ignored and a warning line is appended to
the returned string (e.g. `Warning: failed to parse filter JSON; searching
without a filter`),
so the caller knows results are unfiltered. Same for invalid `metadata`
on save: warning line in the return value
(`Warning: failed to parse metadata JSON; saved with empty metadata`),
empty metadata stored.

## Configuration

Order: CLI flags > env vars > defaults.

| Setting | CLI flag | Env var | Default |
|---|---|---|---|
| Ollama URL | `--ollama-url` | `OLLAMA_URL` | `http://192.168.X.X:11434` |
| Qdrant URL | `--qdrant-url` | `QDRANT_URL` | `http://192.168.X.X:6333` |
| Embed model | `--embed-model` | `EMBED_MODEL` | `qwen3-embedding:8b` |
| Collection | `--collection` | `COLLECTION_NAME` | `agent_scenarios` |

Collection is created on server startup if missing (dimension probed from the
model). Run via `uv run mcp-ollama-qdrant` from the repo directory.

## Gotchas

- **Dimension mismatch fails fast.** If a collection already exists with a
  different vector size than the current `EMBED_MODEL` produces, the server
  exits with a descriptive error (and suggests: delete and recreate the
  collection, or revert EMBED_MODEL). You cannot mix models in one
  collection. Same model change also requires re-embedding existing
  memories — old vectors stay valid only under the original model.
- **metadata must be a JSON string**, not an object. Pass `'{"tags":["a"]}'`.
  Invalid JSON is stored as empty metadata **and a warning line is returned
  in the tool's output** — check it if a filter unexpectedly matches
  nothing. Non-dict JSON also becomes empty metadata (own warning line).
- **update_memory replaces the whole payload metadata when given** — pass
  `metadata=""` (default) to keep existing metadata and change only text.
  The text field is always replaced.
- **First-index latency**: the startup collection-creation probes the model
  with a real embed call, and every `save/search/update_memory` pays one
  embedding round trip per call (~0.8 s GPU-hosted / ~6 s CPU-hosted, 8B
  model). Batch saves (`save_memories`) amortize this. Design waits
  accordingly.
- Every save stores the full text inside the payload — search results include
  the text, ID, and metadata, no follow-up fetch needed.
- Server startup itself requires both Ollama and Qdrant reachable; failure
  surfaces as connection errors at startup, not at first tool call.
- Diagnostics go to stderr; stdout is the MCP transport. Don't pollute stdout.

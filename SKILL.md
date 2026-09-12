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

## Tools (stdio MCP)

| Tool | Params | Returns |
|---|---|---|
| `save_memory` | `text: str`, `metadata: str = "{}"` (JSON string) | Confirmation with new point UUID, e.g. `기억 저장 완료 (ID: <uuid>)`. The saved text is also stored in the payload under `text`. |
| `save_memories` | `texts: list[str]`, `metadata: str` (applied to all) | Count + comma-separated UUIDs. Single Ollama batch call, single upsert. |
| `search_memory` | `query: str`, `limit: int = 3`, `filter: str = ""` (payload-filter JSON) | Lines like `- [유사도: 0.8421] <text>`, best first. Empty → `관련된 기억을 찾을 수 없습니다.` |
| `delete_memory` | `point_id: str` (the UUID from save) | Confirmation. Deletes the point permanently. |
| `list_collections` | — | Collection names (comma-separated). |

All tool text returns are Korean-language strings; scores are cosine
similarity, 0–1.

## Payload filtering (search_memory `filter`)

JSON string `{"field": value}`. List value → MatchAny (field contains any
value); scalar → exact match; multiple fields AND-ed together.
Invalid JSON → filter silently ignored (warning logged to stderr), search
runs unfiltered.

## Configuration

Order: CLI flags > env vars > defaults.

| Setting | CLI flag | Env var | Default |
|---|---|---|---|
| Ollama URL | `--ollama-url` | `OLLAMA_URL` | `http://192.168.1.103:11434` |
| Qdrant URL | `--qdrant-url` | `QDRANT_URL` | `http://192.168.1.105:6333` |
| Embed model | `--embed-model` | `EMBED_MODEL` | `qwen3-embedding:8b` |
| Collection | `--collection` | `COLLECTION_NAME` | `agent_scenarios` |

Collection is created on server startup if missing (dimension probed from the
model). Run via `uv run mcp-ollama-qdrant` from the repo directory.

## Gotchas

- **Dimension mismatch fails fast at startup.** If the collection already
  exists with a different vector size than the current `EMBED_MODEL` produces,
  the server exits with a Korean-language error (and suggests: delete and
  recreate the collection, or revert EMBED_MODEL). You cannot mix models in
  one collection. Same model change also requires re-embedding existing
  memories — old vectors stay valid only under the original model.
- **metadata must be a JSON string**, not an object. Pass `'{"tags":["a"]}'`.
  Invalid JSON is stored as empty metadata (with a stderr warning) — check the
  warning if a filter unexpectedly matches nothing. Non-dict JSON also becomes
  empty metadata.
- **First-index latency**: the startup collection-creation probes the model
  with a real embed call; on a CPU-hosted 8B model this takes seconds, and
  every `save/search_memory` pays one ~1–6 s embedding round trip per call.
  Batch saves (`save_memories`) amortize this. Design waits accordingly.
- Every save stores the full text inside the payload — search results include
  the text, no follow-up fetch needed.
- Server startup itself requires both Ollama and Qdrant reachable; failure
  surfaces as connection errors at startup, not at first tool call.
- Diagnostics go to stderr; stdout is the MCP transport. Don't pollute stdout.
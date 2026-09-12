# mcp-ollama-qdrant

An [MCP](https://modelcontextprotocol.io) server that gives your AI agent a
**persistent vector memory** backed by:

- **Ollama** for embeddings (tested with `qwen3-embedding:8b`)
- **Qdrant** as the vector store

It exposes six tools over the stdio MCP transport:

| Tool | Description |
|---|---|
| `save_memory(text, metadata, collection)` | Embeds `text` via Ollama and upserts it into Qdrant. `metadata` is optional metadata — a JSON object or a JSON string, both accepted — stored alongside the vector. Optional `collection` targets a specific collection (created on the fly if missing; empty = server default). |
| `save_memories(texts, metadata, collection)` | Batch version: embeds a list of texts in one Ollama call and upserts them as a single batch. `metadata` (object or JSON string) applies to all documents. |
| `search_memory(query, limit, filter, collection)` | Embeds `query` and returns the `limit` most similar stored memories — each hit includes its point **ID**, similarity **score**, **metadata**, and text, so you can `delete_memory`/`update_memory` straight from search output. Optional `filter` is a payload filter — JSON object or JSON string (see below). |
| `update_memory(point_id, text, metadata, collection)` | Re-embeds `text` and overwrites the point in place (same ID). Empty `metadata` keeps the existing payload metadata; a JSON object or JSON string replaces it. Nonexistent IDs return an error. |
| `delete_memory(point_id, collection)` | Deletes the stored memory (point) with the given ID. |
| `list_collections()` | Lists all existing Qdrant collections. |

Every data tool takes an optional `collection` string; an empty value uses
the server-configured collection (`--collection` / `COLLECTION_NAME`).

### Payload filtering

`search_memory` accepts an optional `filter` — a JSON object or a JSON string
built from payload fields. List values become a `MatchAny` condition (matches if the payload
field contains **any** of the values), scalar values become exact matches.
Multiple conditions are AND-ed together:

```json
{"tags": ["x"]}                      // payload.tags contains "x"
{"source": "doc1"}                   // exact match
{"tags": ["a", "b"], "source": "s"}  // AND of MatchAny + match
```

On startup the server connects to Ollama and Qdrant and creates the collection
automatically if it does not exist (cosine distance, dimension probed from the
embedding model). If a collection **already exists** with a different vector
dimension than the current `EMBED_MODEL` produces — whether the default
collection at startup or an ad-hoc one named in a tool call — the server fails
fast with a clear error instead of silently storing corrupt vectors — fix it
by deleting and recreating the collection, or by switching back to the
original embedding model.

Invalid `metadata`/`filter` JSON is not silently ignored: the tool's return
string includes a warning line (e.g.
`Warning: failed to parse metadata JSON; saved with empty metadata`).

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- A reachable Ollama instance (default `http://192.168.X.X:11434`)
- A reachable Qdrant instance (default `http://192.168.X.X:6333`)

## Configuration

Settings resolve in order: **CLI flags > environment variables > defaults**.

| Setting | CLI flag | Env var | Default |
|---|---|---|---|
| Ollama base URL | `--ollama-url` | `OLLAMA_URL` | `http://192.168.X.X:11434` |
| Qdrant base URL | `--qdrant-url` | `QDRANT_URL` | `http://192.168.X.X:6333` |
| Embedding model | `--embed-model` | `EMBED_MODEL` | `qwen3-embedding:8b` |
| Collection name | `--collection` | `COLLECTION_NAME` | `agent_scenarios` |

## Running

With uv (recommended — handles the venv and sync automatically):

```bash
uv sync
uv run mcp-ollama-qdrant            # or: uv run python mcp_server.py
# with overrides:
uv run mcp-ollama-qdrant --qdrant-url http://localhost:6333
```

Interactive testing / inspection:

```bash
uv run mcp dev mcp_server.py
```

### MCP client config

Add to your client's MCP config (Claude Desktop, Hermes, etc.):

```json
{
  "mcpServers": {
    "vector-memory": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/mcp-ollama-qdrant",
        "run", "mcp-ollama-qdrant"
      ],
      "env": {
        "OLLAMA_URL": "http://192.168.X.X:11434",
        "QDRANT_URL": "http://192.168.X.X:6333",
        "EMBED_MODEL": "qwen3-embedding:8b",
        "COLLECTION_NAME": "agent_scenarios"
      }
    }
  }
}
```

(Env entries are optional if the defaults already point at your instances.)

For Hermes `~/.hermes/config.yaml`:

```yaml
mcp:
  servers:
    vector-memory:
      command: uv
      args: ["--directory", "/path/to/mcp-ollama-qdrant", "run", "mcp-ollama-qdrant"]
```

## Testing

Offline unit tests (Ollama and Qdrant are mocked — no live services needed):

```bash
uv run pytest tests/
```

End-to-end smoke test against live Ollama + Qdrant (saves a few memories,
searches for them, prints similarity scores):

```bash
uv sync
uv run python test_server.py
```

## Notes

- All diagnostics are logged to **stderr**; stdout is reserved for the stdio
  MCP transport.
- Dependency pins: `numpy<2`, `qdrant-client<1.15`, `mcp[cli]<2` — chosen for
  compatibility with older x86-64 hardware (pre-x86-64-v2) and the mcp v2
  FastMCP rename. Adjust only with reason.

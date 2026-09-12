# mcp-ollama-qdrant

An [MCP](https://modelcontextprotocol.io) server that gives your AI agent a
**persistent vector memory** backed by:

- **Ollama** for embeddings (tested with `qwen3-embedding:8b`)
- **Qdrant** as the vector store

It exposes two tools over the stdio MCP transport:

| Tool | Description |
|---|---|
| `save_memory(text, metadata)` | Embeds `text` via Ollama and upserts it into Qdrant. `metadata` is an optional JSON string stored alongside the vector. |
| `search_memory(query, limit)` | Embeds `query` and returns the `limit` most similar stored memories with similarity scores. |

On startup the server connects to Ollama and Qdrant and creates the collection
automatically if it does not exist (cosine distance, dimension probed from the
embedding model).

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

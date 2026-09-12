"""MCP server integrating Ollama embeddings with Qdrant vector memory.

Stdio-transport MCP server exposing six tools:

- ``save_memory``      — embed text via Ollama and upsert into Qdrant.
- ``save_memories``    — batch version: embed a list of texts in one call
                          and upsert them as a single batch.
- ``search_memory``    — embed a query and return nearest neighbours,
                          with optional payload filtering.
- ``update_memory``    — re-embed and overwrite an existing point in place
                          (same point ID, vector + payload replaced).
- ``delete_memory``    — delete a stored point by its ID.
- ``list_collections`` — list all Qdrant collections.

Configuration resolution order (highest wins):
    CLI flags -> environment variables -> built-in defaults.

Environment variables:
    OLLAMA_URL      (default http://192.168.1.103:11434)
    QDRANT_URL      (default http://192.168.1.105:6333)
    EMBED_MODEL     (default qwen3-embedding:8b)
    COLLECTION_NAME (default agent_scenarios)
"""

from .embedding import embed, embed_many, embed_batch
from .server import (
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_URL,
    QDRANT_URL,
    delete_memory,
    list_collections,
    main,
    mcp,
    save_memories,
    save_memory,
    search_memory,
    update_memory,
)
from .store import qdrant

__all__ = [
    "COLLECTION_NAME",
    "EMBED_MODEL",
    "OLLAMA_URL",
    "QDRANT_URL",
    "delete_memory",
    "embed",
    "embed_many",
    "list_collections",
    "main",
    "mcp",
    "qdrant",
    "save_memories",
    "save_memory",
    "search_memory",
    "update_memory",
]

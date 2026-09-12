"""Back-compat shim: re-export the package API under the old module name.

The implementation moved into the ``mcp_ollama_qdrant`` package
(embedding.py / store.py / server.py). Importing ``mcp_server`` still works
for the repo-root entry point and the live smoke test.
"""

from mcp_ollama_qdrant import *  # noqa: F401,F403
from mcp_ollama_qdrant import (  # noqa: F401  (explicit for tools)
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_URL,
    QDRANT_URL,
    delete_memory,
    list_collections,
    main,
    mcp,
    qdrant,
    save_memories,
    save_memory,
    search_memory,
    update_memory,
)

if __name__ == "__main__":
    main()

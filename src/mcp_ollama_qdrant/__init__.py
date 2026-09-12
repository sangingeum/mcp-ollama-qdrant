"""Console entry point for the ``mcp-ollama-qdrant`` script.

Thin re-export: the implementation lives in mcp_server.py at the repo root
(the server is a single-file MCP server, not a package layout).
"""

import os
import sys

# Allow running the installed script from a source checkout (repo root
# contains mcp_server.py directly, not under src/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mcp_server import main  # noqa: E402,F401

if __name__ == "__main__":
    main()

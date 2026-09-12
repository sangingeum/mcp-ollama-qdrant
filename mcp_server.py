"""MCP server integrating Ollama embeddings with Qdrant vector memory.

Stdio-transport MCP server exposing two tools:

- ``save_memory``  — embed text via Ollama and upsert into Qdrant.
- ``search_memory`` — embed a query and return nearest neighbours.

Configuration resolution order (highest wins):
    CLI flags -> environment variables -> built-in defaults.

Environment variables:
    OLLAMA_URL      (default http://192.168.1.103:11434)
    QDRANT_URL      (default http://192.168.1.105:6333)
    EMBED_MODEL     (default qwen3-embedding:8b)
    COLLECTION_NAME (default agent_scenarios)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from typing import Any

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from mcp.server.fastmcp import FastMCP

# All diagnostics go to stderr — stdout carries the stdio MCP transport.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp-ollama-qdrant")

# ---------------------------------------------------------------------------
# Configuration (env first, overridden by CLI)
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, str] = {
    "ollama_url": "http://192.168.1.103:11434",
    "qdrant_url": "http://192.168.1.105:6333",
    "embed_model": "qwen3-embedding:8b",
    "collection_name": "agent_scenarios",
}


def load_config(argv: list[str] | None = None) -> dict[str, str]:
    """Resolve configuration from env vars, overridden by CLI flags."""
    cfg = {key: os.environ.get(key.upper(), default) for key, default in _DEFAULTS.items()}

    parser = argparse.ArgumentParser(
        description="MCP server: Ollama embeddings + Qdrant vector memory"
    )
    parser.add_argument("--ollama-url", default=None, help="Ollama base URL")
    parser.add_argument("--qdrant-url", default=None, help="Qdrant base URL")
    parser.add_argument("--embed-model", default=None, help="Embedding model name")
    parser.add_argument("--collection", default=None, help="Qdrant collection name")
    args, _unknown = parser.parse_known_args(argv)

    cli_overrides = {
        "ollama_url": args.ollama_url,
        "qdrant_url": args.qdrant_url,
        "embed_model": args.embed_model,
        "collection_name": args.collection,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


CFG = load_config()

OLLAMA_URL = CFG["ollama_url"]
QDRANT_URL = CFG["qdrant_url"]
EMBED_MODEL = CFG["embed_model"]
COLLECTION_NAME = CFG["collection_name"]

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

qdrant = QdrantClient(url=QDRANT_URL, timeout=30)
ollama_client = ollama.Client(host=OLLAMA_URL)


def embed(text: str) -> list[float]:
    """Return the embedding vector for *text*.

    Tries the modern ``client.embed()`` API first and falls back to the
    legacy ``client.embeddings()`` for older ollama servers.
    """
    try:
        res = ollama_client.embed(model=EMBED_MODEL, input=text)
        return res["embeddings"][0]
    except (AttributeError, KeyError, TypeError):
        res = ollama_client.embeddings(model=EMBED_MODEL, prompt=text)
        return res["embedding"]


def ensure_collection() -> None:
    """Create the Qdrant collection if missing, probing vector size once."""
    if not qdrant.collection_exists(COLLECTION_NAME):
        vector_size = len(embed("dimension probe"))
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created collection %r (dim=%d)", COLLECTION_NAME, vector_size)


ensure_collection()

# ---------------------------------------------------------------------------
# MCP server + tools
# ---------------------------------------------------------------------------

mcp = FastMCP("Agent Vector Memory")


def _parse_metadata(metadata: str) -> dict[str, Any]:
    """Parse a JSON metadata string; invalid input yields an empty dict."""
    if not metadata or not metadata.strip():
        return {}
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError:
        logger.warning("Invalid metadata JSON %r — storing empty metadata", metadata)
        return {}
    return parsed if isinstance(parsed, dict) else {}


@mcp.tool()
def save_memory(text: str, metadata: str = "{}") -> str:
    """새로운 문서나 시나리오 결과를 벡터 DB에 저장합니다.

    metadata는 JSON 문자열 (예: '{"source": "doc1", "tags": ["a"]}') —
    파싱 실패 시 빈 딕셔너리로 대체합니다.
    """
    meta_dict = _parse_metadata(metadata)
    try:
        vector = embed(text)
        point_id = str(uuid.uuid4())
        meta_dict["text"] = text
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=point_id, vector=vector, payload=meta_dict)],
        )
        return f"기억 저장 완료 (ID: {point_id})"
    except Exception as exc:
        logger.exception("save_memory failed")
        return f"저장 실패: {exc}"


@mcp.tool()
def search_memory(query: str, limit: int = 3) -> str:
    """질문과 의미적으로 유사한 과거 문서/시나리오를 벡터 DB에서 검색합니다."""
    try:
        query_vector = embed(query)
        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=max(1, limit),
        )
    except Exception as exc:
        logger.exception("search_memory failed")
        return f"검색 실패: {exc}"

    if not hits:
        return "관련된 기억을 찾을 수 없습니다."

    out = [f"검색 결과 ({len(hits)}건):"]
    for hit in hits:
        text = (hit.payload or {}).get("text", "")
        out.append(f"- [유사도: {hit.score:.4f}] {text}")
    return "\n".join(out)


def main() -> None:
    """CLI entry point (``mcp-ollama-qdrant``)."""
    logger.info(
        "Agent Vector Memory MCP starting — ollama=%s qdrant=%s model=%s collection=%s",
        OLLAMA_URL, QDRANT_URL, EMBED_MODEL, COLLECTION_NAME,
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

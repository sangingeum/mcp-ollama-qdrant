"""FastMCP server exposing the vector-memory tools (thin tool layer).

Six tools: save_memory, save_memories, search_memory, update_memory,
delete_memory, list_collections. Embedding and Qdrant access live in the
``embedding`` and ``store`` modules.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from typing import Any

from qdrant_client.models import PointStruct

from mcp.server.fastmcp import FastMCP

from .embedding import EMBED_MODEL, OLLAMA_URL, embed, embed_many
from .store import (
    COLLECTION_NAME,
    QDRANT_URL,
    build_filter,
    ensure_collection_for,
    parse_metadata,
    qdrant,
)

# All diagnostics go to stderr — stdout carries the stdio MCP transport.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp-ollama-qdrant")

mcp = FastMCP("Agent Vector Memory")


def _append_warning(result: str, warning: str | None) -> str:
    """Append a warning line to a result string, if any."""
    if warning:
        return f"{result}\n{warning}"
    return result


def _apply_collection(collection: str | None) -> str:
    """Resolve the effective collection name and create it if missing."""
    name = collection if collection and collection.strip() else COLLECTION_NAME
    ensure_collection_for(name, embed, embed_model=EMBED_MODEL)
    return name


@mcp.tool()
def save_memory(text: str, metadata: str = "{}", collection: str = "") -> str:
    """Save a new document or scenario outcome into the vector DB.

    metadata is a JSON string (e.g. '{"source": "doc1", "tags": ["a"]}') —
    on parse failure an empty dict is stored instead and a warning is
    returned alongside the result. If collection is given, the memory is
    stored there (created automatically if missing; default: the
    server-configured collection).
    """
    meta_dict, warning = parse_metadata(metadata)
    try:
        name = _apply_collection(collection)
        vector = embed(text)
        point_id = str(uuid.uuid4())
        meta_dict["text"] = text
        qdrant.upsert(
            collection_name=name,
            points=[PointStruct(id=point_id, vector=vector, payload=meta_dict)],
        )
        return _append_warning(f"Memory saved (ID: {point_id}, collection: {name})", warning)
    except Exception as exc:
        logger.exception("save_memory failed")
        return _append_warning(f"Save failed: {exc}", warning)


@mcp.tool()
def save_memories(texts: list[str], metadata: str = "{}", collection: str = "") -> str:
    """Save multiple documents into the vector DB in one batch.

    All texts are embedded in a single Ollama call and upserted together.
    metadata is a JSON string applied to every document. If collection is
    given, the memories are stored there (created automatically if missing).
    """
    meta_dict, warning = parse_metadata(metadata)
    if not texts:
        return _append_warning("No texts to save.", warning)
    try:
        name = _apply_collection(collection)
        vectors = embed_many(texts)
        points = []
        for text, vector in zip(texts, vectors, strict=True):
            payload = dict(meta_dict)
            payload["text"] = text
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
        qdrant.upsert(collection_name=name, points=points)
        ids = [str(p.id) for p in points]
        return _append_warning(
            f"Saved {len(points)} memories (collection: {name}, IDs: {', '.join(ids)})",
            warning,
        )
    except Exception as exc:
        logger.exception("save_memories failed")
        return _append_warning(f"Batch save failed: {exc}", warning)


@mcp.tool()
def search_memory(query: str, limit: int = 3, filter: str = "", collection: str = "") -> str:
    """Search the vector DB for past documents/scenarios semantically similar to a query.

    filter is an optional payload-filter JSON string (e.g.
    '{"tags": ["x"]}' — only items whose tags field contains "x").
    List values use MatchAny, scalars use exact match, and multiple
    conditions are AND-ed. On parse failure the search runs without a filter
    and a warning is returned alongside the results.
    If collection is given, the search runs there (default: the
    server-configured collection).
    """
    qdrant_filter, warning = build_filter(filter)
    try:
        name = collection if collection and collection.strip() else COLLECTION_NAME
        query_vector = embed(query)
        response = qdrant.query_points(
            collection_name=name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=max(1, limit),
            with_payload=True,
        )
        hits = response.points
    except Exception as exc:
        logger.exception("search_memory failed")
        return _append_warning(f"Search failed: {exc}", warning)

    if not hits:
        return _append_warning("No matching memories found.", warning)

    out = [f"Search results ({len(hits)} hits, collection: {name}):"]
    for hit in hits:
        payload = hit.payload or {}
        text = payload.get("text", "")
        meta = {k: v for k, v in payload.items() if k != "text"}
        meta_str = json.dumps(meta, ensure_ascii=False) if meta else "{}"
        out.append(
            f"- [ID: {hit.id}] [score: {hit.score:.4f}] "
            f"metadata: {meta_str} | content: {text}"
        )
    return _append_warning("\n".join(out), warning)


@mcp.tool()
def update_memory(point_id: str, text: str | None = None, metadata: str = "", collection: str = "") -> str:
    """Update an existing memory (point) in place under the same ID.

    If point_id does not exist, an error is returned (existence is checked
    first). When text is given, it is re-embedded and overwrites the stored
    text; when text is None the existing text and vector are kept. When
    metadata is given, the payload metadata is replaced entirely; when left
    empty ("") the existing metadata is kept. Passing both text=None and an
    empty metadata is an error (nothing to update).
    If collection is given, the update happens there.
    """
    try:
        if (text is None or not text.strip()) and not (metadata and metadata.strip()):
            return "Error: nothing to update — provide new text, new metadata, or both."
        name = collection if collection and collection.strip() else COLLECTION_NAME
        if not qdrant.collection_exists(name):
            return f"Update failed: collection {name!r} does not exist."
        existing = qdrant.retrieve(collection_name=name, ids=[point_id], with_payload=True)
        if not existing:
            return f"Update failed: point_id {point_id} not found."
        point = existing[0]
        if metadata and metadata.strip():
            meta_dict, warning = parse_metadata(metadata)
            payload: dict[str, Any] = dict(meta_dict)
        else:
            warning = None
            payload = dict(point.payload or {})
        if text is None or not text.strip():
            # Metadata-only update: keep the existing text and vector, replace
            # the payload without re-embedding.
            payload["text"] = (point.payload or {}).get("text", "") if metadata and metadata.strip() else payload["text"]
            qdrant.set_payload(collection_name=name, payload=payload, points=[point_id])
            return _append_warning(f"Memory updated (ID: {point_id}, collection: {name})", warning)
        payload["text"] = text
        vector = embed(text)
        qdrant.upsert(
            collection_name=name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return _append_warning(f"Memory updated (ID: {point_id}, collection: {name})", warning)
    except Exception as exc:
        logger.exception("update_memory failed")
        return f"Update failed: {exc}"


@mcp.tool()
def delete_memory(point_id: str, collection: str = "") -> str:
    """Delete a stored memory (point) from the vector DB by ID.

    If point_id does not exist, an error is returned (existence is checked
    first). If collection is given, the deletion happens there (default: the
    server-configured collection).
    """
    try:
        name = collection if collection and collection.strip() else COLLECTION_NAME
        if not qdrant.collection_exists(name):
            return f"Delete failed: collection {name!r} does not exist."
        existing = qdrant.retrieve(collection_name=name, ids=[point_id], with_payload=False)
        if not existing:
            return f"Error: point_id {point_id} not found."
        qdrant.delete(collection_name=name, points_selector=[point_id])
        return f"Memory deleted (ID: {point_id}, collection: {name})"
    except Exception as exc:
        logger.exception("delete_memory failed")
        return f"Delete failed: {exc}"


@mcp.tool()
def list_collections() -> str:
    """List all collections currently present in Qdrant."""
    try:
        collections = qdrant.get_collections().collections
        if not collections:
            return "No collections found."
        return "Collections: " + ", ".join(c.name for c in collections)
    except Exception as exc:
        logger.exception("list_collections failed")
        return f"Failed to list collections: {exc}"


def main() -> None:
    """CLI entry point (``mcp-ollama-qdrant``)."""
    logger.info(
        "Agent Vector Memory MCP starting — ollama=%s qdrant=%s model=%s collection=%s",
        OLLAMA_URL, QDRANT_URL, EMBED_MODEL, COLLECTION_NAME,
    )
    mcp.run(transport="stdio")

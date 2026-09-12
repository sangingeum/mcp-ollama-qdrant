"""Qdrant collection management and payload helpers.

Owns the Qdrant client and the collection lifecycle: creation with a
dimension probe, dimension-mismatch fail-fast, and metadata/filter parsing
used by the tool layer.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

logger = logging.getLogger("mcp-ollama-qdrant")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://192.168.1.105:6333")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "agent_scenarios")

# Single shared Qdrant client; tests replace this attribute.
qdrant: QdrantClient = QdrantClient(url=QDRANT_URL, timeout=30)


def ensure_collection_for(
    collection_name: str,
    embed_fn,
    client: QdrantClient | None = None,
    embed_model: str = "",
) -> None:
    """Create *collection_name* if missing; verify dimension safety if it exists.

    Applies to both the default collection (at startup) and any ad-hoc
    collection named in a tool call (created on first save). On an existing
    collection whose vector size does not match the current embedding model's
    actual vector length, fail fast with a clear error — silent
    dimension-mismatched upserts would corrupt the index.
    """
    client = client or qdrant
    if not client.collection_exists(collection_name):
        vector_size = len(embed_fn("dimension probe"))
        from qdrant_client.models import VectorParams

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created collection %r (dim=%d)", collection_name, vector_size)
        return

    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    # Named-vector collections store a dict; we use a single unnamed vector.
    if isinstance(vectors, dict):
        vectors = vectors.get("")
    if vectors is None:
        sys.stderr.write(
            f"[mcp-ollama-qdrant] fatal error: collection {collection_name!r} "
            f"has no default vector configuration (appears to be a named-vectors collection).\n"
            f"  fix: delete and recreate the collection, or pass --collection with a different name.\n"
        )
        sys.exit(1)

    actual_dim = len(embed_fn("dimension probe"))
    existing_size = int(vectors.size)
    existing_distance = vectors.distance
    if existing_size != actual_dim:
        sys.stderr.write(
            f"[mcp-ollama-qdrant] fatal error (dimension mismatch): collection "
            f"{collection_name!r} was created with {existing_size} dimensions, but the "
            f"current embedding model {embed_model!r} produces {actual_dim}-dimensional "
            f"vectors. Continuing would store broken vectors and corrupt search.\n"
            f"  fix (pick one):\n"
            f"    1) Delete and recreate the collection if the old data is not needed "
            f"(e.g. Qdrant DELETE /collections/{collection_name}).\n"
            f"    2) Change the model: set EMBED_MODEL back to a "
            f"{existing_size}-dimensional embedding model.\n"
        )
        sys.exit(1)
    if existing_distance != Distance.COSINE:
        logger.warning(
            "Collection %r distance is not COSINE (%s) — similarity ranking may differ from expectations.",
            collection_name, existing_distance,
        )
    logger.info(
        "Collection %r OK (dim=%d, distance=%s, model=%r)",
        collection_name, existing_size, existing_distance, embed_model,
    )


WARN_INVALID_METADATA = "Warning: failed to parse metadata JSON; saved with empty metadata"
WARN_INVALID_METADATA_TYPE = "Warning: metadata is not a JSON object (dict); saved with empty metadata"
WARN_INVALID_FILTER = "Warning: failed to parse filter JSON; searching without a filter"


def parse_metadata(metadata: str | dict[str, Any] | None) -> tuple[dict[str, Any], str | None]:
    """Parse metadata, tolerating both a JSON string and an already-parsed dict.

    Agents frequently send metadata as a JSON object instead of a string;
    both are accepted. Returns ``(dict, warning)``; invalid input yields an
    empty dict plus a warning line to surface in the tool's return value.
    """
    if metadata is None:
        return {}, None
    if isinstance(metadata, dict):
        return dict(metadata), None
    if not isinstance(metadata, str) or not metadata.strip():
        return {}, None
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError:
        logger.warning("Invalid metadata JSON %r — storing empty metadata", metadata)
        return {}, WARN_INVALID_METADATA
    if not isinstance(parsed, dict):
        return {}, WARN_INVALID_METADATA_TYPE
    return parsed, None


def build_filter(filter_json: str | dict[str, Any] | None) -> tuple[Filter | None, str | None]:
    """Build a Qdrant :class:`Filter` from a JSON string or an already-parsed dict.

    Accepts ``{"field": value}`` pairs. A list value becomes ``MatchAny``
    (field matches any of the values), a scalar becomes ``MatchValue``.
    All conditions are AND-ed (Filter.must). Invalid input yields
    ``(None, warning)`` (search without a filter) so the caller can surface
    the warning.
    """
    if filter_json is None:
        return None, None
    if isinstance(filter_json, dict):
        parsed = filter_json
    elif isinstance(filter_json, str) and filter_json.strip():
        try:
            parsed = json.loads(filter_json)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid filter JSON %r (%s) — ignoring filter", filter_json, exc)
            return None, WARN_INVALID_FILTER
    else:
        return None, None
    if not isinstance(parsed, dict) or not parsed:
        return None, None
    conditions = []
    for key, value in parsed.items():
        if isinstance(value, list):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions), None

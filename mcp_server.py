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
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

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


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in a single Ollama call.

    ``client.embed()`` natively accepts ``input: Sequence[str]``.
    """
    try:
        res = ollama_client.embed(model=EMBED_MODEL, input=texts)
        return res["embeddings"]
    except (AttributeError, KeyError, TypeError):
        # Legacy fallback: one call per text.
        return [ollama_client.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]


def ensure_collection_for(collection_name: str) -> None:
    """Create *collection_name* if missing; verify dimension safety if it exists.

    Applies to both the default collection (at startup) and any ad-hoc
    collection named in a tool call (created on first save). On an existing
    collection whose vector size/distance does not match the current
    ``EMBED_MODEL``'s actual embedding length, fail fast with a clear error —
    silent dimension-mismatched upserts would corrupt the index.
    """
    if not qdrant.collection_exists(collection_name):
        vector_size = len(embed("dimension probe"))
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created collection %r (dim=%d)", collection_name, vector_size)
        return

    info = qdrant.get_collection(collection_name)
    vectors = info.config.params.vectors
    # Named-vector collections store a dict; we use a single unnamed vector.
    if isinstance(vectors, dict):
        vectors = vectors.get("")
    if vectors is None:
        sys.stderr.write(
            f"[mcp-ollama-qdrant] 치명적 오류: 컬렉션 {collection_name!r}이(가) "
            f"기본 벡터 설정을 찾을 수 없습니다 (named vectors 컬렉션인 듯함).\n"
            f"  해결: 컬렉션을 삭제 후 재생성하거나 --collection 으로 다른 이름을 지정하세요.\n"
        )
        sys.exit(1)

    actual_dim = len(embed("dimension probe"))
    existing_size = int(vectors.size)
    existing_distance = vectors.distance
    if existing_size != actual_dim:
        sys.stderr.write(
            f"[mcp-ollama-qdrant] 치명적 오류 (dimension mismatch): 컬렉션 "
            f"{collection_name!r}은(는) {existing_size}차원으로 생성되어 있으나, "
            f"현재 EMBED_MODEL {EMBED_MODEL!r}의 임베딩 길이는 {actual_dim}입니다. "
            f"계속하면 벡터가 잘못 저장되어 검색이 깨집니다.\n"
            f"  해결 방법 (택일):\n"
            f"    1) 컬렉션 삭제 후 재생성: 기존 데이터가 필요 없다면 컬렉션을 삭제하세요 "
            f"(예: qdrant의 DELETE /collections/{collection_name}).\n"
            f"    2) 모델 변경: EMBED_MODEL을 기존 {existing_size}차원 임베딩 모델로 되돌리세요.\n"
        )
        sys.exit(1)
    if existing_distance != Distance.COSINE:
        logger.warning(
            "컬렉션 %r 거리가 COSINE이 아닙니다 (%s) — 유사도 순위가 예상과 다를 수 있습니다.",
            collection_name, existing_distance,
        )
    logger.info(
        "Collection %r OK (dim=%d, distance=%s, model=%r)",
        collection_name, existing_size, existing_distance, EMBED_MODEL,
    )


ensure_collection_for(COLLECTION_NAME)

# ---------------------------------------------------------------------------
# MCP server + tools
# ---------------------------------------------------------------------------

mcp = FastMCP("Agent Vector Memory")

_WARN_INVALID_METADATA = "경고: metadata JSON 파싱 실패, 빈 메타데이터로 저장됨"
_WARN_INVALID_METADATA_TYPE = "경고: metadata가 JSON 객체(dict)가 아니어서 빈 메타데이터로 저장됨"
_WARN_INVALID_FILTER = "경고: filter JSON 파싱 실패, 필터 없이 검색함"


def _parse_metadata(metadata: str | None) -> tuple[dict[str, Any], str | None]:
    """Parse a JSON metadata string.

    Returns ``(dict, warning)``; invalid input yields an empty dict plus a
    Korean warning line to surface in the tool's return value.
    """
    if not metadata or not metadata.strip():
        return {}, None
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError:
        logger.warning("Invalid metadata JSON %r — storing empty metadata", metadata)
        return {}, _WARN_INVALID_METADATA
    if not isinstance(parsed, dict):
        return {}, _WARN_INVALID_METADATA_TYPE
    return parsed, None


def _build_filter(filter_json: str | None) -> tuple[Filter | None, str | None]:
    """Build a Qdrant :class:`Filter` from a JSON string.

    Accepts ``{"field": value}`` pairs. A list value becomes ``MatchAny``
    (필드에 목록 값 중 하나라도 일치), a scalar becomes ``MatchValue``.
    All conditions are AND-ed (Filter.must). Invalid input yields
    ``(None, warning)`` (필터 없이 검색) so the caller can surface the warning.
    """
    if not filter_json or not filter_json.strip():
        return None, None
    try:
        parsed = json.loads(filter_json)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid filter JSON %r (%s) — ignoring filter", filter_json, exc)
        return None, _WARN_INVALID_FILTER
    if not isinstance(parsed, dict) or not parsed:
        return None, None
    conditions = []
    for key, value in parsed.items():
        if isinstance(value, list):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions), None


def _append_warning(result: str, warning: str | None) -> str:
    """Append a warning line to a Korean result string, if any."""
    if warning:
        return f"{result}\n{warning}"
    return result


def _apply_collection(collection: str | None) -> str:
    """Resolve the effective collection name and create it if missing."""
    name = collection if collection and collection.strip() else COLLECTION_NAME
    ensure_collection_for(name)
    return name


@mcp.tool()
def save_memory(text: str, metadata: str = "{}", collection: str = "") -> str:
    """새로운 문서나 시나리오 결과를 벡터 DB에 저장합니다.

    metadata는 JSON 문자열 (예: '{"source": "doc1", "tags": ["a"]}') —
    파싱 실패 시 빈 딕셔너리로 대체하며 경고를 함께 반환합니다.
    collection을 지정하면 해당 컬렉션에 저장합니다 (없으면 자동 생성,
    기본값: 서버 설정 컬렉션).
    """
    meta_dict, warning = _parse_metadata(metadata)
    try:
        name = _apply_collection(collection)
        vector = embed(text)
        point_id = str(uuid.uuid4())
        meta_dict["text"] = text
        qdrant.upsert(
            collection_name=name,
            points=[PointStruct(id=point_id, vector=vector, payload=meta_dict)],
        )
        return _append_warning(f"기억 저장 완료 (ID: {point_id}, 컬렉션: {name})", warning)
    except Exception as exc:
        logger.exception("save_memory failed")
        return _append_warning(f"저장 실패: {exc}", warning)


@mcp.tool()
def save_memories(texts: list[str], metadata: str = "{}", collection: str = "") -> str:
    """여러 문서를 한 번에 벡터 DB에 저장합니다 (배치).

    모든 texts를 하나의 Ollama 임베딩 호출로 처리한 뒤 한 번에 upsert합니다.
    metadata는 모든 문서에 공통으로 적용되는 JSON 문자열입니다.
    collection을 지정하면 해당 컬렉션에 저장합니다 (없으면 자동 생성).
    """
    meta_dict, warning = _parse_metadata(metadata)
    if not texts:
        return _append_warning("저장할 텍스트가 없습니다.", warning)
    try:
        name = _apply_collection(collection)
        vectors = embed_batch(texts)
        points = []
        for text, vector in zip(texts, vectors, strict=True):
            payload = dict(meta_dict)
            payload["text"] = text
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
        qdrant.upsert(collection_name=name, points=points)
        ids = [str(p.id) for p in points]
        return _append_warning(
            f"기억 {len(points)}건 저장 완료 (컬렉션: {name}, IDs: {', '.join(ids)})",
            warning,
        )
    except Exception as exc:
        logger.exception("save_memories failed")
        return _append_warning(f"배치 저장 실패: {exc}", warning)


@mcp.tool()
def search_memory(query: str, limit: int = 3, filter: str = "", collection: str = "") -> str:
    """질문과 의미적으로 유사한 과거 문서/시나리오를 벡터 DB에서 검색합니다.

    filter는 선택적 payload 필터 JSON 문자열입니다 (예:
    '{"tags": ["x"]}' — tags 필드가 "x"를 포함하는 항목만 검색).
    목록 값은 MatchAny, 단일 값은 일치 조건으로 처리되며, 여러 조건은 AND로 결합됩니다.
    파싱 실패 시 필터 없이 검색하며 경고를 함께 반환합니다.
    collection을 지정하면 해당 컬렉션에서 검색합니다 (기본값: 서버 설정 컬렉션).
    """
    qdrant_filter, warning = _build_filter(filter)
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
        return _append_warning(f"검색 실패: {exc}", warning)

    if not hits:
        return _append_warning("관련된 기억을 찾을 수 없습니다.", warning)

    out = [f"검색 결과 ({len(hits)}건, 컬렉션: {name}):"]
    for hit in hits:
        payload = hit.payload or {}
        text = payload.get("text", "")
        meta = {k: v for k, v in payload.items() if k != "text"}
        meta_str = json.dumps(meta, ensure_ascii=False) if meta else "{}"
        out.append(
            f"- [ID: {hit.id}] [유사도: {hit.score:.4f}] "
            f"메타데이터: {meta_str} | 내용: {text}"
        )
    return _append_warning("\n".join(out), warning)


@mcp.tool()
def update_memory(point_id: str, text: str, metadata: str = "", collection: str = "") -> str:
    """기존 기억(포인트)을 제자리에 갱신합니다: 재임베딩 후 같은 ID로 덮어씁니다.

    point_id가 존재하지 않으면 오류를 반환합니다 (존재 여부를 먼저 확인).
    metadata를 지정하면 payload의 메타데이터를 통째로 교체하고,
    비워 두면(\"\") 기존 메타데이터를 유지합니다. text는 항상 새 값으로 교체됩니다.
    collection을 지정하면 해당 컬렉션에서 갱신합니다.
    """
    try:
        name = collection if collection and collection.strip() else COLLECTION_NAME
        if not qdrant.collection_exists(name):
            return f"갱신 실패: 컬렉션 {name!r}이(가) 없습니다."
        existing = qdrant.retrieve(collection_name=name, ids=[point_id], with_payload=True)
        if not existing:
            return f"갱신 실패: point_id {point_id}을(를) 찾을 수 없습니다."
        point = existing[0]
        payload = dict(point.payload or {})
        if metadata and metadata.strip():
            meta_dict, warning = _parse_metadata(metadata)
            payload = meta_dict
        else:
            warning = None
        vector = embed(text)
        payload["text"] = text
        qdrant.upsert(
            collection_name=name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return _append_warning(f"기억 갱신 완료 (ID: {point_id}, 컬렉션: {name})", warning)
    except Exception as exc:
        logger.exception("update_memory failed")
        return f"갱신 실패: {exc}"


@mcp.tool()
def delete_memory(point_id: str, collection: str = "") -> str:
    """ID로 저장된 기억(포인트)을 벡터 DB에서 삭제합니다.

    collection을 지정하면 해당 컬렉션에서 삭제합니다 (기본값: 서버 설정 컬렉션).
    """
    try:
        name = collection if collection and collection.strip() else COLLECTION_NAME
        qdrant.delete(collection_name=name, points_selector=[point_id])
        return f"기억 삭제 완료 (ID: {point_id}, 컬렉션: {name})"
    except Exception as exc:
        logger.exception("delete_memory failed")
        return f"삭제 실패: {exc}"


@mcp.tool()
def list_collections() -> str:
    """Qdrant에 현재 존재하는 모든 컬렉션 목록을 반환합니다."""
    try:
        collections = qdrant.get_collections().collections
        if not collections:
            return "컬렉션이 없습니다."
        return "컬렉션 목록: " + ", ".join(c.name for c in collections)
    except Exception as exc:
        logger.exception("list_collections failed")
        return f"컬렉션 조회 실패: {exc}"


def main() -> None:
    """CLI entry point (``mcp-ollama-qdrant``)."""
    logger.info(
        "Agent Vector Memory MCP starting — ollama=%s qdrant=%s model=%s collection=%s",
        OLLAMA_URL, QDRANT_URL, EMBED_MODEL, COLLECTION_NAME,
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

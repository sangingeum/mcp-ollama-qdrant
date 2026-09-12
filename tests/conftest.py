"""Patch fake Ollama/Qdrant clients BEFORE mcp_server is imported.

Importing mcp_server runs the startup collection check, which would touch
live services. Replacing the clients on the embedding/store modules first
keeps every pytest run fully offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qdrant_client.models import Distance, Record, ScoredPoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_ollama_qdrant import embedding as _emb
from mcp_ollama_qdrant import store as _store


class FakeEmbedder:
    """Deterministic offline embedder mimicking the Ollama client API."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        return [float((len(text) + i) % self.dim) / self.dim for i in range(self.dim)]

    def embed(self, model=None, input=None, **kw):
        items = input if isinstance(input, list) else [input]
        return {"embeddings": [self._vec(t) for t in items]}

    def embeddings(self, model=None, prompt=None, **kw):
        return {"embedding": self._vec(prompt)}


class FakeQdrant:
    """In-memory stand-in for QdrantClient covering the calls the tools make."""

    def __init__(self):
        self.collections: dict[str, dict] = {}
        self.existing_dims: dict[str, int] = {}
        self.fail_get_collection = False

    def collection_exists(self, name):
        return name in self.collections or name in self.existing_dims

    def create_collection(self, collection_name, vectors_config, **kw):
        self.collections[collection_name] = {"dim": vectors_config.size, "points": {}}

    def _make_info(self, name, dim, distance=Distance.COSINE, points=None):
        dim_val = dim
        dist_val = distance

        class V:
            size = dim_val
            distance = dist_val

        class Params:
            vectors = V()

        class Config:
            params = Params()

        class Info:
            config = Config()
            status = "green"
            points_count = len(points or {})

        return Info()

    def get_collection(self, name):
        if self.fail_get_collection:
            raise RuntimeError("connection refused")
        if name in self.existing_dims:
            return self._make_info(name, self.existing_dims[name])
        if name in self.collections:
            return self._make_info(name, self.collections[name]["dim"], points=self.collections[name]["points"])
        raise RuntimeError(f"collection {name} not found")

    def get_collections(self):
        class C:
            def __init__(self, name):
                self.name = name

        class R:
            def __init__(self, names):
                self.collections = [C(n) for n in names]

        return R(list(self.collections) + list(self.existing_dims))

    def upsert(self, collection_name, points, **kw):
        coll = self.collections.setdefault(collection_name, {"dim": 8, "points": {}})
        for p in points:
            coll["points"][str(p.id)] = dict(p.payload or {})

    def retrieve(self, collection_name, ids, with_payload=False, **kw):
        coll = self.collections.get(collection_name, {"points": {}})
        return [
            Record(id=pid, payload=coll["points"][str(pid)], vector=None)
            for pid in ids
            if str(pid) in coll["points"]
        ]

    def delete(self, collection_name, points_selector, **kw):
        coll = self.collections.get(collection_name, {"points": {}})
        for pid in points_selector:
            coll["points"].pop(str(pid), None)

    def query_points(self, collection_name, query, query_filter=None, limit=3, with_payload=False, **kw):
        coll = self.collections.get(collection_name)
        if coll is None:
            raise RuntimeError(f"collection {collection_name} not found")

        def keep(payload):
            if not query_filter:
                return True
            for cond in query_filter.must:
                val = payload.get(cond.key)
                match = cond.match
                if getattr(match, "any", None) is not None:
                    if not isinstance(val, list) or not any(v in val for v in match.any):
                        return False
                else:
                    if val != match.value:
                        return False
            return True

        hits = [
            ScoredPoint(id=pid, version=1, score=round(1.0 - i * 0.1, 4), payload=pl, vector=None)
            for i, (pid, pl) in enumerate(coll["points"].items())
            if keep(pl)
        ]
        return type("R", (), {"points": hits[:limit]})()


# Patch the module-level clients used by the package and the back-compat shim.
_fake_embedder = FakeEmbedder()
_fake_qdrant = FakeQdrant()
_emb.ollama_client = _fake_embedder
_store.qdrant = _fake_qdrant

import mcp_server  # noqa: E402

# server.py bound the store client via `from .store import qdrant`; point the
# same name at the fake so tool functions use the in-memory store too.
mcp_server.qdrant = _fake_qdrant

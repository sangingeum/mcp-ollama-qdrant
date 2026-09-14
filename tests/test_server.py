"""Offline pytest suite: Embedder and Qdrant are fully mocked, no network.

Fakes live in conftest.py (patched in before mcp_server is imported so the
startup collection check never touches the network).
"""

from __future__ import annotations

import json

import pytest

import mcp_server as s
from mcp_ollama_qdrant import embedding as emb
from tests.conftest import FakeEmbedder, FakeQdrant


@pytest.fixture
def env(monkeypatch):
    """Wire a fresh fake embedder + fake qdrant into the server modules."""
    embedder = FakeEmbedder(dim=8)
    fq = FakeQdrant()
    monkeypatch.setattr(emb, "ollama_client", embedder)
    # Point every alias the tools read at the fresh fake.
    import mcp_ollama_qdrant.server as srv
    from mcp_ollama_qdrant import store
    monkeypatch.setattr(store, "qdrant", fq)   # used by ensure_collection_for
    monkeypatch.setattr(s, "qdrant", fq)
    monkeypatch.setattr(srv, "qdrant", fq)
    # Pre-create the default collection so per-test saves go through the
    # existing-collection path without touching the network.
    fq.collections[s.COLLECTION_NAME] = {"dim": embedder.dim, "points": {}}
    yield fq, embedder


# ---------------------------------------------------------------------------
# update_memory
# ---------------------------------------------------------------------------

def test_update_preserves_metadata(env):
    fq, _ = env
    saved = s.save_memory("doc one", json.dumps({"source": "a", "tags": ["t1"]}))
    assert "Memory saved" in saved
    pid = saved.split("ID: ")[1].split(",")[0].strip()

    out = s.update_memory(pid, "doc one revised")
    assert "Memory updated" in out

    stored = fq.collections[s.COLLECTION_NAME]["points"][pid]
    assert stored["text"] == "doc one revised"
    assert stored["source"] == "a"
    assert stored["tags"] == ["t1"]


def test_update_replaces_metadata(env):
    fq, _ = env
    saved = s.save_memory("doc two", json.dumps({"source": "a"}))
    pid = saved.split("ID: ")[1].split(",")[0].strip()

    out = s.update_memory(pid, "doc two revised", json.dumps({"source": "b", "v": 2}))
    assert "Memory updated" in out

    stored = fq.collections[s.COLLECTION_NAME]["points"][pid]
    assert stored["source"] == "b"
    assert stored["v"] == 2
    assert stored["text"] == "doc two revised"


def test_update_nonexistent_id_error(env):
    out = s.update_memory("00000000-0000-0000-0000-000000000000", "nope")
    assert "Update failed" in out
    assert "not found" in out


def test_update_metadata_only_keeps_text(env):
    fq, _ = env
    saved = s.save_memory("metadata-only doc", json.dumps({"source": "a"}))
    pid = saved.split("ID: ")[1].split(",")[0].strip()

    out = s.update_memory(pid, None, json.dumps({"source": "b", "v": 9}))
    assert "Memory updated" in out

    stored = fq.collections[s.COLLECTION_NAME]["points"][pid]
    assert stored["text"] == "metadata-only doc"  # text untouched
    assert stored["source"] == "b"
    assert stored["v"] == 9


def test_update_both_none_error(env):
    out = s.update_memory("00000000-0000-0000-0000-000000000000")
    assert out.startswith("Error: nothing to update")


def test_delete_nonexistent_id_error(env):
    out = s.delete_memory("00000000-0000-0000-0000-000000000000")
    assert "Error: point_id 00000000-0000-0000-0000-000000000000 not found" in out


def test_delete_existing_ok(env):
    fq, _ = env
    saved = s.save_memory("to be deleted")
    pid = saved.split("ID: ")[1].split(",")[0].strip()
    out = s.delete_memory(pid)
    assert "Memory deleted" in out
    assert pid not in fq.collections[s.COLLECTION_NAME]["points"]


def test_update_nonexistent_collection_error(env):
    out = s.update_memory("abc", "nope", collection="missing_coll")
    assert "Update failed" in out
    assert "does not exist" in out


# ---------------------------------------------------------------------------
# multi-collection
# ---------------------------------------------------------------------------

def test_save_and_search_in_secondary_collection(env):
    fq, _ = env
    out = s.save_memory("secondary doc", json.dumps({"scope": "c2"}), collection="coll_b")
    assert "Memory saved" in out
    assert "coll_b" in out
    assert fq.collections["coll_b"]["points"]

    results = s.search_memory("secondary doc", limit=3, collection="coll_b")
    assert "Search results" in results
    assert "secondary doc" in results

    pid = out.split("ID: ")[1].split(",")[0].strip()
    upd = s.update_memory(pid, "secondary doc v2", collection="coll_b")
    assert "Memory updated" in upd

    dele = s.delete_memory(pid, collection="coll_b")
    assert "Memory deleted" in dele
    assert pid not in fq.collections["coll_b"]["points"]


def test_secondary_collection_created_on_save(env):
    fq, embedder = env
    s.save_memory("auto create me", collection="fresh_coll")
    assert fq.collections["fresh_coll"]["dim"] == embedder.dim


# ---------------------------------------------------------------------------
# tolerant metadata/filter types (JSON string OR object)
# ---------------------------------------------------------------------------

def test_save_metadata_as_dict(env):
    fq, _ = env
    out = s.save_memory("dict meta doc", {"source": "obj", "tags": ["t"]})
    assert "Memory saved" in out
    pid = out.split("ID: ")[1].split(",")[0].strip()
    stored = fq.collections[s.COLLECTION_NAME]["points"][pid]
    assert stored["source"] == "obj"
    assert stored["tags"] == ["t"]


def test_save_memories_metadata_as_dict(env):
    fq, _ = env
    out = s.save_memories(["a", "b"], {"src": "batch-obj"})
    assert out.startswith("Saved 2 memories")
    ids = out.split("IDs: ")[1].rstrip(")").split(", ")
    stored = [fq.collections[s.COLLECTION_NAME]["points"][pid] for pid in ids]
    assert all(p["src"] == "batch-obj" for p in stored)


def test_search_filter_as_dict(env):
    s.save_memory("filtered doc", {"tags": ["only-this"]})
    out = s.search_memory("filtered doc", filter={"tags": ["only-this"]})
    assert "filtered doc" in out
    out2 = s.search_memory("filtered doc", filter={"tags": ["nope"]})
    assert "No matching memories found." in out2


def test_search_project_scoping(env):
    """project= ANDs an exact 'project' condition with any filter."""
    s.save_memory("alpha note", {"project": "alpha", "type": "decision"})
    s.save_memory("beta note", {"project": "beta", "type": "decision"})
    out = s.search_memory("note", project="alpha")
    assert "alpha note" in out and "beta note" not in out
    out2 = s.search_memory("note", project="beta")
    assert "beta note" in out2 and "alpha note" not in out2
    # Unscoped search still sees both.
    out3 = s.search_memory("note")
    assert "alpha note" in out3 and "beta note" in out3


def test_search_project_combined_with_filter(env):
    s.save_memory("alpha arch", {"project": "alpha", "type": "architecture"})
    s.save_memory("alpha debug", {"project": "alpha", "type": "debugging"})
    out = s.search_memory("alpha", project="alpha",
                          filter={"type": "architecture"})
    assert "alpha arch" in out and "alpha debug" not in out


def test_update_metadata_as_dict(env):
    fq, _ = env
    saved = s.save_memory("upd dict doc", {"v": 1})
    pid = saved.split("ID: ")[1].split(",")[0].strip()
    out = s.update_memory(pid, None, {"v": 2, "src": "obj"})
    assert "Memory updated" in out
    stored = fq.collections[s.COLLECTION_NAME]["points"][pid]
    assert stored["v"] == 2
    assert stored["text"] == "upd dict doc"  # text untouched


def test_update_both_none_error_with_dict(env):
    out = s.update_memory("00000000-0000-0000-0000-000000000000", None, {})
    assert out.startswith("Error: nothing to update")


def test_json_string_still_accepted_backcompat(env):
    fq, _ = env
    out = s.save_memory("string meta", json.dumps({"k": "v"}))
    pid = out.split("ID: ")[1].split(",")[0].strip()
    assert fq.collections[s.COLLECTION_NAME]["points"][pid]["k"] == "v"


# ---------------------------------------------------------------------------
# invalid JSON warnings
# ---------------------------------------------------------------------------

def test_save_invalid_metadata_json_warning(env):
    out = s.save_memory("doc", "{not-valid-json")
    assert "Memory saved" in out
    assert "Warning: failed to parse metadata JSON; saved with empty metadata" in out


def test_save_metadata_not_dict_warning(env):
    out = s.save_memory("doc", "[1, 2, 3]")
    assert "Warning: metadata is not a JSON object (dict); saved with empty metadata" in out


def test_search_invalid_filter_json_warning(env):
    s.save_memory("tagged doc", json.dumps({"tags": ["x"]}))
    out = s.search_memory("tagged doc", filter="{bad json")
    assert "Warning: failed to parse filter JSON; searching without a filter" in out
    assert "Search failed" not in out
    assert "tagged doc" in out  # search still ran, just unfiltered


def test_search_no_hits_message(env):
    fq, _ = env
    fq.collections["empty_coll"] = {"dim": 8, "points": {}}
    out = s.search_memory("anything", collection="empty_coll")
    assert out == "No matching memories found."


# ---------------------------------------------------------------------------
# dimension-mismatch fail-fast
# ---------------------------------------------------------------------------

def test_dimension_mismatch_fails_fast(env, capsys):
    fq, embedder = env
    # Pre-existing collection with a dimension different from the fake embedder's.
    fq.existing_dims["mismatched"] = 4096
    with pytest.raises(SystemExit) as excinfo:
        s.save_memory("boom", collection="mismatched")
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "dimension mismatch" in captured.err
    assert "mismatched" in captured.err


def test_named_vectors_collection_fails_fast(env, capsys, monkeypatch):
    fq, _ = env

    class V:
        pass

    class NamedInfo:
        class config:
            class params:
                vectors = {"sparse": V()}  # dict without the "" unnamed-vector key

    monkeypatch.setattr(fq, "get_collection", lambda name: NamedInfo())
    # collection_exists must report True so the existing-collection path runs.
    monkeypatch.setattr(fq, "collection_exists", lambda name: True)
    with pytest.raises(SystemExit) as excinfo:
        s.save_memory("boom", collection="named_vec_coll")
    assert excinfo.value.code == 1
    assert "no default vector configuration" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# English message shapes
# ---------------------------------------------------------------------------

def test_save_memories_batch_message_shape(env):
    out = s.save_memories(["a", "b", "c"], json.dumps({"src": "batch"}))
    assert out.startswith("Saved 3 memories")
    assert "IDs:" in out


def test_save_memories_empty_message(env):
    out = s.save_memories([])
    assert out == "No texts to save."


def test_search_hit_line_shape(env):
    s.save_memory("findable text", json.dumps({"tags": ["x"]}))
    out = s.search_memory("findable text")
    hit = [line for line in out.splitlines() if line.startswith("- [ID:")][0]
    assert "[score: " in hit
    assert "metadata: " in hit
    assert "| content: findable text" in hit


def test_list_collections_message(env):
    fq, _ = env
    fq.collections.clear()
    fq.collections["c1"] = {"dim": 8, "points": {}}
    out = s.list_collections()
    assert out == "Collections: c1"


def test_list_collections_empty_message(env):
    fq, _ = env
    fq.collections.clear()
    out = s.list_collections()
    assert out == "No collections found."


def test_list_collections_error_is_english(env, monkeypatch):
    fq, _ = env
    monkeypatch.setattr(fq, "get_collections", lambda: (_ for _ in ()).throw(RuntimeError("connection refused")))
    out = s.list_collections()
    assert out.startswith("Failed to list collections")


# ---------------------------------------------------------------------------
# embedding fallback + tool registry
# ---------------------------------------------------------------------------

def test_legacy_embed_fallback(monkeypatch):
    calls = []

    class LegacyOnly:
        def embeddings(self, model=None, prompt=None, **kw):
            calls.append(prompt)
            return {"embedding": [0.5, 0.5]}

    monkeypatch.setattr(emb, "ollama_client", LegacyOnly())
    assert emb.embed("hello") == [0.5, 0.5]
    assert emb.embed_many(["a", "b"]) == [[0.5, 0.5], [0.5, 0.5]]
    assert len(calls) == 3  # legacy fallback goes one call per text


def test_tool_count_is_six():
    import anyio

    tools = anyio.run(s.mcp.list_tools)
    assert len(tools) == 6
    names = {t.name for t in tools}
    assert names == {
        "save_memory", "save_memories", "search_memory",
        "update_memory", "delete_memory", "list_collections",
    }

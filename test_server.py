import json, os, sys
sys.path.insert(0, ".")
import mcp_server as s

COLL2 = "test_secondary_collection"

before = s.qdrant.get_collection(s.COLLECTION_NAME)
print("BEFORE points:", before.points_count or 0)

# save single memories
r1 = s.save_memory("Agent error scenario: if the DB connection fails, retry 3 times then switch to the backup node.", json.dumps({"source":"test","tags":["scenario"]}))
r2 = s.save_memory("Today's weather retrospective: the server-room temperature check happens every Friday.", "{}")
print(r1); print(r2)

# save_memories batch
batch = s.save_memories(
    ["Batch doc A: during a queuing-system outage, check for priority inversion.",
     "Batch doc B: cache invalidation runs preemptively one minute before TTL expiry.",
     "Batch doc C: on a rolling deploy, if a health check fails, roll back to the previous version."],
    json.dumps({"source":"batch-test","tags":["batch","ops"]}),
)
print(batch)

after_batch = s.qdrant.get_collection(s.COLLECTION_NAME)
print("AFTER BATCH points:", after_batch.points_count or 0, "(delta:", (after_batch.points_count or 0) - (before.points_count or 0), ")")

# search without filter — must include ID + metadata
search_out = s.search_memory("How should I handle a DB outage?", limit=2)
print(search_out)
first_id = None
for line in search_out.splitlines():
    if line.startswith("- [ID:"):
        first_id = line.split("[ID: ")[1].split("]")[0]
        break
print("EXTRACTED first search hit ID:", first_id)
assert first_id, "search result must contain point ID"

# search with filter (tags=batch)
print("FILTERED(tags=batch):", s.search_memory("What is the deploy rollback procedure?", limit=2, filter=json.dumps({"tags":["batch"]})))
# search with filter that matches nothing
print("FILTERED(tags=nonexistent):", s.search_memory("What is the deploy rollback procedure?", limit=2, filter=json.dumps({"tags":["nonexistent-tag"]})))

# invalid metadata JSON → warning line in return, not silent silence
bad_meta = s.save_memory("This is an invalid-metadata test document.", "{not-valid-json")
print(bad_meta)
assert "Warning" in bad_meta, "invalid metadata must surface a warning line"

# invalid filter JSON → warning line
bad_filter_out = s.search_memory("deploy procedure", limit=2, filter="{bad json")
print("BAD FILTER:", bad_filter_out)
assert "Warning" in bad_filter_out, "invalid filter must surface a warning line"

# update_memory: update text, keep metadata
upd1 = s.update_memory(first_id, "If the DB connection fails, retry 5 times then perform a cluster failover.")
print(upd1)
assert "Memory updated" in upd1
# verify payload text updated, metadata preserved
pts = s.qdrant.retrieve(collection_name=s.COLLECTION_NAME, ids=[first_id], with_payload=True)
print("AFTER UPDATE payload:", pts[0].payload)
assert pts[0].payload["text"].startswith("If the DB connection fails, retry 5 times")
assert pts[0].payload.get("source") == "test", "existing metadata must be preserved when metadata param empty"

# update_memory: replace metadata
upd2 = s.update_memory(first_id, "If the DB connection fails, retry 7 times then request manual intervention.", json.dumps({"source":"updated","tags":["scenario","v3"]}))
print(upd2)
pts2 = s.qdrant.retrieve(collection_name=s.COLLECTION_NAME, ids=[first_id], with_payload=True)
print("AFTER UPDATE2 payload:", pts2[0].payload)
assert pts2[0].payload["source"] == "updated"

# update_memory: nonexistent ID → error
upd3 = s.update_memory("00000000-0000-0000-0000-000000000000", "Update attempt on a missing point")
print(upd3)
assert "Update failed" in upd3

# collection param: save into ad-hoc collection (created on save)
cs = s.save_memory("Secondary collection save test: index rebuild runs overnight.", json.dumps({"scope":"coll2"}), collection=COLL2)
print(cs)
assert COLL2 in cs
c2info = s.qdrant.get_collection(COLL2)
print("COLL2 points:", c2info.points_count, "dim:", c2info.config.params.vectors.size)
assert c2info.points_count == 1

# search in ad-hoc collection
cs_search = s.search_memory("When does the index rebuild run?", limit=2, collection=COLL2)
print("COLL2 SEARCH:", cs_search)
assert "Secondary collection" in cs_search

# update in ad-hoc collection
cs_id = cs.split("ID: ")[1].split(",")[0].strip()
cs_upd = s.update_memory(cs_id, "The index rebuild runs every day at 3 AM.", collection=COLL2)
print(cs_upd)
assert "Memory updated" in cs_upd

# delete in ad-hoc collection
cs_del = s.delete_memory(cs_id, collection=COLL2)
print(cs_del)
assert "Memory deleted" in cs_del

# list_collections
print("LIST COLLECTIONS:", s.list_collections())

# delete: extract one point id from batch result
mid = batch.split("IDs: ")[-1].split(",")[0].strip()
dres = s.delete_memory(mid)
print(dres)
after_del = s.qdrant.get_collection(s.COLLECTION_NAME)
print("AFTER DELETE points:", after_del.points_count or 0, "(delta:", (after_del.points_count or 0) - (after_batch.points_count or 0), ")")

# collection info
info = s.qdrant.get_collection(s.COLLECTION_NAME)
print("collection:", info.status, "points:", info.points_count or 0)

print("ALL TESTS PASSED")

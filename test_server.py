import json, os, sys
sys.path.insert(0, ".")
import mcp_server as s

COLL2 = "test_secondary_collection"

before = s.qdrant.get_collection(s.COLLECTION_NAME)
print("BEFORE points:", before.points_count or 0)

# save single memories
r1 = s.save_memory("에이전트 오류 시나리오: DB 연결 실패 시 재시도 3회 후 백업 노드로 전환한다.", json.dumps({"source":"test","tags":["scenario"]}))
r2 = s.save_memory("오늘 날씨 회고: 서버실 온도 점검은 매주 금요일 실시한다.", "{}")
print(r1); print(r2)

# save_memories batch
batch = s.save_memories(
    ["배치 문서 A: 큐잉 시스템 장애 시 우선순위 역전을 점검한다.",
     "배치 문서 B: 캐시 무효화는 TTL 만료 전 1분에 선제 실행한다.",
     "배치 문서 C: 롤링 배포 시 헬스체크 실패하면 이전 버전으로 롤백한다."],
    json.dumps({"source":"batch-test","tags":["batch","ops"]}),
)
print(batch)

after_batch = s.qdrant.get_collection(s.COLLECTION_NAME)
print("AFTER BATCH points:", after_batch.points_count or 0, "(delta:", (after_batch.points_count or 0) - (before.points_count or 0), ")")

# search without filter — must include ID + metadata
search_out = s.search_memory("DB 장애가 났을 때 어떻게 대처해야 하나?", limit=2)
print(search_out)
first_id = None
for line in search_out.splitlines():
    if line.startswith("- [ID:"):
        first_id = line.split("[ID: ")[1].split("]")[0]
        break
print("EXTRACTED first search hit ID:", first_id)
assert first_id, "search result must contain point ID"

# search with filter (tags=batch)
print("FILTERED(tags=batch):", s.search_memory("배포 롤백 절차는?", limit=2, filter=json.dumps({"tags":["batch"]})))
# search with filter that matches nothing
print("FILTERED(tags=nonexistent):", s.search_memory("배포 롤백 절차는?", limit=2, filter=json.dumps({"tags":["nonexistent-tag"]})))

# invalid metadata JSON → warning line in return, not silent silence
bad_meta = s.save_memory("잘못된 메타데이터 테스트 문서입니다.", "{not-valid-json")
print(bad_meta)
assert "경고" in bad_meta, "invalid metadata must surface a warning line"

# invalid filter JSON → warning line
bad_filter_out = s.search_memory("배포 절차", limit=2, filter="{bad json")
print("BAD FILTER:", bad_filter_out)
assert "경고" in bad_filter_out, "invalid filter must surface a warning line"

# update_memory: update text, keep metadata
upd1 = s.update_memory(first_id, "DB 연결 실패 시 재시도 5회 후 클러스터 페일오버를 수행한다.")
print(upd1)
assert "갱신 완료" in upd1
# verify payload text updated, metadata preserved
pts = s.qdrant.retrieve(collection_name=s.COLLECTION_NAME, ids=[first_id], with_payload=True)
print("AFTER UPDATE payload:", pts[0].payload)
assert pts[0].payload["text"].startswith("DB 연결 실패 시 재시도 5회")
assert pts[0].payload.get("source") == "test", "existing metadata must be preserved when metadata param empty"

# update_memory: replace metadata
upd2 = s.update_memory(first_id, "DB 연결 실패 시 재시도 7회 후 수동 개입을 요청한다.", json.dumps({"source":"updated","tags":["scenario","v3"]}))
print(upd2)
pts2 = s.qdrant.retrieve(collection_name=s.COLLECTION_NAME, ids=[first_id], with_payload=True)
print("AFTER UPDATE2 payload:", pts2[0].payload)
assert pts2[0].payload["source"] == "updated"

# update_memory: nonexistent ID → error
upd3 = s.update_memory("00000000-0000-0000-0000-000000000000", "없는 포인트 갱신 시도")
print(upd3)
assert "갱신 실패" in upd3

# collection param: save into ad-hoc collection (created on save)
cs = s.save_memory("보조 컬렉션 저장 테스트: 인덱스 재생성은 새벽에 수행한다.", json.dumps({"scope":"coll2"}), collection=COLL2)
print(cs)
assert COLL2 in cs
c2info = s.qdrant.get_collection(COLL2)
print("COLL2 points:", c2info.points_count, "dim:", c2info.config.params.vectors.size)
assert c2info.points_count == 1

# search in ad-hoc collection
cs_search = s.search_memory("인덱스 재생성은 언제 하나?", limit=2, collection=COLL2)
print("COLL2 SEARCH:", cs_search)
assert "보조 컬렉션" in cs_search

# update in ad-hoc collection
cs_id = cs.split("ID: ")[1].split(",")[0].strip()
cs_upd = s.update_memory(cs_id, "인덱스 재생성은 매일 새벽 3시에 수행한다.", collection=COLL2)
print(cs_upd)
assert "갱신 완료" in cs_upd

# delete in ad-hoc collection
cs_del = s.delete_memory(cs_id, collection=COLL2)
print(cs_del)
assert "삭제 완료" in cs_del

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

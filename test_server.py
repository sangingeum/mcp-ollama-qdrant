import json, os, sys
sys.path.insert(0, ".")
import mcp_server as s

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

# search without filter
print(s.search_memory("DB 장애가 났을 때 어떻게 대처해야 하나?", limit=2))
# search with filter (tags=batch)
print("FILTERED(tags=batch):", s.search_memory("배포 롤백 절차는?", limit=2, filter=json.dumps({"tags":["batch"]})))
# search with filter that matches nothing
print("FILTERED(tags=nonexistent):", s.search_memory("배포 롤백 절차는?", limit=2, filter=json.dumps({"tags":["nonexistent-tag"]})))

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

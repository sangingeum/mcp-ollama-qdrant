import json, os, sys
sys.path.insert(0, ".")
import mcp_server as s

# save a memory
r1 = s.save_memory("에이전트 오류 시나리오: DB 연결 실패 시 재시도 3회 후 백업 노드로 전환한다.", json.dumps({"source":"test","tags":["scenario"]}))
r2 = s.save_memory("오늘 날씨 회고: 서버실 온도 점검은 매주 금요일 실시한다.", "{}")
print(r1); print(r2)

# search
print(s.search_memory("DB 장애가 났을 때 어떻게 대처해야 하나?", limit=2))
print(s.search_memory("점검 일정은?", limit=1))
print(s.search_memory("전혀 관련없는 양자역학 이야기", limit=1))

# collection info
info = s.qdrant.get_collection(s.COLLECTION_NAME)
print("collection:", info.status, "points:", info.points_count)

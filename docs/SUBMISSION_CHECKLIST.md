# 제출 체크리스트

- `docker build -t rag-submission:latest .` 가 성공하는지 확인
- `/health` 가 `200 OK` 와 `status=ok` 를 반환하는지 확인
- `/predict` 가 정확히 5개의 고유한 `retrieved_chunk_ids` 를 반환하는지 확인
- 반환되는 `chunk_id` 가 실제 `raw/*_hybrid.json` 코퍼스에 존재하는지 확인
- 외부 인터넷 연결 없이 동작하는지 확인
- 생성 모델을 추가하는 경우 8B 이하 모델만 포함했는지 확인
- 응답 시간이 30초 이내인지 확인

# 꽃보다 의결 제출 세팅

이 저장소는 `인공지능 모델 개발 기획서_꽃보다 의결.pdf`의 방향을 반영해 공정위 공모전 제출용 기본 패키지로 정리되어 있습니다.

## 반영한 기획 방향

- 문제 유형: 공정거래위원회 의결서 기반 질의응답
- 구조: Hybrid RAG 중심 제출 구조
- 검색: `raw/*_hybrid.json` 코퍼스 기반 오프라인 검색
- 생성: 검색 결과 5개 청크에 근거한 Grounded Answer
- 오프라인 제약 대응: 외부 API 호출 없이 동작

## 현재 구현 범위

- `server.py`: 평가 서버가 호출할 `/health`, `/predict` 제공
- `app/retriever.py`: 실제 코퍼스 `chunk_id`를 반환하는 경량 검색기
- `app/generator.py`: 검색 청크 기반 추출형 답변 생성기
- `models/`, `index/`: 추후 BGE-M3, Qwen 계열 모델/인덱스 탑재용 디렉터리

## 기획서 대비 향후 보강 포인트

- Dense Retriever와 Sparse Retriever를 분리 구현하고 RRF 융합 적용
- QueryRouter를 추가해 질의 유형별 검색 전략 분기
- Qwen 계열 8B 이하 모델 또는 양자화 모델을 `models/`에 포함
- 사전 구축 인덱스를 `index/`에 저장해 기동 시간을 단축

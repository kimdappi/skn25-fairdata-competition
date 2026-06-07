# 라우팅 LLM 태그 기반 후보 축소 1차 구현 보고서

## 구현 요약

`docs/라우팅_LLM_태그_실험계획.md`의 1차 범위에 맞춰 LLM 태그 생성, 태그 JSON 저장, 문서/질문 태그 캐시 로딩, 후보 문서 기반 검색 결과 필터링, 새 실험 스크립트를 추가했다.

1차 구현은 검색 엔진 내부 인덱스를 바꾸지 않는다. Dense Chroma metadata filter, BM25 subset index, route bucket index는 포함하지 않았다.

## 추가/수정 파일

- `app/retrieval/llm_router.py`
  - LangChain LCEL 기반 LLM 태그 체인 추가.
  - `app/retrieval/keywords.py`의 키워드 사전을 프롬프트 힌트로 사용.
  - `PydanticOutputParser`와 `LLMRouteDecision`으로 JSON 출력을 파싱.
  - `chain.batch(..., max_concurrency=N)`로 문서/질문 대량 태깅 지원.

- `scripts/build_route_tags.py`
  - `data/raw` 문서와 평가셋 질문을 라우팅 태그로 변환.
  - 출력: `route_tags.json`
  - `--router-backend llm` 기본값.
  - `--router-backend rule`로 LLM 없이 fallback 경로 검증 가능.
  - LLM 태깅 실패 시 기존 `QueryRouter` 결과를 fallback으로 저장.

- `app/retrieval/route_tags.py`
  - `route_tags.json` 로딩 유틸 추가.
  - 문서는 `doc_id` 기준으로 route를 조회.
  - 질문은 정규화된 question hash 기준으로 route를 조회.
  - cache miss 시 기존 rule-based router로 fallback.

- `app/preprocessing/corpus.py`
  - `load_corpus()`에 `route_document_fn` 인자를 추가.
  - 문서별 LLM 태그를 `Document.route`와 `Chunk.route`에 주입할 수 있게 함.
  - 기존 호출 방식은 그대로 유지.

- `app/retrieval/pipeline.py`
  - `FAIRDATA_ROUTE_FILTER=1`일 때 후보 문서 필터를 적용.
  - 기본 정책:
    1. 질문 `theme`와 같은 문서를 후보로 선택.
    2. 가능하면 `industry`로 추가 축소.
    3. 후보 문서 수가 `FAIRDATA_ROUTE_MIN_CANDIDATE_DOCS`보다 적으면 fallback.
    4. 후보 청크 hit가 최종 `top_k`보다 적으면 전체 fused result를 사용.

- `server.py`
  - `FAIRDATA_ROUTE_TAGS_PATH`가 있으면 route tag JSON을 읽어 서버 라우터와 코퍼스 로딩에 사용.
  - `/health`에 `route_tags_loaded` 값을 추가.

- `scripts/build_indexes.py`
  - 인덱스 빌드 시에도 동일한 route tag JSON을 읽어 문서/청크 route를 맞춤.

- `scripts/exp_v2_e4_e5_bm25_route_tags.sh`
  - 기존 `scripts/exp_v2_e4_e5_bm25.sh` 기반 새 실험 스크립트.
  - 서버 시작 전 `scripts/build_route_tags.py`를 실행.
  - `FAIRDATA_ROUTE_FILTER=1`과 `FAIRDATA_ROUTE_TAGS_PATH`를 설정.
  - 기존 인덱스와 섞이지 않도록 `FAIRDATA_INDEX_NAMESPACE=v2_e4_e5_bm25_route_tags` 사용.

- `requirements.txt`
  - `langchain-core`
  - `langchain-huggingface`

## 새 실험 실행 파일

```bash
scripts/exp_v2_e4_e5_bm25_route_tags.sh
```

주요 환경변수:

```bash
export FAIRDATA_ROUTE_FILTER=1
export FAIRDATA_ROUTE_MIN_CANDIDATE_DOCS=20
export FAIRDATA_ROUTE_MAX_CONCURRENCY=4
export FAIRDATA_ROUTE_TAGS_PATH=cache/routes/V2-E4-E5-BM25-ROUTE/route_tags.json
export FAIRDATA_INDEX_NAMESPACE=v2_e4_e5_bm25_route_tags
export FAIRDATA_EXPERIMENT_TAG=V2-E4-E5-BM25-ROUTE
```

## 검증 결과

문법 검사를 통과했다.

```bash
python3 -m py_compile \
  app/utils/schemas.py \
  app/utils/config.py \
  app/retrieval/route_tags.py \
  app/retrieval/llm_router.py \
  app/preprocessing/corpus.py \
  app/retrieval/pipeline.py \
  server.py \
  scripts/build_indexes.py \
  scripts/build_route_tags.py
```

LLM 없이 rule backend로 태그 생성 경로를 검증했다.

```bash
python3 -u scripts/build_route_tags.py \
  --data-dir ./data/raw \
  --eval-file ./data/test/eval_dataset_260505.json \
  --output-file /tmp/fairdata_route_tags_rule_test.json \
  --router-backend rule \
  --force
```

결과:

```text
documents=500
questions=600
```

생성된 route tag JSON을 실제 코퍼스 로딩에 연결해 검증했다.

```text
documents=500
chunks=29269
route tag documents loaded=True
route tag questions loaded=True
```

후보 문서 선택도 확인했다. 예시 질문 `세라젬의 허위 과장 광고에 대한 시정조치는?`는 `theme=소비자보호`로 분류되며, 후보 문서 수는 26개로 축소된다.

## 남은 주의점

- 기본 실험 스크립트는 `--router-backend llm`을 사용하므로 LangChain 의존성과 로컬 LLM 모델이 준비되어 있어야 한다.
- 1차 구현은 검색 결과를 후보 문서 기준으로 제한하는 방식이다. Chroma/BM25 내부 계산량 자체를 줄이는 구현은 아니다.
- route filter가 recall을 떨어뜨리면 `FAIRDATA_ROUTE_MIN_CANDIDATE_DOCS`를 키우거나 `FAIRDATA_ROUTE_FILTER=0`으로 태그 품질만 먼저 비교해야 한다.


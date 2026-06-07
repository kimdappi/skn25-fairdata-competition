# 라우팅 LLM 태그 기반 후보 축소 1차 구현 계획

## 목표

`app/retrieval/keywords.py`의 키워드 사전을 LLM 프롬프트 힌트로 넣고, 질문과 문서를 같은 태그 스키마로 분류한다. 이후 질문 태그와 문서 태그가 같은 문서만 1차 후보로 선택해 검색 대상 문서 수를 줄인다.

1차 구현은 검색 엔진 내부 최적화가 아니라, **LLM 태그 생성 + JSON 저장 + 후보 문서 선택 + fallback**까지만 한다.

## 1차 구현 범위

포함한다.

- 문서별 LLM 라우팅 태그 생성
- 질문별 LLM 라우팅 태그 생성
- 태그 결과를 JSON 파일로 저장
- 서버/검색 파이프라인에서 JSON 태그를 읽어 사용
- 질문 태그와 같은 문서 태그를 가진 문서만 후보로 선택
- 후보가 너무 적으면 전체 문서 검색으로 fallback
- 기존 rule-based router는 LLM 태깅 실패 시 fallback으로 유지

포함하지 않는다.

- Chroma DB metadata 필드 추가
- Chroma `where` filter 적용
- BM25 subset index 생성
- route bucket별 별도 검색 인덱스 생성
- 검색 엔진 내부 score 계산량 최적화

위 항목들은 1차 실험에서 정확도와 후보 축소 가능성을 확인한 뒤 2차 최적화로 진행한다.

## 태그 스키마

기존 `RouteDecision`과 같은 축을 사용한다.

```text
theme
company_size
legal_role
industry
focus
```

LLM 출력에는 디버깅용 필드를 추가할 수 있다.

```python
class LLMRouteDecision(BaseModel):
    theme: str
    company_size: str
    legal_role: str
    industry: str
    focus: str
    confidence: float = 0.0
    reason: str = ""
```

검색 파이프라인에 넘길 때는 기존 `RouteDecision`으로 변환한다.

## LLM 프롬프트 방향

프롬프트에는 `app/retrieval/keywords.py`의 카테고리와 키워드를 힌트로 넣는다.

예시:

```text
아래 텍스트를 공정위 의결서 검색용 라우팅 태그로 분류하라.
반드시 제공된 label 중 하나만 선택하라.

[theme]
- 공정거래(독과점/담합): 담합, 공동행위, 시장지배적지위, 남용, 부당지원, 기업결합, 독점
- 갑을관계(불공정거래): 하도급, 가맹, 대리점, 유통업, 기술유용, 부당특약, 대규모유통
- 소비자보호: 표시광고, 전자상거래, 약관, 방문판매, 할부거래, 허위, 과장, 기만
- 기타

[입력 텍스트]
...

[출력]
JSON only
```

문서와 질문은 같은 label 체계를 사용한다. 다만 입력 설명만 다르게 한다.

- 문서 입력: 제목, 피심인기업명, 위반유형, 세부위반유형, 문서 초반 본문 일부
- 질문 입력: 질문 텍스트

## LCEL 구성

`app/retrieval/llm_router.py`를 추가한다.

구성:

```python
prompt | llm | parser
```

사용 요소:

- LangChain LCEL
- `PydanticOutputParser`
- 로컬 HuggingFace LLM wrapper
- `chain.batch(..., config={"max_concurrency": N})`

질문과 문서를 대량 태깅할 때는 batch를 사용해 병렬 처리한다.

## 태그 생성 스크립트

`scripts/build_route_tags.py`를 추가한다.

입력:

```bash
python3 -u scripts/build_route_tags.py \
  --data-dir ./data/raw \
  --eval-file ./data/test/eval_dataset_260505.json \
  --output-file cache/routes/V2-E4-E5-BM25/route_tags.json
```

출력 파일 구조:

```json
{
  "documents": {
    "doc_id": {
      "theme": "소비자보호",
      "company_size": "기타",
      "legal_role": "거래 상대방(을)",
      "industry": "디지털/플랫폼/IT",
      "focus": "처분",
      "confidence": 0.84,
      "reason": "..."
    }
  },
  "questions": {
    "eval_id": {
      "question": "...",
      "route": {
        "theme": "소비자보호",
        "company_size": "기타",
        "legal_role": "거래 상대방(을)",
        "industry": "디지털/플랫폼/IT",
        "focus": "처분",
        "confidence": 0.81,
        "reason": "..."
      }
    }
  }
}
```

태깅 실패 시:

- 기존 `QueryRouter().route_from_text(...)`로 fallback
- fallback 여부를 JSON에 기록
- 스크립트는 실패 문서 몇 개 때문에 전체 실험을 중단하지 않음

## 코퍼스 로딩 수정

`app/preprocessing/corpus.py`의 `load_corpus()`가 문서별 LLM 태그를 사용할 수 있게 한다.

권장 변경:

```python
def load_corpus(
    data_dir: Path,
    route_text_fn: Callable[[str], RouteDecision],
    route_document_fn: Callable[[str, str], RouteDecision] | None = None,
) -> CorpusStore:
    ...
    if route_document_fn is not None:
        route = route_document_fn(doc_id, full_text)
    else:
        route = route_text_fn(full_text)
```

이렇게 하면 기존 코드 호환성을 유지하면서, 실험에서는 `doc_id` 기반 JSON 태그를 안정적으로 붙일 수 있다.

## 질문 라우팅 수정

`app/retrieval/router.py` 또는 새 모듈에 JSON 태그를 읽는 wrapper를 추가한다.

```python
class CachedQuestionRouter:
    def __init__(self, fallback: QueryRouter, route_tags_path: Path):
        self.fallback = fallback
        self.question_routes = load_question_routes(route_tags_path)

    def route_from_text(self, text: str) -> RouteDecision:
        key = normalize_question_key(text)
        if key in self.question_routes:
            return self.question_routes[key]
        return self.fallback.route_from_text(text)
```

로컬 평가셋은 `id`가 있으나, 현재 `retriever.search(question)`에는 `id`가 전달되지 않는다. 1차 구현에서는 질문 텍스트 정규화 key 또는 hash를 사용한다.

## 후보 문서 선택

`app/retrieval/pipeline.py`에 후보 문서 선택 로직을 추가한다.

1차에서는 검색 엔진 내부를 바꾸지 않고, 검색 결과를 후보 문서 기준으로 제한하는 방식부터 시작한다.

정책:

```text
1. query.route.theme가 기타가 아니면 같은 theme 문서를 후보로 선택
2. 후보 문서가 충분히 많고 query.route.industry가 기타가 아니면 theme + industry로 좁힘
3. 좁힌 후보가 너무 적으면 theme only로 되돌림
4. theme 후보도 너무 적으면 전체 문서 사용
```

권장 env:

```bash
FAIRDATA_ROUTE_FILTER=1
FAIRDATA_ROUTE_MIN_CANDIDATE_DOCS=20
```

후보 문서에서 청크 id set을 만든다.

```python
candidate_doc_ids = select_candidate_doc_ids(analysis.route, corpus_store.documents)
candidate_chunk_ids = {
    chunk_id
    for doc_id in candidate_doc_ids
    for chunk_id in corpus_store.document_to_chunk_ids[doc_id]
}
```

검색 엔진이 반환한 hit 중 `candidate_chunk_ids`에 포함된 것만 우선 사용한다. 필터 후 결과가 부족하면 전체 검색 결과를 섞어 top-k를 채운다.

## 실험 스크립트 수정

`scripts/exp_v2_e4_e5_bm25.sh`에 태그 생성 단계를 추가하여 새 스크립트 파일을 만든다.

```bash
export FAIRDATA_ROUTE_FILTER=1
export FAIRDATA_ROUTE_TAGS_PATH=cache/routes/V2-E4-E5-BM25/route_tags.json
export FAIRDATA_ROUTE_MIN_CANDIDATE_DOCS=20
export FAIRDATA_ROUTE_MAX_CONCURRENCY=4

mkdir -p "$(dirname "$FAIRDATA_ROUTE_TAGS_PATH")"

python3 -u scripts/build_route_tags.py \
  --data-dir ./data/raw \
  --eval-file ./data/test/eval_dataset_260505.json \
  --output-file "$FAIRDATA_ROUTE_TAGS_PATH" \
  --max-concurrency "$FAIRDATA_ROUTE_MAX_CONCURRENCY" \
  | tee "$RESULTS_DIR/route_tags.log"

python3 -u scripts/build_indexes.py | tee "$RESULTS_DIR/build.log"
```

중요: `build_route_tags.py`는 `build_indexes.py`와 서버 시작보다 먼저 실행한다. 그래야 문서 로딩 시 LLM 태그를 사용할 수 있다.

## 1차 검증 기준

비교 대상은 기존 `V2-E4-E5-BM25` 결과다.

확인할 지표:

- `recall@5`
- `mrr@5`
- 평균 후보 문서 수
- 평균 후보 청크 수
- fallback 발생 비율
- LLM 태깅 실패 비율

1차 성공 기준:

- `recall@5`가 크게 떨어지지 않는다.
- 후보 문서/청크 수가 유의미하게 줄어든다.
- fallback이 과도하게 많이 발생하지 않는다.




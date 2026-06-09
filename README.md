# 공정위 FairData RAG 제출 코드

공정거래위원회 FairData 공모전 Track 2 제출용 RAG 서버입니다.  
현재 코드는 FastAPI 기반 `/health`, `/predict` API를 제공하고, 검색은 backend 분리형 retrieval + reranker, 생성은 env 기반 로컬 LLM 선택 구조로 구성돼 있습니다.

질문 라우터는 아래 3종을 공통 팩토리로 선택할 수 있습니다.

- `keyword`
  - 기존 키워드 규칙 기반 라우팅
- `lcel`
  - LangChain LCEL 기반 라우팅
- `lcel_prompt_boost`
  - LangChain LCEL 기반 라우팅 + 강화 프롬프트

서버 질문 라우터는 `FAIRDATA_QUESTION_ROUTER_BACKEND`로, 태그 생성 스크립트는 `scripts/build_route_tags.py --router-backend ...`로 맞춰 실험합니다.

## 현재 코드 구조

```text
.
├── app/
│   ├── evaluation/      # 로컬 평가 지표 계산
│   ├── generation/      # Qwen 기반 답변 생성
│   ├── preprocessing/   # raw/hybrid 코퍼스 로드
│   ├── rerank/          # BGE reranker
│   ├── retrieval/       # dense / sparse / multi-vector 검색
│   └── utils/           # 경로/스키마/텍스트 유틸
├── data/
│   ├── raw/             # 실제 검색 코퍼스
│   └── test/            # 로컬 평가셋
├── docs/
├── index/               # 검색 인덱스 저장 위치
├── models/              # 로컬 모델 저장 위치
├── scripts/
│   ├── build_indexes.py
│   ├── download_bgem3_model.sh
│   ├── download_model_matrix.sh
│   └── evaluate_local.py
├── Dockerfile
├── requirements.txt
└── server.py
```

## 서버 동작 방식

서버 진입점은 [server.py](/home/ming9/skn25-fairdata-competition/server.py:1) 입니다.

- `GET /health`
  - 코퍼스 청크 수, data 경로, 현재 retrieval profile, dense/sparse/multivector backend 정보를 반환합니다.
- `POST /predict`
  - 입력: `id`, `question`
  - 출력: `id`, `retrieved_chunk_ids`, `answer`

라우터 선택은 `FAIRDATA_QUESTION_ROUTER_BACKEND`로 제어합니다. `/health`의 `question_routing` 필드에서 현재 적용된 값을 확인할 수 있습니다.

루트 기준 실행 흐름은 아래와 같습니다.

1. `server.py` import 시점에 `load_corpus()`가 `data/raw/*_metadata.json`, `*_hybrid.json`을 읽어 문서/청크 저장소를 만듭니다.
2. `HybridRetriever`가 질문을 `QueryAnalysis`로 분석합니다.
3. 활성화된 검색 경로 `dense`, `sparse`, `multi-vector`를 각각 전체 코퍼스에 대해 실행합니다.
4. 선택된 검색 경로 결과만 `RRF` 또는 score fusion으로 합칩니다.
5. 라우팅 일치도(`theme`, `focus`, `legal_role`, `industry`, `company_size`)를 retrieval bonus로 더합니다.
6. 현재 선택된 reranker backend가 상위 후보를 재정렬합니다.
7. 최종 5개 청크를 현재 선택된 로컬 LLM에 넘겨 답변을 생성합니다.

검색 파이프라인 핵심 파일은 아래입니다.

- [app/retrieval/pipeline.py](/home/ming9/skn25-fairdata-competition/app/retrieval/pipeline.py:1)
- [app/retrieval/engines.py](/home/ming9/skn25-fairdata-competition/app/retrieval/engines.py:1)
- [app/retrieval/backends.py](/home/ming9/skn25-fairdata-competition/app/retrieval/backends.py:1)
- [app/retrieval/interfaces.py](/home/ming9/skn25-fairdata-competition/app/retrieval/interfaces.py:1)
- [app/rerank/reranker.py](/home/ming9/skn25-fairdata-competition/app/rerank/reranker.py:1)
- [app/rerank/backends.py](/home/ming9/skn25-fairdata-competition/app/rerank/backends.py:1)
- [app/rerank/interfaces.py](/home/ming9/skn25-fairdata-competition/app/rerank/interfaces.py:1)

## Retrieval 구조

현재 코드는 retrieval을 아래 세 축으로 구분합니다.

- `dense`
  - 일반 임베딩 벡터 기반 검색
  - 예: `bgem3`, `e5`, `jina_v4`, `gte_multilingual`, `sbert`, `kure_v1`, `snowflake_ko`
- `sparse`
  - 두 종류가 있습니다.
  - `lexical_sparse`
    - 예: `bm25`
    - 학습된 모델 없이 토큰 통계로 동작합니다.
  - `learned_sparse`
    - 예: `bgem3`, `upskyy_bgem3_ko`
    - 전용 sparse 출력이 필요한 learned sparse입니다.
- `multi-vector`
  - 문서당 여러 벡터를 사용하는 late interaction 검색
  - 현재는 `bgem3`, `upskyy_bgem3_ko` 계열만 허용합니다.

### Retrieval profile

현재 실행 중인 retrieval 조합은 `retrieval_profile`로 구분됩니다.

- `dense_only`
- `dense_lexical_hybrid`
- `dense_learned_sparse_hybrid`
- `full_hybrid`
- `custom`

`GET /health` 또는 `scripts/validate_model_matrix.py` 출력에서 현재 profile을 확인할 수 있습니다.

### 중요한 제약

- 일반 dense 모델(`e5`, `jina_v4`, `gte_multilingual`, `sbert`, `kure_v1`, `snowflake_ko`)은 `dense-only` 또는 `dense + bm25` 조합으로만 운영하는 것이 맞습니다.
- learned sparse와 multi-vector는 현재 구현상 `bgem3` family만 안정 지원합니다.
- `e5 + bgem3 sparse`, `e5 + bgem3 multivector` 같은 조합은 validation 단계에서 차단됩니다.

## 검색 인덱스

인덱스는 `models/`가 아니라 `index/` 아래에 저장됩니다.

- dense: `index/chroma_<namespace>/...`
- sparse: `index/sparse_<backend>_chunks_<model_tag>.npz`
- multi-vector: `index/multivector_<backend>_chunks_<model_tag>.npz`

각 인덱스는 manifest와 corpus fingerprint를 비교해서:

- 기존 인덱스가 유효하면 재사용
- corpus가 바뀌었으면 자동 재생성

## 데이터 한 건 시나리오

예를 들어 아래 요청이 들어온다고 가정합니다.

```json
{
  "id": "demo-1",
  "question": "한국계란유통협회의 위반 사실과 처분 내용을 설명해 주세요."
}
```

코드상 흐름은 아래와 같습니다.

1. [server.py](/home/ming9/skn25-fairdata-competition/server.py:42) 의 `predict()`가 호출됩니다.
2. [app/retrieval/pipeline.py](/home/ming9/skn25-fairdata-competition/app/retrieval/pipeline.py:175) 의 `analyze_query()`가 질문을 토큰화하고 라우팅합니다.
3. [app/retrieval/router.py](/home/ming9/skn25-fairdata-competition/app/retrieval/router.py:63) 가 `theme`, `focus`, `legal_role`, `industry`, `company_size`를 규칙 기반으로 분류합니다.
4. [app/retrieval/pipeline.py](/home/ming9/skn25-fairdata-competition/app/retrieval/pipeline.py:185) 가 활성화된 retrieval 경로를 실행합니다.
5. [app/retrieval/pipeline.py](/home/ming9/skn25-fairdata-competition/app/retrieval/pipeline.py:207) 가 선택된 경로 결과를 `RRF` 또는 score fusion으로 합치고 route bonus를 더합니다.
6. [app/retrieval/pipeline.py](/home/ming9/skn25-fairdata-competition/app/retrieval/pipeline.py:175) 가 현재 선택된 reranker backend를 만들고, [app/rerank/reranker.py](/home/ming9/skn25-fairdata-competition/app/rerank/reranker.py:87) 가 최종 청크 순서를 다시 정합니다.
7. [app/generation/generator.py](/home/ming9/skn25-fairdata-competition/app/generation/generator.py:92) 가 상위 청크를 근거로 답변을 생성합니다.

즉 질문 1건은 `질문 분석 -> retrieval paths -> fusion -> rerank -> generation` 순서로 처리됩니다.

## 모델 교체와 실험

현재 구조는 `app/retrieval`, `app/rerank`, `app/generation` 안에서 모델 실험을 하기 쉽게 분리된 상태입니다. 실험 설정은 [app/utils/config.py](/home/ming9/skn25-fairdata-competition/app/utils/config.py:1) 의 env 해석 함수로 제어합니다.

- `app/utils/config.py`
  - 어떤 retrieval / reranker / LLM 모델을 쓸지, 어떤 retrieval 조합이 허용되는지 해석
- `app/retrieval/interfaces.py`
  - dense / sparse / multivector backend 인터페이스 정의
- `app/retrieval/backends.py`
  - 실제 backend 구현 등록
- `app/rerank/interfaces.py`
  - reranker backend 인터페이스 정의
- `app/rerank/backends.py`
  - 실제 reranker 구현 등록

### 1. env로 바로 바꿀 수 있는 경우

현재는 아래가 `env`만으로 바로 교체 가능합니다.

- dense-only
  - `bgem3`
  - `e5`
  - `jina_v4`
  - `gte_multilingual`
  - `sbert`
  - `kure_v1`
  - `snowflake_ko`
- dense + lexical sparse
  - `e5 + bm25`
  - `jina_v4 + bm25`
  - `gte_multilingual + bm25`
  - `sbert + bm25`
- full-hybrid
  - `bgem3 + learned_sparse + multivector`

### 2. env만으로는 안 되는 경우

아래 조합은 현재 코드 기준으로 바로 갈아끼울 수 없습니다.

- `OpenAI embedding + dense only`
  - OpenAI API backend 미구현
- `bge-small`, `bge-base + dense only`
  - alias / model_dir / runtime 미연결
- `e5 + bgem3 sparse`
  - learned sparse family mismatch
- `e5 + bgem3 multivector`
  - multi-vector family mismatch
- `SPLADE`
  - sparse backend 미구현
- `ColBERT`
  - multi-vector backend 미구현

### 3. 같은 계열 체크포인트 경로만 바꾸는 경우

가장 쉬운 경우입니다. [app/utils/config.py](/home/ming9/skn25-fairdata-competition/app/utils/config.py:15) 의 반환 경로만 수정하면 됩니다.

- `resolve_dense_model_dir()`
- `resolve_sparse_model_dir()`
- `resolve_multivector_model_dir()`
- `resolve_bge_reranker_model_dir()`
- `resolve_qwen_model_dir()`

#### root 기준으로 보면 실제로 어떻게 읽히는지

말 그대로 `config.py`는 "모델을 직접 로드하는 파일"이라기보다 "지금 repo root를 기준으로 어느 폴더를 모델 경로로 넘길지 정하는 파일"입니다.

구어체로 풀면 이런 느낌입니다.

1. 코드가 먼저 프로젝트 root를 잡습니다.
   - 지금 기준으로는 `BASE_DIR = /home/ming9/skn25-fairdata-competition` 입니다.
2. 그다음 "dense는 무슨 backend 쓸 건데?"를 env에서 읽습니다.
   - 예를 들어 `FAIRDATA_DENSE_BACKEND=e5` 라고 적혀 있으면 내부적으로 `e5`로 정규화합니다.
3. 그 backend에 맞는 기본 모델 폴더를 root 기준으로 붙입니다.
   - 예를 들면 `e5`면 `BASE_DIR / models / embedding / multilingual-e5-large`
   - 그래서 실제 경로는 `/home/ming9/skn25-fairdata-competition/models/embedding/multilingual-e5-large` 가 됩니다.
4. 마지막으로 retrieval backend/runtime 코드가 그 경로를 받아서 진짜 모델을 로드합니다.

즉 코드 입장에서는 이런 식입니다.

- "dense backend는 e5네"
- "그럼 기본 모델 폴더는 root 아래 `models/embedding/multilingual-e5-large` 쓰면 되겠다"
- "이 경로를 SentenceTransformer 쪽에 넘겨서 읽자"

반대로 env로 경로를 직접 주면 기본 경로 테이블보다 그 값이 우선입니다.

- 예: `FAIRDATA_DENSE_MODEL_DIR=/mnt/exp/e5-large`
- 그러면 backend는 여전히 `e5` 방식으로 동작하지만, 실제 모델은 `/mnt/exp/e5-large` 에서 읽습니다.

한 줄로 정리하면 이렇습니다.

- backend 이름은 "어떤 방식으로 로드하고 쓸지"를 정합니다.
- model_dir 경로는 "어디서 실제 파일을 읽을지"를 정합니다.

### 4. 다른 retrieval 모델을 붙이는 경우

예를 들어 dense만 다른 임베딩 모델로 바꾸려면 아래 순서로 합니다.

1. [app/retrieval/interfaces.py](/home/ming9/skn25-fairdata-competition/app/retrieval/interfaces.py:8) 의 `DenseRetrievalBackend` 인터페이스를 구현하는 새 클래스를 추가
2. [app/retrieval/backends.py](/home/ming9/skn25-fairdata-competition/app/retrieval/backends.py:77) 의 `build_dense_backend()`에 새 backend 이름 등록
3. [app/utils/config.py](/home/ming9/skn25-fairdata-competition/app/utils/config.py:8) 기준으로 `resolve_dense_backend_name()` 이 읽는 env, 즉 `FAIRDATA_DENSE_BACKEND` 값을 새 backend 이름으로 변경
4. `resolve_dense_model_dir()`가 새 모델 디렉터리를 가리키도록 수정

sparse, multivector도 같은 방식이지만 제약이 있습니다.

- sparse
  - `bm25`는 lexical sparse이므로 별도 모델 없이 붙일 수 있습니다.
  - learned sparse는 현재 `bgem3` family만 있습니다.
- multivector
  - 현재 `bgem3` family만 있습니다.

### 5. 다른 reranker를 붙이는 경우

리랭커도 retrieval과 같은 패턴으로 분리되어 있습니다.

1. [app/rerank/interfaces.py](/home/ming9/skn25-fairdata-competition/app/rerank/interfaces.py:9) 의 `RerankerBackend` 인터페이스를 구현하는 새 클래스를 추가
2. [app/rerank/backends.py](/home/ming9/skn25-fairdata-competition/app/rerank/backends.py:10) 의 `build_reranker_backend()`에 새 backend 이름 등록
3. [app/utils/config.py](/home/ming9/skn25-fairdata-competition/app/utils/config.py:11) 기준으로 `resolve_reranker_backend_name()` 이 읽는 env, 즉 `FAIRDATA_RERANK_BACKEND` 값을 새 backend 이름으로 변경
4. `resolve_bge_reranker_model_dir()`가 새 모델 디렉터리를 가리키도록 수정

즉 현재 기본 구현은 `bge_reranker`지만, 다른 cross-encoder 계열 리랭커도 같은 패턴으로 꽂을 수 있습니다.

### 6. RRF를 넣었다 뺐다 하는 경우

[app/utils/config.py](/home/ming9/skn25-fairdata-competition/app/utils/config.py:12) 기준으로 `use_rrf_fusion()` 이 읽는 env, 즉 `FAIRDATA_USE_RRF_FUSION`을 수정합니다.

- `True`: [app/retrieval/pipeline.py](/home/ming9/skn25-fairdata-competition/app/retrieval/pipeline.py:220) 의 RRF 사용
- `False`: [app/retrieval/pipeline.py](/home/ming9/skn25-fairdata-competition/app/retrieval/pipeline.py:227) 의 score fusion 사용

### 7. 경로를 부분적으로만 실험하는 경우

[app/utils/config.py](/home/ming9/skn25-fairdata-competition/app/utils/config.py:69) 이후의 함수로 제어합니다.

- `is_dense_enabled()`
- `is_sparse_enabled()`
- `is_multivector_enabled()`

필요하면 특정 경로만 `True`로 두고 단독 평가할 수 있습니다.

### 8. 지금 직접 바꾸는 포인트 정리

실험할 때 실제로 가장 자주 수정하는 값은 아래입니다.

- retrieval backend 종류
  - `FAIRDATA_DENSE_BACKEND`
  - `FAIRDATA_SPARSE_BACKEND`
  - `FAIRDATA_MULTIVECTOR_BACKEND`
- reranker backend 종류
  - `FAIRDATA_RERANK_BACKEND`
- fusion 방식
  - `FAIRDATA_USE_RRF_FUSION`
- 모델 경로
  - `resolve_dense_model_dir()`
  - `resolve_sparse_model_dir()`
  - `resolve_multivector_model_dir()`
  - `resolve_bge_reranker_model_dir()`
  - `resolve_llm_model_dir()`
- 경로 on/off
  - `FAIRDATA_ENABLE_DENSE`
  - `FAIRDATA_ENABLE_SPARSE`
  - `FAIRDATA_ENABLE_MULTIVECTOR`

함수 기준으로 보면 대응 관계는 이렇게 읽으면 됩니다.

- `resolve_dense_backend_name()` -> `FAIRDATA_DENSE_BACKEND`
- `resolve_sparse_backend_name()` -> `FAIRDATA_SPARSE_BACKEND`
- `resolve_multivector_backend_name()` -> `FAIRDATA_MULTIVECTOR_BACKEND`
- `resolve_reranker_backend_name()` -> `FAIRDATA_RERANK_BACKEND`
- `use_rrf_fusion()` -> `FAIRDATA_USE_RRF_FUSION`
- `is_dense_enabled()` -> `FAIRDATA_ENABLE_DENSE`
- `is_sparse_enabled()` -> `FAIRDATA_ENABLE_SPARSE`
- `is_multivector_enabled()` -> `FAIRDATA_ENABLE_MULTIVECTOR`
- `resolve_dense_model_dir()` -> `FAIRDATA_DENSE_MODEL_DIR`
- `resolve_sparse_model_dir()` -> `FAIRDATA_SPARSE_MODEL_DIR`
- `resolve_multivector_model_dir()` -> `FAIRDATA_MULTIVECTOR_MODEL_DIR`
- `resolve_bge_reranker_model_dir()` -> `FAIRDATA_RERANK_MODEL_DIR`
- `resolve_llm_model_dir()` -> `FAIRDATA_LLM_MODEL_DIR`

## 모델 경로

코드가 기본으로 기대하는 대표 모델 경로는 아래와 같습니다.

- BGE-M3: `models/embedding/bge-m3`
- BM25 runtime placeholder: `models/bm25`
- BGE reranker: `models/reranker/bge-reranker-v2-m3`
- Qwen 생성 모델: `models/llm/qwen2.5-7b-instruct`

루트 직하 alias(예: `models/qwen2.5-7b-instruct`)나 symlink는 운영 기준으로 유지하지 않습니다.
모델 실체는 반드시 `models/{embedding,reranker,llm}` 아래 canonical 경로에만 둡니다.

경로를 바꾸려면 [app/utils/config.py](/home/ming9/skn25-fairdata-competition/app/utils/config.py:1) 의 반환값을 직접 수정하면 됩니다.

## 실험 진행 가이드

아래 절차는 "모델 파일은 `models/` 아래에 이미 있고, 필요한 인덱스도 `index/` 아래에 이미 있다"는 가정으로 적었습니다. 기준 위치는 모두 repo root, 즉 `/home/ming9/skn25-fairdata-competition` 입니다.

이 섹션은 현재 코드의 [app/utils/config.py](/home/ming9/skn25-fairdata-competition/app/utils/config.py:1), [server.py](/home/ming9/skn25-fairdata-competition/server.py:1), [scripts/evaluate_local.py](/home/ming9/skn25-fairdata-competition/scripts/evaluate_local.py:1), 그리고 제출 규격 설명이 있는 `docs/공정위AI공모전 모델제출 가이드.md`를 같이 참고해서 정리한 사용 절차입니다.

### 1. 먼저 어떤 조합으로 돌릴지 정합니다

실험은 결국 `config.py`가 읽는 env 조합으로 결정됩니다. root에서 실행할 때 가장 자주 만지는 값은 아래입니다.

- retrieval 경로 on/off
  - `FAIRDATA_ENABLE_DENSE`
  - `FAIRDATA_ENABLE_SPARSE`
  - `FAIRDATA_ENABLE_MULTIVECTOR`
- retrieval backend 선택
  - `FAIRDATA_DENSE_BACKEND`
  - `FAIRDATA_SPARSE_BACKEND`
  - `FAIRDATA_MULTIVECTOR_BACKEND`
- reranker / generation
  - `FAIRDATA_RERANK_BACKEND`
  - `FAIRDATA_RERANK_TOP_N`
  - `FAIRDATA_RERANK_WEIGHT`
  - `FAIRDATA_LLM_BACKEND`
- 실험 분리용 이름
  - `FAIRDATA_INDEX_NAMESPACE`
  - `FAIRDATA_EXPERIMENT_TAG`

실제로는 아래 두 패턴이 가장 무난합니다.

- full hybrid
  - dense=`bgem3`, sparse=`bgem3`, multivector=`bgem3`
- dense + bm25
  - dense=`e5` 같은 dense-only 모델, sparse=`bm25`, multivector off

`FAIRDATA_LLM_BACKEND`는 이제 family 이름이 아니라 사실상 `models/llm` 아래 폴더명입니다.

- 예: `qwen2.5-7b-instruct`
- 예: `qwen3-4b`
- 예: `exaone-3.5-7.8b-instruct`
- 예: `llama-3-open-ko-8b`
- 예: `llama-varco-8b-instruct`
- 예: `phi-3.5-mini-instruct`

짧은 alias도 일부 유지합니다.

- `qwen` -> `qwen2.5-7b-instruct`
- `qwen3` -> `qwen3-8b`
- `exaone` -> `exaone-3.5-7.8b-instruct`
- `llama3` -> `llama-3-open-ko-8b`
- `phi` -> `phi-3.5-mini-instruct`

예를 들면 full hybrid는 이렇게 잡습니다.

```bash
export FAIRDATA_ENABLE_DENSE=1
export FAIRDATA_ENABLE_SPARSE=1
export FAIRDATA_ENABLE_MULTIVECTOR=1
export FAIRDATA_DENSE_BACKEND=bgem3
export FAIRDATA_SPARSE_BACKEND=bgem3
export FAIRDATA_MULTIVECTOR_BACKEND=bgem3
export FAIRDATA_RERANK_BACKEND=bge_reranker
export FAIRDATA_LLM_BACKEND=qwen2.5-7b-instruct
export FAIRDATA_INDEX_NAMESPACE=exp_full_hybrid_bgem3
export FAIRDATA_EXPERIMENT_TAG=full_hybrid_bgem3_qwen
```

dense + bm25는 이렇게 잡으면 됩니다.

```bash
export FAIRDATA_ENABLE_DENSE=1
export FAIRDATA_ENABLE_SPARSE=1
export FAIRDATA_ENABLE_MULTIVECTOR=0
export FAIRDATA_DENSE_BACKEND=e5
export FAIRDATA_SPARSE_BACKEND=bm25
export FAIRDATA_RERANK_BACKEND=bge_reranker
export FAIRDATA_LLM_BACKEND=qwen2.5-7b-instruct
export FAIRDATA_INDEX_NAMESPACE=exp_e5_bm25
export FAIRDATA_EXPERIMENT_TAG=e5_bm25_qwen
```

주의할 점은, `config.py`가 backend family 제약을 강하게 검사한다는 점입니다.

- `e5 + bm25` 는 허용
- `e5 + bgem3 sparse` 는 차단
- multivector를 켜면 현재 구현상 dense backend와 같은 family여야 함

### 2. 설정이 맞는지 먼저 검증합니다

제일 먼저 현재 env 조합이 유효한지 확인합니다.

```bash
python3 scripts/validate_model_matrix.py
```

여기서 확인할 포인트는 아래입니다.

- `retrieval_profile` 이 내가 의도한 조합으로 나오는지
- dense/sparse/multivector backend 이름이 기대한 값인지
- validation error 없이 끝나는지

### 3. 인덱스를 그대로 쓸지, 다시 만들지 결정합니다

이번 요청은 "index 파일이 이미 있다"는 가정이지만, 아래 둘 중 하나라도 바뀌었으면 인덱스를 다시 만드는 쪽이 안전합니다.

- 코퍼스 내용이 바뀐 경우
- backend 또는 `FAIRDATA_INDEX_NAMESPACE` 를 바꾼 경우

다시 만들 필요가 있으면 root에서 이렇게 실행합니다.

```bash
python3 scripts/build_indexes.py
```

이미 같은 namespace와 같은 코퍼스 기준 인덱스가 준비돼 있으면 이 단계는 건너뛰어도 됩니다. 다만 헷갈리면 한 번 실행해서 manifest/fingerprint 기준으로 재사용 또는 재생성을 맡기는 편이 안전합니다.

### 4. 서버를 띄우고 제출 규격대로 상태를 확인합니다

제출 가이드 기준으로 심사 서버는 먼저 `GET /health`, 그 다음 `POST /predict`를 호출합니다. 그래서 실험할 때도 이 순서로 보는 게 맞습니다.

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

다른 터미널에서 health 확인:

```bash
curl http://127.0.0.1:8000/health
```

여기서 최소한 아래를 확인합니다.

- `status` 가 `ok` 인지
- `retrieval_profile` 이 의도한 실험 조합인지
- `dense_backend`, `sparse_backend`, `multivector_backend` 가 env와 맞는지

그다음 predict를 한 번 직접 때려봅니다.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"id":"demo-1","question":"한국계란유통협회의 위반 사실과 처분 내용을 설명해 주세요."}'
```

`docs/공정위AI공모전 모델제출 가이드.md` 기준으로 여기서 꼭 맞아야 하는 규칙은 아래입니다.

- `retrieved_chunk_ids` 는 정확히 5개여야 함
- 중복 chunk_id가 있으면 안 됨
- 코퍼스에 존재하는 chunk_id만 반환해야 함
- 배열 순서가 랭킹 의미를 가져야 함
- 응답 시간은 문항당 30초를 넘기지 않는 쪽으로 맞춰야 함

### 5. 로컬 평가를 돌립니다

서버가 올라온 상태에서 `evaluate_local.py` 를 실행하면, 제출 환경과 비슷하게 `/predict`를 HTTP로 호출하면서 평가를 진행합니다.

전체 평가셋 기준 예시는 아래입니다.

```bash
python3 scripts/evaluate_local.py \
  --base-url http://127.0.0.1:8000 \
  --eval-file ./data/test/eval_dataset_260505.json
```

빠르게 일부만 볼 때는 이렇게 합니다.

```bash
python3 scripts/evaluate_local.py \
  --base-url http://127.0.0.1:8000 \
  --eval-file ./data/test/eval_dataset_260505.json \
  --limit 20
```

특정 구간만 재평가할 때는 이렇게 씁니다.

```bash
python3 scripts/evaluate_local.py \
  --base-url http://127.0.0.1:8000 \
  --eval-file ./data/test/eval_dataset_260505.json \
  --offset 100 \
  --limit 20
```

결과는 기본적으로 `results/` 아래에 저장됩니다.

- `summary.json`
  - 전체 지표 요약
- `predictions.jsonl`
  - 문항별 retrieved_chunk_ids, answer, recall_at_5, mrr, token_f1

`summary.json`의 `config_snapshot`에는 현재 env 해석 결과가 같이 남습니다. 그래서 나중에 "이 점수가 어떤 backend 조합에서 나온 거였지?"를 다시 추적할 수 있습니다.

### 6. 실험 비교는 `experiment_tag`와 `index_namespace`를 같이 봅니다

실험을 여러 번 돌릴 때는 아래처럼 생각하면 편합니다.

- `FAIRDATA_INDEX_NAMESPACE`
  - 어떤 인덱스를 읽을지 구분하는 이름
- `FAIRDATA_EXPERIMENT_TAG`
  - 결과 파일에서 이 실행을 식별하는 이름

보통은 둘 다 같이 바꿔 주는 게 덜 헷갈립니다.

- backend 조합이 달라졌다
  - `INDEX_NAMESPACE`도 바꾸기
  - 필요하면 `build_indexes.py` 다시 실행
- 같은 backend 조합인데 reranker weight나 LLM만 바꿨다
  - `EXPERIMENT_TAG`만 바꿔도 비교 가능

### 7. 가장 현실적인 실험 루틴

root 기준으로 실제로는 아래 순서로 반복하면 됩니다.

1. env export로 실험 조합 설정
2. `python3 scripts/validate_model_matrix.py`
3. 필요할 때만 `python3 scripts/build_indexes.py`
4. `uvicorn server:app --host 0.0.0.0 --port 8000`
5. `curl /health`
6. `curl /predict` 샘플 1건
7. `python3 scripts/evaluate_local.py --eval-file ./data/test/eval_dataset_260505.json`
8. `results/.../summary.json` 비교

## 준비 절차

### 1. 모델 다운로드

기존 기본 스크립트:

```bash
bash scripts/download_bgem3_model.sh
```

기본값으로 아래 3개를 모두 다운로드합니다.

- `BAAI/bge-m3`
- `BAAI/bge-reranker-v2-m3`
- `Qwen/Qwen2-7B-Instruct`

선택적으로 끌 수 있습니다.

```bash
DOWNLOAD_BGEM3=1 DOWNLOAD_RERANKER=1 DOWNLOAD_QWEN=0 bash scripts/download_bgem3_model.sh
```

패치노트 기준 모델 매트릭스를 기능군별로 받으려면 아래 스크립트를 사용합니다.

```bash
bash scripts/download_model_matrix.sh
```

주의:

- BGE-M3 backend는 추가로 `sparse_linear.pt`, `colbert_linear.pt`가 필요합니다.
- 이 두 파일이 없으면 현재 기본 retrieval backend(`bgem3`) 초기화가 실패합니다.

### 2. 조합 검증

서버 실행 전 현재 retrieval 조합이 유효한지 먼저 검증하는 것을 권장합니다.

```bash
python3 scripts/validate_model_matrix.py
```

예시:

```bash
FAIRDATA_DENSE_BACKEND=e5 \
FAIRDATA_ENABLE_DENSE=1 \
FAIRDATA_ENABLE_SPARSE=1 \
FAIRDATA_SPARSE_BACKEND=bm25 \
FAIRDATA_ENABLE_MULTIVECTOR=0 \
python3 scripts/validate_model_matrix.py
```

이 경우 `retrieval_profile=dense_lexical_hybrid`가 출력되어야 합니다.

### 3. 인덱스 사전 생성

```bash
python3 scripts/build_indexes.py
```

이 스크립트는 corpus를 읽고 현재 활성화된 retrieval backend 기준으로 인덱스를 `index/` 아래에 만듭니다.

출력 로그에서 아래를 바로 확인할 수 있습니다.

- 현재 데이터 경로
- 활성화된 dense/sparse/multivector backend 이름
- 각 경로가 참조하는 모델 디렉터리
- dense / sparse / multivector 인덱스 생성 완료 여부

### 4. 서버 실행

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

간단 확인:

```bash
curl http://127.0.0.1:8000/health
```

`/health` 응답에는 현재 retrieval 구조가 함께 포함됩니다.

- `retrieval_profile`
- `dense_backend`
- `sparse_backend`
- `sparse_backend_kind`
- `multivector_backend`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"id":"demo-1","question":"한국계란유통협회의 위반 사실과 처분 내용을 설명해 주세요."}'
```

## 로컬 평가

로컬 평가는 `scripts/` 기준으로 아래 두 스크립트를 조합해서 진행합니다.

- [scripts/build_indexes.py](/home/ming9/skn25-fairdata-competition/scripts/build_indexes.py:1)
- [scripts/evaluate_local.py](/home/ming9/skn25-fairdata-competition/scripts/evaluate_local.py:1)

권장 순서는 아래와 같습니다.

1. `python3 scripts/build_indexes.py`
2. `uvicorn server:app --host 0.0.0.0 --port 8000`
3. `python3 scripts/evaluate_local.py ...`

지원하는 평가셋 형식은 두 가지입니다.

- JSONL
  - `id`, `question`, `gold_chunk_ids`, `gold_answer`
- JSON 배열
  - 현재 저장소의 `data/test/eval_dataset_260505.json`
  - `query`, `answer_chunks[].chunk_id`, `reference_answer`를 사용

실행 예시:

`evaluate_local.py` 는 심사 환경과 동일하게 `server.py` 가 제공하는 `POST /predict` 를 HTTP로 호출합니다.

먼저 서버를 띄운 뒤 평가를 실행하면 됩니다.

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

```bash
python3 scripts/evaluate_local.py \
  --base-url http://127.0.0.1:8000 \
  --eval-file ./data/test/eval_dataset_260505.json
```

`evaluate_local.py` 는 `server.py` 의 `/predict`를 실제로 호출하므로, 제출 환경과 가장 비슷한 방식으로 로컬 평가를 돌립니다.

기본적으로 결과는 `results/` 아래에 자동 저장됩니다.

- 실행별 폴더: `results/<평가셋이름>_offset0_limitall/`
- 종합 성능 지표: `results/<평가셋이름>_offset0_limitall/summary.json`
- 문항별 예측 결과: `results/<평가셋이름>_offset0_limitall/predictions.jsonl`

`summary.json`의 `config_snapshot`에는 아래 같은 retrieval 메타데이터가 포함됩니다.

- `retrieval_profile`
- `dense_backend`
- `sparse_backend`
- `sparse_backend_kind`
- `multivector_backend`
- `index_namespace`

`--results-dir` 로 다른 저장 위치를 지정할 수도 있습니다.

일부 문항만 빠르게 점검하려면 `--limit`, `--offset`을 사용할 수 있습니다.

```bash
python3 scripts/evaluate_local.py \
  --eval-file ./data/test/eval_dataset_260505.json \
  --limit 10
```

```bash
python3 scripts/evaluate_local.py \
  --eval-file ./data/test/eval_dataset_260505.json \
  --offset 50 \
  --limit 20
```

평가 지표:

- `Recall@5`
- `MRR`
- `token_f1`
- `bertscore_f1` (`bert-score` 설치 시)
- `final_score` (`0.35 * Recall@5 + 0.15 * MRR + 0.30 * BERTScore + 0.20 * F1`)

### 평가 방법

이 섹션은 [scripts/evaluate_local.py](/home/ming9/skn25-fairdata-competition/scripts/evaluate_local.py:1) 와 [app/evaluation/metrics.py](/home/ming9/skn25-fairdata-competition/app/evaluation/metrics.py:1) 기준으로 정리한 실제 평가 흐름입니다.

`evaluate_local.py` 는 모델 파일을 직접 읽어서 점수를 계산하지 않습니다. 반드시 먼저 올라와 있는 `server.py`의 `POST /predict`를 HTTP로 호출하고, 그 응답을 평가합니다. 즉 평가는 항상 아래 입출력 계약을 전제로 합니다.

- 입력 요청
  - `{"id": "...", "question": "..."}`
- 기대 응답
  - `{"id": "...", "retrieved_chunk_ids": [...], "answer": "..."}`

코드 흐름은 아래 순서입니다.

1. `--eval-file`에서 평가셋을 읽습니다.
   - JSONL이면 각 줄에 `id`, `question`, `gold_chunk_ids`, `gold_answer`가 있어야 합니다.
   - JSON 배열이면 저장소의 변환 규칙에 맞춰 같은 내부 포맷으로 변환합니다.
2. 각 문항마다 `/predict`를 1번 호출합니다.
3. 응답에서 `retrieved_chunk_ids`, `answer`를 꺼냅니다.
4. retrieval 지표와 answer 지표를 각각 계산합니다.
5. 모든 문항의 평균을 내고 `summary.json`과 `predictions.jsonl`에 저장합니다.

문항별로 계산하는 값은 아래와 같습니다.

- `recall_at_5`
  - 상위 5개 `retrieved_chunk_ids` 안에 gold chunk가 얼마나 포함됐는지 계산합니다.
  - 구현은 [app/evaluation/metrics.py](/home/ming9/skn25-fairdata-competition/app/evaluation/metrics.py:132) 의 `compute_recall_at_5()` 입니다.
- `mrr`
  - 첫 정답 chunk가 top-5 안에서 몇 번째에 나왔는지의 역수입니다.
  - 구현은 [app/evaluation/metrics.py](/home/ming9/skn25-fairdata-competition/app/evaluation/metrics.py:151) 의 `compute_mrr()` 입니다.
- `token_f1`
  - 생성 답변과 gold answer를 SQuAD 방식 토큰 F1으로 비교합니다.
  - 구현은 [app/evaluation/metrics.py](/home/ming9/skn25-fairdata-competition/app/evaluation/metrics.py:184) 의 `compute_token_f1()` 입니다.

전체 문항을 모은 뒤 추가로 계산하는 값은 아래입니다.

- `bertscore_f1`
  - 전체 예측 답변 리스트와 정답 답변 리스트를 한국어 BERTScore로 비교합니다.
  - 구현은 [app/evaluation/metrics.py](/home/ming9/skn25-fairdata-competition/app/evaluation/metrics.py:202) 의 `compute_bertscore_f1()` 입니다.
- `final_score`
  - `0.35 * recall_at_5 + 0.15 * mrr + 0.30 * bertscore_f1 + 0.20 * token_f1`
  - 구현은 [app/evaluation/metrics.py](/home/ming9/skn25-fairdata-competition/app/evaluation/metrics.py:216) 의 `compute_final_score()` 입니다.

산출 파일도 역할이 나뉩니다.

- `predictions.jsonl`
  - 문항별 원시 결과 파일입니다.
  - 각 줄에 `id`, `question`, `gold_chunk_ids`, `gold_answer`, `predicted_chunk_ids`, `predicted_answer`, `recall_at_5`, `mrr`, `token_f1`가 들어갑니다.
- `summary.json`
  - 전체 평균 지표와 최종 점수 요약 파일입니다.
  - `config_snapshot`도 같이 저장돼서, 어떤 retrieval/reranker/llm 조합으로 나온 결과인지 나중에 추적할 수 있습니다.

중요한 해석 포인트는 아래입니다.

- 이 평가는 retrieval과 generation을 분리해서 따로 측정하지 않고, `/predict` 응답 하나를 기준으로 둘 다 함께 측정합니다.
- `Recall@5`, `MRR`는 검색 품질을 주로 보고, `token_f1`, `bertscore_f1`는 답변 품질을 봅니다.
- 따라서 같은 LLM을 두고 retrieval만 바꾸거나, 같은 retrieval을 두고 LLM만 바꿔도 `final_score`가 달라질 수 있습니다.

평가셋 현황:

- `data/test/eval_dataset_260505.json`
- 문항 수: 356

문항별 예측 파일 `predictions.jsonl` 에는 아래 정보가 함께 저장됩니다.

- `id`
- `question`
- `gold_chunk_ids`
- `gold_answer`
- `predicted_chunk_ids`
- `predicted_answer`
- `recall_at_5`
- `mrr`
- `token_f1`

## 현재 환경에서 확인한 사항

이 README는 코드 전체를 읽고 현재 워크스페이스 상태까지 반영해서 작성했습니다.

마지막 확인 시각:

- `2026-06-05 11:59:53 KST`

확인 결과:

- `models/` 디렉터리는 현재 비어 있습니다.
- `transformers` 패키지가 현재 Python 환경에 설치되어 있지 않습니다.
- `torch`, `chromadb`, `scipy`, `fastapi`는 설치되어 있습니다.
- 따라서 현재 워크스페이스 상태 그대로는 검색/생성/로컬 평가를 끝까지 실행할 수 없습니다.

즉, 실제 평가를 돌리려면 최소한 아래가 먼저 필요합니다.

1. `transformers` 설치
2. `models/embedding/bge-m3`, `models/reranker/bge-reranker-v2-m3`, `models/llm/qwen2.5-7b-instruct` 준비
3. BGE-M3용 `sparse_linear.pt`, `colbert_linear.pt` 준비
4. `python3 scripts/validate_model_matrix.py` 실행
5. `python3 scripts/build_indexes.py` 실행

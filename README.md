# 공정위 FairData RAG 제출 코드

공정거래위원회 FairData 공모전 Track 2 제출용 RAG 서버입니다.  
현재 코드는 FastAPI 기반 `/health`, `/predict` API를 제공하고, 검색은 backend 분리형 retrieval + reranker, 생성은 env 기반 LLM backend 구조로 구성돼 있습니다.

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

루트 기준 실행 흐름은 아래와 같습니다.

1. `server.py` import 시점에 `load_corpus()`가 `data/raw/*_metadata.json`, `*_hybrid.json`을 읽어 문서/청크 저장소를 만듭니다.
2. `HybridRetriever`가 질문을 `QueryAnalysis`로 분석합니다.
3. 활성화된 검색 경로 `dense`, `sparse`, `multi-vector`를 각각 전체 코퍼스에 대해 실행합니다.
4. 선택된 검색 경로 결과만 `RRF` 또는 score fusion으로 합칩니다.
5. 라우팅 일치도(`theme`, `focus`, `legal_role`, `industry`, `company_size`)를 retrieval bonus로 더합니다.
6. 현재 선택된 reranker backend가 상위 후보를 재정렬합니다.
7. 최종 5개 청크를 현재 선택된 LLM backend에 넘겨 답변을 생성합니다.

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
  - 어떤 retrieval / reranker / LLM backend를 쓸지, 어떤 retrieval 조합이 허용되는지 해석
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

### 4. 다른 retrieval 모델을 붙이는 경우

예를 들어 dense만 다른 임베딩 모델로 바꾸려면 아래 순서로 합니다.

1. [app/retrieval/interfaces.py](/home/ming9/skn25-fairdata-competition/app/retrieval/interfaces.py:8) 의 `DenseRetrievalBackend` 인터페이스를 구현하는 새 클래스를 추가
2. [app/retrieval/backends.py](/home/ming9/skn25-fairdata-competition/app/retrieval/backends.py:77) 의 `build_dense_backend()`에 새 backend 이름 등록
3. [app/utils/config.py](/home/ming9/skn25-fairdata-competition/app/utils/config.py:8) 의 `DENSE_BACKEND` 값을 새 backend 이름으로 변경
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
3. [app/utils/config.py](/home/ming9/skn25-fairdata-competition/app/utils/config.py:11) 의 `RERANK_BACKEND` 값을 새 backend 이름으로 변경
4. `resolve_bge_reranker_model_dir()`가 새 모델 디렉터리를 가리키도록 수정

즉 현재 기본 구현은 `bge_reranker`지만, 다른 cross-encoder 계열 리랭커도 같은 패턴으로 꽂을 수 있습니다.

### 6. RRF를 넣었다 뺐다 하는 경우

[app/utils/config.py](/home/ming9/skn25-fairdata-competition/app/utils/config.py:12) 의 `USE_RRF_FUSION`을 수정합니다.

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
  - `DENSE_BACKEND`
  - `SPARSE_BACKEND`
  - `MULTIVECTOR_BACKEND`
- reranker backend 종류
  - `RERANK_BACKEND`
- fusion 방식
  - `USE_RRF_FUSION`
- 모델 경로
  - `resolve_dense_model_dir()`
  - `resolve_sparse_model_dir()`
  - `resolve_multivector_model_dir()`
  - `resolve_bge_reranker_model_dir()`
  - `resolve_qwen_model_dir()`
- 경로 on/off
  - `is_dense_enabled()`
  - `is_sparse_enabled()`
  - `is_multivector_enabled()`

## 모델 경로

코드가 기본으로 기대하는 대표 모델 경로는 아래와 같습니다.

- BGE-M3: `models/bge-m3`
- BM25 runtime placeholder: `models/bm25`
- BGE reranker: `models/bge-reranker-v2-m3`
- Qwen 생성 모델: `models/qwen2.5-7b-instruct`

경로를 바꾸려면 [app/utils/config.py](/home/ming9/skn25-fairdata-competition/app/utils/config.py:1) 의 반환값을 직접 수정하면 됩니다.

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

확인 결과:

- `models/` 디렉터리는 현재 비어 있습니다.
- `transformers` 패키지가 현재 Python 환경에 설치되어 있지 않습니다.
- `torch`, `chromadb`, `scipy`, `fastapi`는 설치되어 있습니다.
- 따라서 현재 워크스페이스 상태 그대로는 검색/생성/로컬 평가를 끝까지 실행할 수 없습니다.

즉, 실제 평가를 돌리려면 최소한 아래가 먼저 필요합니다.

1. `transformers` 설치
2. `models/bge-m3`, `models/bge-reranker-v2-m3`, `models/qwen2.5-7b-instruct` 준비
3. BGE-M3용 `sparse_linear.pt`, `colbert_linear.pt` 준비
4. `python3 scripts/validate_model_matrix.py` 실행
5. `python3 scripts/build_indexes.py` 실행

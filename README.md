# 공정위 FairData RAG 제출 코드

공정거래위원회 FairData 공모전 Track 2 제출용 RAG 서버입니다.  
현재 코드는 FastAPI 기반 `/health`, `/predict` API를 제공하고, 검색은 BGE-M3 hybrid retrieval + reranker, 생성은 Qwen2-7B-Instruct를 사용하도록 구성돼 있습니다.

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
│   └── evaluate_local.py
├── Dockerfile
├── requirements.txt
└── server.py
```

## 서버 동작 방식

서버 진입점은 [server.py](/home/ming9/skn25-fairdata-competition/server.py:1) 입니다.

- `GET /health`
  - 코퍼스 청크 수와 data 경로를 반환합니다.
- `POST /predict`
  - 입력: `id`, `question`
  - 출력: `id`, `retrieved_chunk_ids`, `answer`

실행 흐름은 아래와 같습니다.

1. `load_corpus()`가 `data/raw/*_metadata.json`, `*_hybrid.json`을 읽어 문서/청크 저장소를 만듭니다.
2. `HybridRetriever`가 질의를 분석하고 문서 후보를 먼저 줄입니다.
3. BGE-M3로 `dense`, `sparse`, `multi-vector` 검색을 각각 수행합니다.
4. 세 경로 결과를 RRF로 합칩니다.
5. `bge-reranker-v2-m3`로 상위 후보를 재정렬합니다.
6. 최종 5개 청크를 Qwen2-7B-Instruct에 넘겨 답변을 생성합니다.

## 검색 인덱스

인덱스는 `models/`가 아니라 `index/` 아래에 저장됩니다.

- dense: `index/chroma_bgem3`
- sparse: `index/sparse_bgem3_chunks.npz`
- multi-vector: `index/multivector_bgem3_chunks.npz`

각 인덱스는 manifest와 corpus fingerprint를 비교해서:

- 기존 인덱스가 유효하면 재사용
- corpus가 바뀌었으면 자동 재생성

## 모델 경로

코드가 기본으로 기대하는 모델 경로는 아래와 같습니다.

- BGE-M3: `models/bge-m3`
- BGE reranker: `models/bge-reranker-v2-m3`
- Qwen 생성 모델: `models/qwen2-7b-instruct`

환경변수로 바꿀 수 있습니다.

```bash
export FAIRDATA_BGEM3_MODEL_DIR="$(pwd)/models/bge-m3"
export FAIRDATA_BGE_RERANKER_MODEL_DIR="$(pwd)/models/bge-reranker-v2-m3"
export FAIRDATA_QWEN_MODEL_DIR="$(pwd)/models/qwen2-7b-instruct"
export FAIRDATA_INDEX_ROOT_DIR="$(pwd)/index"
export FAIRDATA_DATA_DIR="$(pwd)/data/raw"
```

## 준비 절차

### 1. 모델 다운로드

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

주의:

- BGE-M3는 추가로 `sparse_linear.pt`, `colbert_linear.pt`가 필요합니다.
- 현재 다운로드 스크립트는 이 두 파일이 자동으로 채워진다고 보장하지 않습니다.
- 이 두 파일이 없으면 hybrid retrieval 초기화가 실패합니다.

### 2. 인덱스 사전 생성

```bash
python3 scripts/build_indexes.py
```

이 스크립트는 corpus를 읽고 dense / sparse / multi-vector 인덱스를 `index/` 아래에 만듭니다.

### 3. 서버 실행

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

간단 확인:

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"id":"demo-1","question":"한국계란유통협회의 위반 사실과 처분 내용을 설명해 주세요."}'
```

## 로컬 평가

로컬 평가는 [scripts/evaluate_local.py](/home/ming9/skn25-fairdata-competition/scripts/evaluate_local.py:1) 를 사용합니다.

지원하는 평가셋 형식은 두 가지입니다.

- JSONL
  - `id`, `question`, `gold_chunk_ids`, `gold_answer`
- JSON 배열
  - 현재 저장소의 `data/test/eval_dataset_260505.json`
  - `query`, `answer_chunks[].chunk_id`, `reference_answer`를 사용

실행 예시:

```bash
python3 scripts/evaluate_local.py \
  --eval-file ./data/test/eval_dataset_260505.json \
  --output-file ./data/test/eval_predictions_260505.jsonl
```

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

## 현재 환경에서 확인한 사항

이 README는 코드 전체를 읽고 현재 워크스페이스 상태까지 반영해서 작성했습니다.

확인 결과:

- `models/` 디렉터리는 현재 비어 있습니다.
- `transformers` 패키지가 현재 Python 환경에 설치되어 있지 않습니다.
- `torch`, `chromadb`, `scipy`, `fastapi`는 설치되어 있습니다.
- 따라서 현재 워크스페이스 상태 그대로는 검색/생성/로컬 평가를 끝까지 실행할 수 없습니다.

즉, 실제 평가를 돌리려면 최소한 아래가 먼저 필요합니다.

1. `transformers` 설치
2. `models/bge-m3`, `models/bge-reranker-v2-m3`, `models/qwen2-7b-instruct` 준비
3. BGE-M3용 `sparse_linear.pt`, `colbert_linear.pt` 준비
4. `python3 scripts/build_indexes.py` 실행



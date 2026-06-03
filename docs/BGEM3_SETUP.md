# BGE-M3 준비 가이드

현재 검색 코드는 아래 경로 중 하나에서 BGE-M3 모델을 찾습니다.

- 환경변수 `FAIRDATA_BGEM3_MODEL_DIR`
- 기본 경로 `models/bge-m3`

검색 인덱스는 모델과 분리해서 아래 경로에 저장합니다.

- 환경변수 `FAIRDATA_INDEX_ROOT_DIR` 또는 `FAIRDATA_INDEX_DIR`
- 기본 경로 `index/`

답변 생성기는 아래 경로에서 Qwen 모델을 찾습니다.

- 환경변수 `FAIRDATA_QWEN_MODEL_DIR` 또는 `FAIRDATA_GENERATION_MODEL_DIR`
- 기본 경로 `models/qwen2-7b-instruct`

## 기대 디렉터리 구조

```text
models/bge-m3/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── sparse_linear.pt
└── colbert_linear.pt
```

## 리눅스 환경 세팅

```bash
bash scripts/setup_linux_bgem3_env.sh
source .venv/bin/activate
```

## BGE-M3 본체 다운로드

```bash
bash scripts/download_bgem3_model.sh
```

이 스크립트는 기본적으로 아래 모델을 모두 `models/` 아래에 내려받습니다.

- `models/bge-m3`
- `models/bge-reranker-v2-m3`
- `models/qwen2-7b-instruct`

필요하면 아래 환경변수로 개별 다운로드를 끌 수 있습니다.

```bash
DOWNLOAD_BGEM3=1 DOWNLOAD_RERANKER=1 DOWNLOAD_QWEN=0 bash scripts/download_bgem3_model.sh
```

## 헤드 가중치 확인

현재 코드가 사용하는 `sparse_linear.pt`, `colbert_linear.pt`는 BGE-M3 sparse / multi-vector 검색 경로용 추가 가중치입니다.
다운로드 이후 아래 검증을 수행하세요.

```bash
bash scripts/verify_bgem3_layout.sh
```

## 검색 인덱스 사전 생성

```bash
python3 scripts/build_indexes.py
```

위 스크립트는 아래 인덱스를 `index/` 아래에 생성하거나, 이미 유효한 저장본이 있으면 재사용합니다.

- dense Chroma 인덱스
- sparse matrix 인덱스
- multi-vector 인덱스

## 로컬 평가

```bash
python3 scripts/evaluate_local.py --eval-file ./eval.jsonl --output-file ./eval_predictions.jsonl
```

평가셋은 JSONL 형식이며 각 줄은 아래 키를 포함해야 합니다.

- `id`
- `question`
- `gold_chunk_ids`
- `gold_answer`

저장소에 포함된 `data/test/eval_dataset_260505.json` 배열 형식도 그대로 지원합니다.

## 환경변수 예시

```bash
export FAIRDATA_BGEM3_MODEL_DIR="$(pwd)/models/bge-m3"
export FAIRDATA_QWEN_MODEL_DIR="$(pwd)/models/qwen2-7b-instruct"
export FAIRDATA_DATA_DIR="$(pwd)/data/raw"
export FAIRDATA_INDEX_ROOT_DIR="$(pwd)/index"
```

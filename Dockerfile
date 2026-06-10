FROM python:3.11-slim

WORKDIR /workspace/skn25-fairdata-competition

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV CUDA_VISIBLE_DEVICES=0
ENV FAIRDATA_EXPERIMENT_TAG=V2-E5-BGEM3-DS-QM-MV-W07-TOP75-EXAONE
ENV FAIRDATA_ENABLE_DENSE=1
ENV FAIRDATA_ENABLE_SPARSE=1
ENV FAIRDATA_ENABLE_MULTIVECTOR=1
ENV FAIRDATA_DENSE_BACKEND=bgem3
ENV FAIRDATA_SPARSE_BACKEND=bgem3
ENV FAIRDATA_MULTIVECTOR_BACKEND=bgem3
ENV FAIRDATA_MULTIVECTOR_WEIGHT=0.7
ENV FAIRDATA_MULTIVECTOR_QM_ONLY=1
ENV FAIRDATA_RERANK_BACKEND=bge_reranker
ENV FAIRDATA_RERANK_TOP_N=75
ENV FAIRDATA_RERANK_WEIGHT=1.0
ENV FAIRDATA_LLM_BACKEND=exaone-3.5-7.8b-instruct
ENV FAIRDATA_LLM_TRUST_REMOTE_CODE=1
ENV FAIRDATA_LLM_DEVICE=cuda:0
ENV FAIRDATA_RERANK_DEVICE=cuda
ENV FAIRDATA_EMBED_BATCH_SIZE=8
ENV FAIRDATA_GENERATION_MAX_INPUT_CHARS=3500
ENV FAIRDATA_GENERATION_MAX_NEW_TOKENS=160
ENV FAIRDATA_ROUTE_MAX_NEW_TOKENS=64

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY server.py ./server.py
COPY data ./data
COPY index/chroma_dense_bgem3__sparse_bgem3__multivector_bgem3 ./index/chroma_dense_bgem3__sparse_bgem3__multivector_bgem3
COPY index/sparse_bgem3sparse_chunks_1c2ac4dc1816.npz ./index/sparse_bgem3sparse_chunks_1c2ac4dc1816.npz
COPY index/sparse_bgem3sparse_chunks_1c2ac4dc1816_manifest.json ./index/sparse_bgem3sparse_chunks_1c2ac4dc1816_manifest.json
COPY models/embedding/bge-m3 ./models/embedding/bge-m3
COPY models/reranker/bge-reranker-v2-m3 ./models/reranker/bge-reranker-v2-m3
COPY models/llm/exaone-3.5-7.8b-instruct ./models/llm/exaone-3.5-7.8b-instruct

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

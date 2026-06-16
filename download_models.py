import os, subprocess, time
from pathlib import Path

MODELS_DIR = Path("/workspace/skn25-fairdata-competition/models")
GROUPS = {
    "embedding": [
        ("BAAI/bge-m3", "bge-m3"),
        ("upskyy/bge-m3-korean", "bge-m3-korean"),
        ("intfloat/multilingual-e5-large", "multilingual-e5-large"),
        ("jhgan/ko-sbert-nli", "ko-sbert-nli"),
        ("jinaai/jina-embeddings-v4", "jina-embeddings-v4"),
        ("Alibaba-NLP/gte-multilingual-base", "gte-multilingual-base"),
        ("nlpai-lab/KURE-v1", "KURE-v1"),
        ("dragonkue/snowflake-arctic-embed-l-v2.0-ko", "snowflake-arctic-embed-l-v2.0-ko"),
    ],
    "reranker": [
        ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3"),
        ("BAAI/bge-reranker-v2-gemma", "bge-reranker-v2-gemma"),
        ("BAAI/bge-reranker-v2.5-gemma2-lightweight", "bge-reranker-v2.5-gemma2-lightweight"),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-6-v2"),
        ("jinaai/jina-reranker-v3", "jina-reranker-v3"),
        ("Dongjin-kr/ko-reranker", "ko-reranker"),
    ],
    "llm": [
        ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"),
        ("Qwen/Qwen3-8B", "qwen3-8b"),
        ("Qwen/Qwen3-4B", "qwen3-4b"),
        ("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", "exaone-3.5-7.8b-instruct"),
        ("beomi/Llama-3-Open-Ko-8B", "llama-3-open-ko-8b"),
        ("NCSOFT/Llama-VARCO-8B-Instruct", "llama-varco-8b-instruct"),
        ("microsoft/Phi-3.5-mini-instruct", "phi-3.5-mini-instruct"),
    ],
}

total = sum(len(v) for v in GROUPS.values())
done = 0
failed = []
MODELS_DIR.mkdir(parents=True, exist_ok=True)

for gname, models in GROUPS.items():
    gdir = MODELS_DIR / gname
    gdir.mkdir(parents=True, exist_ok=True)
    for repo_id, local_name in models:
        target = gdir / local_name
        if target.exists() and any(target.iterdir()):
            print(f"[SKIP] {repo_id} -> {target}")
            done += 1
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"[{done+1}/{total}] {repo_id} -> {target}")
        rc = subprocess.run(
            ["hf", "download", repo_id, "--local-dir", str(target)],
            capture_output=True, text=True, timeout=1800,
        )
        if rc.returncode == 0:
            done += 1
            print("  OK")
        else:
            failed.append(repo_id)
            print(f"  FAILED: {rc.stderr.strip()[:300]}")

sep = "=" * 50
print(f"\n{sep}")
print(f"COMPLETE: {done}/{total}")

# Show disk usage
import subprocess as sp
du = sp.run(["du", "-sh", str(MODELS_DIR)], capture_output=True, text=True)
print(f"Total size: {du.stdout.strip()}")

if failed:
    print(f"FAILED ({len(failed)}):")
    for f in failed:
        print(f"  - {f}")
    exit(1)
else:
    print("ALL MODELS DOWNLOADED SUCCESSFULLY")

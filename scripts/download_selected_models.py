from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

BASE = Path('/workspace/skn25-fairdata-competition/models')
HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN') or None

MODELS = [
    {
        'repo_id': 'BAAI/bge-m3',
        'local_dir': BASE / 'embedding' / 'bge-m3',
        'allow_patterns': [
            'config.json', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'sentencepiece.bpe.model',
            'pytorch_model.bin', 'sparse_linear.pt', 'colbert_linear.pt',
        ],
    },
    {
        'repo_id': 'BAAI/bge-reranker-v2-m3',
        'local_dir': BASE / 'reranker' / 'bge-reranker-v2-m3',
        'allow_patterns': [
            'config.json', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'sentencepiece.bpe.model',
            'model.safetensors',
        ],
    },
    {
        'repo_id': 'Qwen/Qwen2.5-7B-Instruct',
        'local_dir': BASE / 'llm' / 'qwen2.5-7b-instruct',
        'allow_patterns': [
            'config.json', 'generation_config.json', 'tokenizer.json',
            'tokenizer_config.json', 'vocab.json', 'merges.txt',
            'model.safetensors.index.json', '*.safetensors',
        ],
    },
]

for item in MODELS:
    local_dir = Path(item['local_dir'])
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {item['repo_id']} -> {local_dir}", flush=True)
    snapshot_download(
        repo_id=item['repo_id'],
        local_dir=str(local_dir),
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
        allow_patterns=item['allow_patterns'],
    )
print('[download] selected models complete', flush=True)

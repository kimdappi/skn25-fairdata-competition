import os

import torch

from baseline_rag.config import REQUIRE_CUDA


def require_runtime_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if REQUIRE_CUDA:
        raise RuntimeError(
            "CUDA GPU is required for this run, but no CUDA device is available. "
            "In Colab, switch Runtime to GPU and remount Drive before running."
        )
    return torch.device("cpu")


def preferred_torch_dtype() -> torch.dtype:
    dtype_name = os.getenv("FAIRCOMP_TORCH_DTYPE", "float16").strip().lower()
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32

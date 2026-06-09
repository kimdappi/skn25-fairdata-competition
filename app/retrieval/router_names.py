from __future__ import annotations


ROUTER_BACKEND_ALIASES = {
    "rule": "keyword",
    "keyword": "keyword",
    "keywords": "keyword",
    "llm": "lcel",
    "lcel": "lcel",
    "lcel_prompt_boost": "lcel_prompt_boost",
    "lcel_prompt_enhanced": "lcel_prompt_boost",
    "prompt_boost": "lcel_prompt_boost",
}


def normalize_router_backend_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(".", "_").replace("/", "_")
    return ROUTER_BACKEND_ALIASES.get(normalized, normalized)

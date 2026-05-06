import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib import error, request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline_rag.config import (
    HF_ROUTER_LOCAL_FILES_ONLY,
    HF_ROUTER_MODEL,
    MODEL_DIR,
    OLLAMA_ROUTER_ENABLED,
    OLLAMA_ROUTER_MODEL,
    OLLAMA_ROUTER_TIMEOUT_SECONDS,
    OLLAMA_ROUTER_URL,
    ROUTER_LOG_ENABLED,
    ROUTER_BACKEND,
    STAGE_LOG_ENABLED,
)
from baseline_rag.keywords import ENTERPRISE_SIZE_KEYWORDS, INDUSTRY_KEYWORDS, LEGAL_ROLE_KEYWORDS, THEME_KEYWORDS
from baseline_rag.retrieval_utils import (
    FOCUS_FACT,
    FOCUS_GENERAL,
    FOCUS_LAW,
    FOCUS_ORDER,
    OTHER,
    build_rule_route_fields,
)
from baseline_rag.runtime import preferred_torch_dtype, require_runtime_device
from baseline_rag.schemas import RouteDecision

# This project uses sentence-transformers with PyTorch only.
# Disabling TF import avoids the Keras 3 compatibility error in transformers.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


class QueryRouter:
    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        model_name: Optional[str] = None,
        url: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        backend: Optional[str] = None,
        log_enabled: Optional[bool] = None,
        stage_log_enabled: Optional[bool] = None,
    ) -> None:
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("OLLAMA_ROUTER_ENABLED", str(OLLAMA_ROUTER_ENABLED)).lower() in {"1", "true", "yes", "on"}
        )
        self.model_name = os.getenv("OLLAMA_ROUTER_MODEL", model_name or OLLAMA_ROUTER_MODEL)
        self.url = os.getenv("OLLAMA_ROUTER_URL", url or OLLAMA_ROUTER_URL)
        self.backend = os.getenv("ROUTER_BACKEND", backend or ROUTER_BACKEND).strip().lower()
        self.hf_model_name = os.getenv("HF_ROUTER_MODEL", HF_ROUTER_MODEL)
        self.hf_local_files_only = os.getenv(
            "HF_ROUTER_LOCAL_FILES_ONLY",
            str(HF_ROUTER_LOCAL_FILES_ONLY),
        ).lower() in {"1", "true", "yes", "on"}
        default_timeout = timeout_seconds if timeout_seconds is not None else OLLAMA_ROUTER_TIMEOUT_SECONDS
        self.timeout_seconds = int(os.getenv("OLLAMA_ROUTER_TIMEOUT_SECONDS", str(default_timeout)))
        self.log_enabled = (
            log_enabled
            if log_enabled is not None
            else os.getenv("ROUTER_LOG_ENABLED", str(ROUTER_LOG_ENABLED)).lower() in {"1", "true", "yes", "on"}
        )
        self.stage_log_enabled = (
            stage_log_enabled
            if stage_log_enabled is not None
            else os.getenv("STAGE_LOG_ENABLED", str(STAGE_LOG_ENABLED)).lower() in {"1", "true", "yes", "on"}
        )
        self.local_model_path = self._resolve_local_router_model_path(self.model_name)
        self.hf_model_path = self._resolve_local_router_model_path(self.hf_model_name)
        self.allow_cpu = os.getenv("FAIRCOMP_ROUTER_ALLOW_CPU", "false").lower() in {"1", "true", "yes", "on"}
        self.local_tokenizer = None
        self.local_model = None

    def _model_input_device(self) -> torch.device:
        if self.local_model is None:
            return torch.device("cpu")
        try:
            return next(self.local_model.parameters()).device
        except StopIteration:
            pass
        model_device = getattr(self.local_model, "device", None)
        if model_device is not None:
            return model_device
        return torch.device("cpu")

    def _move_inputs_to_model_device(self, encoded: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = self._model_input_device()
        return {key: value.to(device) for key, value in encoded.items()}

    def _log_router(self, message: str) -> None:
        if self.log_enabled:
            print(f"[router] {message}")

    def _log_stage(self, stage: str, started_at: float, *, extra: Optional[str] = None) -> None:
        if not self.stage_log_enabled:
            return
        elapsed = time.perf_counter() - started_at
        suffix = f" {extra}" if extra else ""
        print(f"[stage] {stage} {elapsed:.3f}s{suffix}")

    def route_from_text(self, text: str) -> RouteDecision:
        return RouteDecision(**build_rule_route_fields(text))

    def route(self, query: str) -> RouteDecision:
        route = self._route_with_backend(query)
        if route is not None:
            return route
        fallback_route = self.route_from_text(query)
        self._log_router(f"Fallback route query={query!r} final={json.dumps(fallback_route.model_dump(), ensure_ascii=False)}")
        return fallback_route

    def route_many(self, queries: List[str], batch_size: int = 12) -> List[RouteDecision]:
        if not queries:
            return []
        if not self.enabled or not self.model_name:
            return [self.route_from_text(query) for query in queries]

        if self.backend == "hf":
            return [self._route_with_hf_transformers(query) or self.route_from_text(query) for query in queries]

        routes: List[Optional[RouteDecision]] = [None] * len(queries)
        for batch_start in range(0, len(queries), batch_size):
            batch_queries = queries[batch_start : batch_start + batch_size]
            payload = {
                "model": self.model_name,
                "prompt": self._build_batch_route_prompt(batch_queries),
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 768,
                },
            }
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout_seconds * 4) as response:
                    body = response.read().decode("utf-8")
                result = json.loads(body)
                raw_text = result.get("response", "")
                json_text = self._extract_json_array(raw_text) or raw_text
                items = json.loads(json_text)
                if not isinstance(items, list):
                    raise ValueError("Batch route response is not a JSON array.")
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    local_index = item.get("index")
                    if not isinstance(local_index, int) or not (0 <= local_index < len(batch_queries)):
                        continue
                    absolute_index = batch_start + local_index
                    routes[absolute_index] = self._route_from_payload(batch_queries[local_index], item)
            except Exception as exc:
                self._log_router(
                    f"Batch route failed for queries {batch_start}-{batch_start + len(batch_queries) - 1}: {type(exc).__name__}: {exc}"
                )

            for local_index, query in enumerate(batch_queries):
                absolute_index = batch_start + local_index
                if routes[absolute_index] is None:
                    single_route = self._route_with_backend(query)
                    routes[absolute_index] = single_route or self.route_from_text(query)

        return [route if route is not None else self.route_from_text(query) for route, query in zip(routes, queries)]

    def _resolve_local_router_model_path(self, model_name: Optional[str]) -> Optional[Path]:
        if not model_name:
            return None
        candidates = [
            Path(model_name).expanduser(),
            MODEL_DIR / model_name,
            MODEL_DIR / model_name.replace(":", "_"),
            MODEL_DIR / model_name.replace(":", "-"),
            MODEL_DIR / "Qwen2.5-7B-Instruct",
        ]
        seen = set()
        for candidate in candidates:
            normalized = str(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            if candidate.exists():
                return candidate
        return None

    def _load_transformers_router_model(
        self,
        *,
        model_name_or_path: str | Path,
        local_files_only: bool,
        stage_name: str,
    ) -> bool:
        if self.local_model is not None and self.local_tokenizer is not None:
            return True
        started_at = time.perf_counter()
        try:
            self.local_tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
            )
            model_kwargs = {"local_files_only": local_files_only}
            device = require_runtime_device()
            if device.type != "cuda" and not self.allow_cpu:
                self._log_router(f"Skipping router model load on CPU for model={model_name_or_path}")
                self.local_tokenizer = None
                self.local_model = None
                return False
            if device.type == "cuda":
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = preferred_torch_dtype()
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs,
            )
            if device.type != "cuda":
                self.local_model.to(device)
            self.local_model.eval()
            self._log_stage(stage_name, started_at, extra=f"model={model_name_or_path}")
            return True
        except Exception as exc:
            self.local_tokenizer = None
            self.local_model = None
            self._log_router(f"Failed to load router model={model_name_or_path}: {type(exc).__name__}: {exc}")
            return False

    def _load_local_router_model(self) -> bool:
        if not self.local_model_path:
            return False
        return self._load_transformers_router_model(
            model_name_or_path=self.local_model_path,
            local_files_only=True,
            stage_name="init.router_local_model",
        )

    def _load_hf_router_model(self) -> bool:
        model_name_or_path: str | Path = self.hf_model_path or self.hf_model_name
        local_files_only = self.hf_local_files_only if self.hf_model_path is None else True
        return self._load_transformers_router_model(
            model_name_or_path=model_name_or_path,
            local_files_only=local_files_only,
            stage_name="init.router_hf_model",
        )

    def _allowed_route_values(self, keyword_map: Dict[str, List[str]]) -> List[str]:
        return list(keyword_map.keys()) + [OTHER]

    def _build_route_prompt(self, query: str) -> str:
        theme_values = ", ".join(self._allowed_route_values(THEME_KEYWORDS))
        size_values = ", ".join(self._allowed_route_values(ENTERPRISE_SIZE_KEYWORDS))
        role_values = ", ".join(self._allowed_route_values(LEGAL_ROLE_KEYWORDS))
        industry_values = ", ".join(self._allowed_route_values(INDUSTRY_KEYWORDS))
        focus_values = ", ".join([FOCUS_ORDER, FOCUS_LAW, FOCUS_FACT, FOCUS_GENERAL])
        return (
            "당신은 공정거래위원회 의결서 검색 시스템의 라우터입니다.\n"
            "사용자 질문을 검색 라우팅용 JSON으로만 분류하세요.\n"
            "설명이나 마크다운 없이 JSON 객체 하나만 출력하세요.\n"
            "필드는 theme, company_size, legal_role, industry, focus, keywords 입니다.\n"
            f"theme 허용값: {theme_values}\n"
            f"company_size 허용값: {size_values}\n"
            f"legal_role 허용값: {role_values}\n"
            f"industry 허용값: {industry_values}\n"
            f"focus 허용값: {focus_values}\n"
            "keywords는 검색에 중요하다고 판단되는 키워드 3개 이상 10개 이하의 문자열 배열입니다.\n"
            "질문만 보고 확실하지 않으면 기타 또는 일반을 선택하세요.\n"
            f"질문: {query}\n"
            '출력 예시: {"theme":"기타","company_size":"기타","legal_role":"기타","industry":"기타","focus":"일반","keywords":["키워드1","키워드2"]}'
        )

    def _build_batch_route_prompt(self, queries: List[str]) -> str:
        theme_values = ", ".join(self._allowed_route_values(THEME_KEYWORDS))
        size_values = ", ".join(self._allowed_route_values(ENTERPRISE_SIZE_KEYWORDS))
        role_values = ", ".join(self._allowed_route_values(LEGAL_ROLE_KEYWORDS))
        industry_values = ", ".join(self._allowed_route_values(INDUSTRY_KEYWORDS))
        focus_values = ", ".join([FOCUS_ORDER, FOCUS_LAW, FOCUS_FACT, FOCUS_GENERAL])
        query_lines = "\n".join(f"{idx}. {query}" for idx, query in enumerate(queries))
        return (
            "당신은 공정거래위원회 의결서 검색용 질의 라우터입니다.\n"
            "아래 여러 질문을 각각 검색 라우팅용 JSON으로 분류하세요.\n"
            "반드시 JSON 배열만 출력하세요.\n"
            "각 원소는 index, theme, company_size, legal_role, industry, focus, keywords 필드를 가져야 합니다.\n"
            f"theme 허용값: {theme_values}\n"
            f"company_size 허용값: {size_values}\n"
            f"legal_role 허용값: {role_values}\n"
            f"industry 허용값: {industry_values}\n"
            f"focus 허용값: {focus_values}\n"
            "keywords는 검색에 중요하다고 판단되는 키워드 3개 이상 10개 이하의 문자열 배열입니다.\n"
            "질문만 보고 확실하지 않으면 기타 또는 일반을 선택하세요.\n"
            f"질문 목록:\n{query_lines}\n"
            '출력 예시: [{"index":0,"theme":"기타","company_size":"기타","legal_role":"기타","industry":"기타","focus":"일반","keywords":["키워드1","키워드2","키워드3"]}]'
        )

    def _route_with_local_model(self, query: str) -> Optional[RouteDecision]:
        if not self._load_local_router_model():
            return None
        started_at = time.perf_counter()
        prompt = self._build_route_prompt(query)
        if self.local_tokenizer is None or self.local_model is None:
            return None
        try:
            encoded = self.local_tokenizer(prompt, return_tensors="pt")
            encoded = self._move_inputs_to_model_device(encoded)
            output = self.local_model.generate(
                **encoded,
                max_new_tokens=192,
                do_sample=False,
                pad_token_id=self.local_tokenizer.eos_token_id,
            )
            generated_tokens = output[0][encoded["input_ids"].shape[1] :]
            raw_text = self.local_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            self._log_stage("query.route_local_llm", started_at, extra=f"query={query!r}")
            return self._decode_route_payload(query, raw_text)
        except Exception as exc:
            self._log_router(f"Local router model failed for query={query!r}: {type(exc).__name__}: {exc}")
            return None

    def _route_with_hf_transformers(self, query: str) -> Optional[RouteDecision]:
        if not self.enabled or not self.hf_model_name:
            return None
        if not self._load_hf_router_model():
            return None
        started_at = time.perf_counter()
        prompt = self._build_route_prompt(query)
        if self.local_tokenizer is None or self.local_model is None:
            return None
        try:
            encoded = self.local_tokenizer(prompt, return_tensors="pt")
            encoded = self._move_inputs_to_model_device(encoded)
            output = self.local_model.generate(
                **encoded,
                max_new_tokens=192,
                do_sample=False,
                pad_token_id=self.local_tokenizer.eos_token_id,
            )
            generated_tokens = output[0][encoded["input_ids"].shape[1] :]
            raw_text = self.local_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            self._log_stage("query.route_hf_llm", started_at, extra=f"query={query!r}")
            return self._decode_route_payload(query, raw_text)
        except Exception as exc:
            self._log_router(f"HF router model failed for query={query!r}: {type(exc).__name__}: {exc}")
            return None

    def _route_with_backend(self, query: str) -> Optional[RouteDecision]:
        if self.backend == "hf":
            return self._route_with_hf_transformers(query)
        return self._route_with_ollama(query)

    def _route_with_ollama(self, query: str) -> Optional[RouteDecision]:
        if not self.enabled or not self.model_name:
            self._log_router("Ollama router disabled or model not configured; using fallback router.")
            return None
        local_route = self._route_with_local_model(query)
        if local_route is not None:
            return local_route
        payload = {
            "model": self.model_name,
            "prompt": self._build_route_prompt(query),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 96,
            },
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except (error.URLError, TimeoutError, OSError, ValueError) as exc:
            self._log_router(f"Ollama request failed for query={query!r}: {type(exc).__name__}: {exc}")
            return None

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            self._log_router(f"Failed to decode Ollama HTTP response for query={query!r}: {exc}")
            return None

        raw_text = result.get("response", "")
        return self._decode_route_payload(query, raw_text)

    def _extract_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    def _extract_json_array(self, text: str) -> Optional[str]:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    def _normalize_route_value(self, value: str, allowed_values: List[str], fallback: str) -> str:
        if value in allowed_values:
            return value
        compact_value = re.sub(r"\s+", "", value)
        for allowed in allowed_values:
            if compact_value == re.sub(r"\s+", "", allowed):
                return allowed
        return fallback

    def _decode_route_payload(self, query: str, raw_text: str) -> Optional[RouteDecision]:
        json_text = self._extract_json_object(raw_text) or raw_text
        try:
            route_payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            self._log_router(f"Failed to decode route JSON for query={query!r}: {exc}; raw={raw_text!r}")
            return None
        route = self._route_from_payload(query, route_payload)
        self._log_router(
            "LLM route "
            f"query={query!r} model={self.model_name} "
            f"raw={json.dumps(route_payload, ensure_ascii=False)} "
            f"final={json.dumps(route.model_dump(), ensure_ascii=False)}"
        )
        return route

    def _route_from_payload(self, query: str, route_payload: Dict[str, object]) -> RouteDecision:
        fallback_route = self.route_from_text(query)
        keywords = route_payload.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        normalized_keywords = [str(item).strip() for item in keywords if str(item).strip()]
        if not normalized_keywords:
            normalized_keywords = fallback_route.keywords
        llm_theme = self._normalize_route_value(
            str(route_payload.get("theme", "")),
            self._allowed_route_values(THEME_KEYWORDS),
            fallback_route.theme,
        )
        llm_company_size = self._normalize_route_value(
            str(route_payload.get("company_size", "")),
            self._allowed_route_values(ENTERPRISE_SIZE_KEYWORDS),
            fallback_route.company_size,
        )
        llm_legal_role = self._normalize_route_value(
            str(route_payload.get("legal_role", "")),
            self._allowed_route_values(LEGAL_ROLE_KEYWORDS),
            fallback_route.legal_role,
        )
        llm_industry = self._normalize_route_value(
            str(route_payload.get("industry", "")),
            self._allowed_route_values(INDUSTRY_KEYWORDS),
            fallback_route.industry,
        )
        llm_focus = self._normalize_route_value(
            str(route_payload.get("focus", "")),
            [FOCUS_ORDER, FOCUS_LAW, FOCUS_FACT, FOCUS_GENERAL],
            fallback_route.focus,
        )
        return RouteDecision(
            theme=fallback_route.theme if fallback_route.theme != OTHER else llm_theme,
            company_size=fallback_route.company_size if fallback_route.company_size != OTHER else llm_company_size,
            legal_role=fallback_route.legal_role if fallback_route.legal_role != OTHER else llm_legal_role,
            industry=fallback_route.industry if fallback_route.industry != OTHER else llm_industry,
            focus=fallback_route.focus if fallback_route.focus != FOCUS_GENERAL else llm_focus,
            keywords=normalized_keywords[:10],
        )

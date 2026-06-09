from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.preprocessing.corpus import (  # noqa: E402
    DOC_FILE_KEY,
    TITLE_KEY,
    build_document_text,
    deduplicate_chunk_records,
    extract_document_metadata,
    iter_document_pairs,
)
from app.retrieval.keywords import (  # noqa: E402
    ENTERPRISE_SIZE_KEYWORDS,
    INDUSTRY_KEYWORDS,
    LEGAL_ROLE_KEYWORDS,
    THEME_KEYWORDS,
)
from app.retrieval.llm_router import LangChainRouteTagger  # noqa: E402
from app.retrieval.router_names import normalize_router_backend_name  # noqa: E402
from app.retrieval.route_tags import normalize_question_key  # noqa: E402
from app.retrieval.router import OTHER, QueryRouter  # noqa: E402
from app.utils.schemas import LLMRouteDecision, RouteDecision  # noqa: E402


FOCUS_LABELS = {"처분", "위법성", "사실관계", "일반"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM/rule 기반 라우팅 태그 JSON을 생성합니다.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--router-backend",
        choices=["keyword", "lcel", "lcel_prompt_boost", "rule", "llm"],
        default="lcel",
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-document-chars", type=int, default=2500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="문서 태그만 생성하고 질문 태그는 출력 파일에서 비웁니다.",
    )
    parser.add_argument(
        "--keep-fallback-cache",
        action="store_true",
        help="LLM 백엔드에서도 기존 fallback 태그를 재처리하지 않고 그대로 사용합니다.",
    )
    return parser.parse_args()


def allowed_labels(keyword_map: dict[str, list[str]]) -> set[str]:
    return set(keyword_map) | {OTHER}


def normalize_label(value: str, allowed: set[str], default: str = OTHER) -> str:
    return value if value in allowed else default


def normalize_focus(value: str) -> str:
    return value if value in FOCUS_LABELS else "일반"


def fallback_payload(route: RouteDecision, *, source: str = "rule_fallback", reason: str = "") -> dict[str, Any]:
    return {
        "theme": route.theme,
        "company_size": route.company_size,
        "legal_role": route.legal_role,
        "industry": route.industry,
        "focus": route.focus,
        "confidence": 0.0,
        "reason": reason,
        "source": source,
        "fallback": True,
    }


def llm_payload(route: LLMRouteDecision, *, source: str = "llm") -> dict[str, Any]:
    return {
        "theme": normalize_label(route.theme, allowed_labels(THEME_KEYWORDS)),
        "company_size": normalize_label(route.company_size, allowed_labels(ENTERPRISE_SIZE_KEYWORDS)),
        "legal_role": normalize_label(route.legal_role, allowed_labels(LEGAL_ROLE_KEYWORDS)),
        "industry": normalize_label(route.industry, allowed_labels(INDUSTRY_KEYWORDS)),
        "focus": normalize_focus(route.focus),
        "confidence": float(route.confidence),
        "reason": route.reason,
        "source": source,
        "fallback": False,
    }


def load_existing_output(path: Path, *, force: bool) -> dict[str, Any]:
    if force or not path.exists():
        return {"documents": {}, "questions": {}, "summary": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("documents", {})
    payload.setdefault("questions", {})
    payload.setdefault("summary", {})
    return payload


def should_process_document(
    item: dict[str, str],
    output: dict[str, Any],
    *,
    force: bool,
    retry_fallbacks: bool,
) -> bool:
    if force:
        return True
    payload = output["documents"].get(item["id"])
    if not isinstance(payload, dict):
        return True
    return retry_fallbacks and bool(payload.get("fallback"))


def should_process_question(
    item: dict[str, str],
    output: dict[str, Any],
    *,
    force: bool,
    retry_fallbacks: bool,
) -> bool:
    if force:
        return True
    payload = output["questions"].get(item["id"])
    if not isinstance(payload, dict):
        return True
    route = payload.get("route")
    if not isinstance(route, dict):
        return True
    return retry_fallbacks and bool(route.get("fallback"))


def load_document_inputs(data_dir: Path, *, max_document_chars: int) -> list[dict[str, str]]:
    inputs: list[dict[str, str]] = []
    for metadata_path, hybrid_path in iter_document_pairs(data_dir):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        hybrid_records = json.loads(hybrid_path.read_text(encoding="utf-8"))
        cleaned_records = deduplicate_chunk_records(hybrid_records)
        doc_name = str(metadata.get(TITLE_KEY, hybrid_path.name.replace("_hybrid.json", ""))).strip()
        doc_id = str(metadata.get(DOC_FILE_KEY, doc_name)).strip() or doc_name
        company_names, violation_types = extract_document_metadata(metadata)
        preview_text = " ".join(record.get("page_content", "") for record in cleaned_records[:8]).strip()
        full_text = build_document_text(doc_name, company_names, violation_types, preview_text)
        inputs.append(
            {
                "id": doc_id,
                "input_kind": "document",
                "text": full_text[:max_document_chars],
                "fallback_text": full_text,
            }
        )
    return inputs


def get_prompt_variant(router_backend: str) -> str:
    normalized = normalize_router_backend_name(router_backend)
    if normalized == "lcel_prompt_boost":
        return "lcel_prompt_boost"
    return "lcel"


def load_question_inputs(eval_file: Path) -> list[dict[str, str]]:
    examples = json.loads(eval_file.read_text(encoding="utf-8"))
    inputs: list[dict[str, str]] = []
    for index, example in enumerate(examples):
        question = str(example.get("query") or example.get("question") or "").strip()
        if not question:
            continue
        question_id = str(example.get("id") or f"Q-{index:04d}")
        inputs.append(
            {
                "id": question_id,
                "input_kind": "question",
                "text": question,
                "fallback_text": question,
                "question_key": normalize_question_key(question),
            }
        )
    return inputs


def tag_with_rule(inputs: list[dict[str, str]], router: QueryRouter) -> list[dict[str, Any]]:
    return [
        fallback_payload(
            router.route_from_text(item["fallback_text"]),
            source="rule",
        )
        for item in inputs
    ]


def tag_with_llm(
    inputs: list[dict[str, str]],
    *,
    tagger: LangChainRouteTagger,
    router: QueryRouter,
    source: str,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    if not inputs:
        return []
    raw_results = tagger.tag_batch(
        [
            {
                "input_kind": item["input_kind"],
                "text": item["text"],
            }
            for item in inputs
        ],
        max_concurrency=max_concurrency,
    )
    payloads: list[dict[str, Any]] = []
    for item, result in zip(inputs, raw_results):
        if isinstance(result, LLMRouteDecision):
            payloads.append(llm_payload(result, source=source))
            continue
        reason = f"LLM route failed: {result!r}"
        payloads.append(
            fallback_payload(
                router.route_from_text(item["fallback_text"]),
                reason=reason,
            )
        )
    return payloads


def attach_document_routes(output: dict[str, Any], inputs: list[dict[str, str]], payloads: list[dict[str, Any]]) -> None:
    documents = output.setdefault("documents", {})
    for item, payload in zip(inputs, payloads):
        documents[item["id"]] = payload


def attach_question_routes(output: dict[str, Any], inputs: list[dict[str, str]], payloads: list[dict[str, Any]]) -> None:
    questions = output.setdefault("questions", {})
    for item, payload in zip(inputs, payloads):
        questions[item["id"]] = {
            "question": item["fallback_text"],
            "question_key": item["question_key"],
            "route": payload,
        }


def update_summary(output: dict[str, Any]) -> None:
    documents = output.get("documents", {})
    questions = output.get("questions", {})
    document_fallbacks = sum(1 for value in documents.values() if value.get("fallback"))
    question_fallbacks = sum(1 for value in questions.values() if value.get("route", {}).get("fallback"))
    output["summary"] = {
        "document_count": len(documents),
        "question_count": len(questions),
        "document_fallback_count": document_fallbacks,
        "question_fallback_count": question_fallbacks,
    }


def main() -> None:
    args = parse_args()
    args.router_backend = normalize_router_backend_name(args.router_backend)
    router = QueryRouter()
    output = load_existing_output(args.output_file, force=args.force)
    if args.documents_only:
        output["questions"] = {}
    retry_fallbacks = args.router_backend != "keyword" and not args.keep_fallback_cache

    document_inputs = [
        item
        for item in load_document_inputs(args.data_dir, max_document_chars=args.max_document_chars)
        if should_process_document(
            item,
            output,
            force=args.force,
            retry_fallbacks=retry_fallbacks,
        )
    ]
    if args.documents_only:
        question_inputs = []
    else:
        question_inputs = [
            item
            for item in load_question_inputs(args.eval_file)
            if should_process_question(
                item,
                output,
                force=args.force,
                retry_fallbacks=retry_fallbacks,
            )
        ]

    print(
        "[build_route_tags] pending "
        f"documents={len(document_inputs)} questions={len(question_inputs)} "
        f"backend={args.router_backend} retry_fallbacks={retry_fallbacks}"
    )

    if args.router_backend == "keyword":
        document_payloads = tag_with_rule(document_inputs, router)
        question_payloads = tag_with_rule(question_inputs, router)
    elif not document_inputs and not question_inputs:
        document_payloads = []
        question_payloads = []
    else:
        tagger = LangChainRouteTagger(prompt_variant=get_prompt_variant(args.router_backend))
        document_payloads = tag_with_llm(
            document_inputs,
            tagger=tagger,
            router=router,
            source=args.router_backend,
            max_concurrency=args.max_concurrency,
        )
        question_payloads = tag_with_llm(
            question_inputs,
            tagger=tagger,
            router=router,
            source=args.router_backend,
            max_concurrency=args.max_concurrency,
        )

    attach_document_routes(output, document_inputs, document_payloads)
    attach_question_routes(output, question_inputs, question_payloads)
    update_summary(output)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[build_route_tags] wrote {args.output_file}")
    print(f"[build_route_tags] summary={json.dumps(output['summary'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()

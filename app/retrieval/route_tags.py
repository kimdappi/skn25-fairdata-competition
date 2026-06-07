from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.retrieval.router import OTHER, QueryRouter
from app.utils.schemas import RouteDecision
from app.utils.text import tokenize_text


QUESTION_SPACE_PATTERN = re.compile(r"\s+")


def normalize_question_key(question: str) -> str:
    normalized = QUESTION_SPACE_PATTERN.sub(" ", question).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _route_payload(payload: dict[str, Any]) -> dict[str, Any]:
    route = payload.get("route")
    if isinstance(route, dict):
        return route
    return payload


def route_decision_from_payload(payload: dict[str, Any], *, source_text: str = "") -> RouteDecision:
    route = _route_payload(payload)
    return RouteDecision(
        theme=str(route.get("theme") or OTHER),
        company_size=str(route.get("company_size") or OTHER),
        legal_role=str(route.get("legal_role") or OTHER),
        industry=str(route.get("industry") or OTHER),
        focus=str(route.get("focus") or "일반"),
        keywords=tokenize_text(source_text)[:20],
    )


@dataclass
class RouteTagStore:
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    questions_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    questions_by_key: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> "RouteTagStore":
        if path is None or not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions_by_id = payload.get("questions", {}) or {}
        questions_by_key: dict[str, dict[str, Any]] = {}
        for question_id, entry in questions_by_id.items():
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("question_key") or "")
            question = str(entry.get("question") or "")
            if not key and question:
                key = normalize_question_key(question)
            if key:
                questions_by_key[key] = entry
            questions_by_key[str(question_id)] = entry
        return cls(
            documents=payload.get("documents", {}) or {},
            questions_by_id=questions_by_id,
            questions_by_key=questions_by_key,
        )

    def route_document(
        self,
        doc_id: str,
        text: str,
        fallback: Callable[[str], RouteDecision],
    ) -> RouteDecision:
        payload = self.documents.get(doc_id)
        if isinstance(payload, dict):
            return route_decision_from_payload(payload, source_text=text)
        return fallback(text)

    def route_question(
        self,
        question: str,
        fallback: Callable[[str], RouteDecision],
    ) -> RouteDecision:
        payload = self.questions_by_key.get(normalize_question_key(question))
        if isinstance(payload, dict):
            return route_decision_from_payload(payload, source_text=question)
        return fallback(question)


class CachedQuestionRouter:
    def __init__(self, fallback: QueryRouter, tag_store: RouteTagStore) -> None:
        self.fallback = fallback
        self.tag_store = tag_store

    def route_from_text(self, text: str) -> RouteDecision:
        return self.tag_store.route_question(text, self.fallback.route_from_text)

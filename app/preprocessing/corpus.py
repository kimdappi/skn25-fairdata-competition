from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.utils.schemas import RouteDecision
from app.utils.text import normalize_name, tokenize_text


@dataclass(frozen=True)
class Chunk:
    # 검색과 생성이 함께 참조하는 최소 청크 단위입니다.
    chunk_id: str
    doc_id: str
    doc_name: str
    normalized_doc_name: str
    header: str
    section: str
    content: str
    enriched_text: str
    route: RouteDecision
    company_names: tuple[str, ...]
    violation_types: tuple[str, ...]
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class Document:
    # 문서 단위 메타데이터와 대표 텍스트를 보관하는 객체입니다.
    doc_id: str
    doc_name: str
    normalized_doc_name: str
    company_names: tuple[str, ...]
    violation_types: tuple[str, ...]
    route: RouteDecision
    full_text: str
    tokens: tuple[str, ...]


@dataclass
class CorpusStore:
    # 문서/청크 전체와 빠른 조회용 맵을 함께 들고 있는 코퍼스 저장소입니다.
    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    document_map: dict[str, Document] = field(default_factory=dict)
    chunk_map: dict[str, Chunk] = field(default_factory=dict)
    document_to_chunk_ids: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


TITLE_KEY = "의결서제목"
DEFENDANT_INFO_KEY = "피심인정보"
COMPANY_NAME_KEY = "피심인기업명"
VIOLATION_KEY = "위반유형"
DETAIL_VIOLATION_KEY = "세부위반유형"
DOC_FILE_KEY = "의결서파일명"


# 하이브리드 JSON 파일만 안정적으로 순회하기 위해 파일 목록을 정렬합니다.
def iter_hybrid_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*_hybrid.json"))


# 메타데이터 JSON과 하이브리드 JSON을 문서 단위로 짝지어 반환합니다.
def iter_document_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    # 메타데이터 파일이 있으면 metadata-hybrid 쌍을 사용하고,
    # 없으면 hybrid 파일 하나만으로도 로딩이 가능하게 fallback 합니다.
    metadata_files = sorted(data_dir.glob("*_metadata.json"))
    pairs: list[tuple[Path, Path]] = []
    for metadata_path in metadata_files:
        hybrid_path = data_dir / metadata_path.name.replace("_metadata.json", "_hybrid.json")
        if hybrid_path.exists():
            pairs.append((metadata_path, hybrid_path))
    if pairs:
        return pairs
    return [(path, path) for path in iter_hybrid_files(data_dir)]


# 피심인 메타데이터에서 회사명과 위반유형 목록을 추출합니다.
def extract_document_metadata(meta: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # 피심인 목록에서 회사명과 위반유형을 펼쳐 문서 대표 메타데이터로 만듭니다.
    defendant_info = meta.get(DEFENDANT_INFO_KEY, []) or []
    company_names = tuple(
        item.get(COMPANY_NAME_KEY, "").strip()
        for item in defendant_info
        if item.get(COMPANY_NAME_KEY, "").strip()
    )
    violation_types = tuple(
        value.strip()
        for item in defendant_info
        for value in (
            item.get(VIOLATION_KEY, ""),
            item.get(DETAIL_VIOLATION_KEY, ""),
        )
        if value.strip()
    )
    return company_names, violation_types


# 문서 대표 텍스트를 구성해 라우팅과 검색용 공통 입력으로 사용합니다.
def build_document_text(
    doc_name: str,
    company_names: tuple[str, ...],
    violation_types: tuple[str, ...],
    preview_text: str,
) -> str:
    # 라우팅과 문서 수준 검색 보조 신호에 사용할 대표 문자열을 만듭니다.
    parts = [doc_name, *company_names, *violation_types, preview_text]
    return " ".join(part for part in parts if part).strip()


# 중복 청크를 줄이기 위해 문서 내부에서 동일 본문은 한 번만 유지합니다.
def deduplicate_chunk_records(records: list[dict]) -> list[dict]:
    # 동일 문서 안에서 사실상 같은 본문이 반복되면 첫 번째만 남깁니다.
    deduplicated: list[dict] = []
    seen_texts: set[str] = set()
    for record in records:
        content = str(record.get("page_content", "")).strip()
        if not content:
            continue
        content_key = normalize_name(content[:500])
        if content_key in seen_texts:
            continue
        seen_texts.add(content_key)
        deduplicated.append(record)
    return deduplicated


# 문서 하나를 검색 가능한 Document 객체로 변환합니다.
def build_document(
    *,
    doc_id: str,
    doc_name: str,
    company_names: tuple[str, ...],
    violation_types: tuple[str, ...],
    route: RouteDecision,
    full_text: str,
) -> Document:
    # 로딩 단계에서 문서명 정규화와 토큰화까지 끝낸 Document 객체를 생성합니다.
    return Document(
        doc_id=doc_id,
        doc_name=doc_name,
        normalized_doc_name=normalize_name(doc_name),
        company_names=company_names,
        violation_types=violation_types,
        route=route,
        full_text=full_text,
        tokens=tuple(tokenize_text(full_text)),
    )


# 문서 청크 하나를 검색 가능한 Chunk 객체로 변환합니다.
def build_chunk(
    *,
    doc_id: str,
    doc_name: str,
    company_names: tuple[str, ...],
    violation_types: tuple[str, ...],
    route: RouteDecision,
    record: dict,
) -> Chunk | None:
    # chunk_id 또는 본문이 비어 있으면 검색 후보로 쓸 수 없으므로 건너뜁니다.
    metadata = record.get("metadata", {})
    chunk_id = str(metadata.get("chunk_id", "")).strip()
    content = str(record.get("page_content", "")).strip()
    if not chunk_id or not content:
        return None

    header = str(metadata.get("Header", "")).strip()
    section = str(metadata.get("section", "")).strip()
    enriched_text = " ".join(
        part
        for part in (
            doc_name,
            " ".join(company_names),
            " ".join(violation_types),
            route.theme,
            route.company_size,
            route.legal_role,
            route.industry,
            route.focus,
            header,
            section,
            content,
        )
        if part
    ).strip()
    # enriched_text에는 문서명, 위반유형, 라우팅 메타데이터를 함께 넣어
    # dense/sparse 검색에서 본문 외 신호도 활용할 수 있게 합니다.
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name=doc_name,
        normalized_doc_name=normalize_name(doc_name),
        header=header,
        section=section,
        content=content,
        enriched_text=enriched_text,
        route=route,
        company_names=company_names,
        violation_types=violation_types,
        tokens=tuple(tokenize_text(enriched_text)),
    )


# 코퍼스 디렉터리를 읽어 문서와 청크 저장소를 구성합니다.
def load_corpus(data_dir: Path, route_text_fn: Callable[[str], RouteDecision]) -> CorpusStore:
    # raw 코퍼스를 순회하면서 문서/청크 객체와 조회용 맵을 한 번에 구성합니다.
    store = CorpusStore()
    for metadata_path, hybrid_path in iter_document_pairs(data_dir):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        hybrid_records = json.loads(hybrid_path.read_text(encoding="utf-8"))
        cleaned_records = deduplicate_chunk_records(hybrid_records)

        doc_name = str(metadata.get(TITLE_KEY, hybrid_path.name.replace("_hybrid.json", ""))).strip()
        doc_id = str(metadata.get(DOC_FILE_KEY, doc_name)).strip() or doc_name
        company_names, violation_types = extract_document_metadata(metadata)
        preview_text = " ".join(record.get("page_content", "") for record in cleaned_records[:8]).strip()
        full_text = build_document_text(doc_name, company_names, violation_types, preview_text)
        route = route_text_fn(full_text)

        document = build_document(
            doc_id=doc_id,
            doc_name=doc_name,
            company_names=company_names,
            violation_types=violation_types,
            route=route,
            full_text=full_text,
        )
        store.documents.append(document)
        store.document_map[document.doc_id] = document

        for record in cleaned_records:
            chunk = build_chunk(
                doc_id=document.doc_id,
                doc_name=document.doc_name,
                company_names=company_names,
                violation_types=violation_types,
                route=route,
                record=record,
            )
            if chunk is None:
                continue
            store.chunks.append(chunk)
            store.chunk_map[chunk.chunk_id] = chunk
            store.document_to_chunk_ids[document.doc_id].append(chunk.chunk_id)
    return store

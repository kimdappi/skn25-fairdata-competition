import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

from baseline_rag.retrieval_types import ChunkRecord, DocumentRecord
from baseline_rag.retrieval_utils import normalize_name
from baseline_rag.schemas import RouteDecision

TITLE_KEY = "\uc758\uacb0\uc11c\uc81c\ubaa9"
DEFENDANT_INFO_KEY = "\ud53c\uc2ec\uc778\uc815\ubcf4"
COMPANY_NAME_KEY = "\ud53c\uc2ec\uc778\uae30\uc5c5\uba85"
VIOLATION_KEY = "\uc704\ubc18\uc720\ud615"
DETAIL_VIOLATION_KEY = "\uc138\ubd80\uc704\ubc18\uc720\ud615"
DISPOSITION_KEY = "\uc870\uce58\uc720\ud615"
DOC_FILE_KEY = "\uc758\uacb0\uc11c\ud30c\uc77c\uba85"


@dataclass
class CorpusStore:
    documents: List[DocumentRecord] = field(default_factory=list)
    chunks: List[ChunkRecord] = field(default_factory=list)
    doc_map: Dict[str, DocumentRecord] = field(default_factory=dict)
    chunk_map: Dict[str, ChunkRecord] = field(default_factory=dict)
    chunk_idx_map: Dict[str, int] = field(default_factory=dict)
    doc_to_chunk_ids: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))


def load_corpus(data_dir: Path, route_text_fn: Callable[[str], RouteDecision]) -> CorpusStore:
    store = CorpusStore()
    meta_files = sorted(name for name in os.listdir(data_dir) if name.endswith("_metadata.json"))
    for meta_name in meta_files:
        meta_path = data_dir / meta_name
        hybrid_name = meta_name.replace("_metadata.json", "_hybrid.json")
        hybrid_path = data_dir / hybrid_name
        if not hybrid_path.exists():
            continue

        with open(meta_path, "r", encoding="utf-8") as file:
            meta = json.load(file)
        with open(hybrid_path, "r", encoding="utf-8") as file:
            hybrid = json.load(file)

        title = meta.get(TITLE_KEY, meta_name.replace("_metadata.json", ""))
        defendant_info = meta.get(DEFENDANT_INFO_KEY, []) or []
        company_names = [item.get(COMPANY_NAME_KEY, "") for item in defendant_info if item.get(COMPANY_NAME_KEY)]
        violation_types = [item.get(VIOLATION_KEY, "") for item in defendant_info if item.get(VIOLATION_KEY)]
        detail_types = [item.get(DETAIL_VIOLATION_KEY, "") for item in defendant_info if item.get(DETAIL_VIOLATION_KEY)]
        disposition_types = [item.get(DISPOSITION_KEY, "") for item in defendant_info if item.get(DISPOSITION_KEY)]
        preview_text = " ".join(chunk.get("page_content", "") for chunk in hybrid[:8])
        route_seed = " ".join([title, *company_names, *violation_types, *detail_types, preview_text[:2500]])
        route = route_text_fn(route_seed)
        doc_id = meta.get(DOC_FILE_KEY, title)
        doc_text = " ".join(
            [
                title,
                " ".join(company_names),
                " ".join(violation_types),
                " ".join(detail_types),
                " ".join(disposition_types),
                route.theme,
                route.company_size,
                route.legal_role,
                route.industry,
                preview_text,
            ]
        )
        document = DocumentRecord(
            doc_id=doc_id,
            title=title,
            normalized_title=normalize_name(title),
            company_names=company_names,
            normalized_company_names=[normalize_name(name) for name in company_names if name],
            doc_text=doc_text,
            route_theme=route.theme,
            route_company_size=route.company_size,
            route_legal_role=route.legal_role,
            route_industry=route.industry,
        )
        store.documents.append(document)
        store.doc_map[doc_id] = document

        for item in hybrid:
            metadata = item.get("metadata", {})
            page_content = item.get("page_content", "")
            header = metadata.get("Header") or metadata.get("section") or ""
            chunk_id = metadata.get("chunk_id", "")
            chunk_index = int(metadata.get("chunk_index", 0) or 0)
            enriched_text = " ".join(
                [
                    title,
                    " ".join(company_names),
                    " ".join(violation_types),
                    " ".join(detail_types),
                    route.theme,
                    route.legal_role,
                    route.industry,
                    header,
                    page_content,
                ]
            )
            chunk = ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc_id,
                title=title,
                header=header,
                chunk_index=chunk_index,
                page_content=page_content,
                enriched_text=enriched_text,
                normalized_title=normalize_name(title),
                route_theme=route.theme,
                route_company_size=route.company_size,
                route_legal_role=route.legal_role,
                route_industry=route.industry,
            )
            store.chunk_idx_map[chunk_id] = len(store.chunks)
            store.chunks.append(chunk)
            store.chunk_map[chunk_id] = chunk
            store.doc_to_chunk_ids[doc_id].append(chunk_id)
    return store

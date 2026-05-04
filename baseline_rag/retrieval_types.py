from dataclasses import dataclass
from typing import Annotated, Dict, List, TypedDict

from baseline_rag.schemas import RetrievedChunk, RouteDecision


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    normalized_title: str
    company_names: List[str]
    normalized_company_names: List[str]
    doc_text: str
    route_theme: str
    route_company_size: str
    route_legal_role: str
    route_industry: str


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    title: str
    header: str
    chunk_index: int
    page_content: str
    enriched_text: str
    normalized_title: str
    route_theme: str
    route_company_size: str
    route_legal_role: str
    route_industry: str


class RetrievalState(TypedDict, total=False):
    query: str
    route: RouteDecision
    candidate_doc_ids: Annotated[List[str], "candidate document ids"]
    results: Annotated[List[RetrievedChunk], "top5 chunks"]
    timings: Dict[str, float]

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from langgraph.graph import END, START, StateGraph
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix, hstack, load_npz, save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# This project uses sentence-transformers with PyTorch only.
# Disabling TF import avoids the Keras 3 compatibility error in transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from baseline_rag.config import (
    BGE_M3_CHUNK_COLBERT_RERANK_K,
    BGE_M3_CHUNK_COLBERT_SCORE_WEIGHT,
    BGE_M3_CHUNK_RRF_SPARSE_WEIGHT,
    BGE_M3_DOC_COLBERT_RERANK_K,
    BGE_M3_DOC_COLBERT_SCORE_WEIGHT,
    BGE_M3_DOC_RRF_SPARSE_WEIGHT,
    BGE_M3_EAGER_SPARSE_CACHE_BUILD,
    BGE_M3_USE_COLBERT,
    BGE_M3_USE_SPARSE,
    CHUNK_CHAR_MAX_FEATURES,
    CHUNK_FAISS_K,
    CHUNK_RRF_BM25_WEIGHT,
    CHUNK_RRF_DENSE_WEIGHT,
    CHUNK_SVD_COMPONENTS,
    CHUNK_WORD_MAX_FEATURES,
    DATA_DIR,
    DOC_CHAR_MAX_FEATURES,
    DOC_FAISS_K,
    DOC_RRF_BM25_WEIGHT,
    DOC_RRF_DENSE_WEIGHT,
    DOC_SVD_COMPONENTS,
    DOC_WORD_MAX_FEATURES,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_CACHE_DIR,
    MODEL_DIR,
    QUERY_TIMING_LOG_ENABLED,
    RRF_K,
    STAGE_LOG_ENABLED,
    TOP_K_CHUNKS,
    TOP_K_DOCS,
    validate_runtime_paths,
)
from baseline_rag.bge_m3 import BGEM3HybridModel
from baseline_rag.corpus import load_corpus
from baseline_rag.embeddings import LocalEmbeddingModel
from baseline_rag.experiments import RetrieverOptions
from baseline_rag.retrieval_types import ChunkRecord, DocumentRecord, RetrievalState
from baseline_rag.retrieval_utils import (
    FOCUS_GENERAL,
    FOCUS_FACT,
    FOCUS_LAW,
    FOCUS_ORDER,
    HEADER_FACT_WORDS,
    HEADER_LAW_WORDS,
    HEADER_ORDER,
    OTHER,
    min_max_scale,
    normalize_name,
    ranked_indices_from_scores,
    reciprocal_rank_fusion,
    tokenize_text,
)
from baseline_rag.router import QueryRouter
from baseline_rag.schemas import QueryInput, RetrievedChunk, RouteDecision


class LegalRAGRetriever:
    def __init__(
        self,
        embedding_model_name: Optional[str] = DEFAULT_EMBEDDING_MODEL,
        router: Optional[QueryRouter] = None,
        options: Optional[RetrieverOptions] = None,
    ) -> None:
        validate_runtime_paths()
        self.embedding_model_name = embedding_model_name or DEFAULT_EMBEDDING_MODEL
        self.options = (options or RetrieverOptions()).normalized(self.embedding_model_name)
        self.embedding_model: Optional[object] = None
        self.stage_log_enabled = os.getenv("STAGE_LOG_ENABLED", str(STAGE_LOG_ENABLED)).lower() in {"1", "true", "yes", "on"}
        self.query_timing_log_enabled = os.getenv(
            "QUERY_TIMING_LOG_ENABLED",
            str(QUERY_TIMING_LOG_ENABLED),
        ).lower() in {"1", "true", "yes", "on"}
        self.bge_m3_use_sparse = os.getenv("BGE_M3_USE_SPARSE", str(BGE_M3_USE_SPARSE)).lower() in {"1", "true", "yes", "on"}
        self.bge_m3_use_colbert = os.getenv("BGE_M3_USE_COLBERT", str(BGE_M3_USE_COLBERT)).lower() in {"1", "true", "yes", "on"}
        self.bge_m3_eager_sparse_cache_build = os.getenv(
            "BGE_M3_EAGER_SPARSE_CACHE_BUILD",
            str(BGE_M3_EAGER_SPARSE_CACHE_BUILD),
        ).lower() in {"1", "true", "yes", "on"}

        default_router = QueryRouter(enabled=self.options.router_mode == "ollama")
        self.router = router or default_router
        self.ollama_router_enabled = self.router.enabled
        self.ollama_router_model = self.router.model_name
        self.ollama_router_url = self.router.url
        self.ollama_router_timeout = self.router.timeout_seconds

        self.documents: List[DocumentRecord] = []
        self.chunks: List[ChunkRecord] = []
        self.doc_map: Dict[str, DocumentRecord] = {}
        self.chunk_map: Dict[str, ChunkRecord] = {}
        self.chunk_idx_map: Dict[str, int] = {}
        self.doc_to_chunk_ids: Dict[str, List[str]] = {}
        self.doc_title_tokens: List[set[str]] = []
        self.chunk_query_tokens: List[set[str]] = []

        init_started = time.perf_counter()
        self._init_embedding_model()
        self._load_corpus()
        self._build_doc_index()
        self._build_chunk_index()
        self.graph = self._build_graph()
        self._log_stage(
            "init.total",
            init_started,
            extra=f"documents={len(self.documents)} chunks={len(self.chunks)}",
        )

    def _init_embedding_model(self) -> None:
        if not self.embedding_model_name:
            return
        started_at = time.perf_counter()
        model_dir = MODEL_DIR / self.embedding_model_name
        if self.embedding_model_name == "embedding_bge_m3":
            self.embedding_model = BGEM3HybridModel(model_dir)
        else:
            self.embedding_model = LocalEmbeddingModel(model_dir)
        self._log_stage("init.embedding_model", started_at, extra=f"name={self.embedding_model_name}")

    @property
    def is_bge_m3(self) -> bool:
        return isinstance(self.embedding_model, BGEM3HybridModel)

    def tokenize(self, text: str) -> List[str]:
        return tokenize_text(text)

    def _log_stage(self, stage: str, started_at: float, *, extra: Optional[str] = None) -> None:
        if not self.stage_log_enabled:
            return
        elapsed = time.perf_counter() - started_at
        suffix = f" {extra}" if extra else ""
        print(f"[stage] {stage} {elapsed:.3f}s{suffix}")

    def route_queries(self, queries: List[str], batch_size: int = 12) -> List[RouteDecision]:
        return self.router.route_many(queries, batch_size=batch_size)

    def _load_corpus(self) -> None:
        started_at = time.perf_counter()
        corpus = load_corpus(DATA_DIR, self.router.route_from_text)
        self.documents = corpus.documents
        self.chunks = corpus.chunks
        self.doc_map = corpus.doc_map
        self.chunk_map = corpus.chunk_map
        self.chunk_idx_map = corpus.chunk_idx_map
        self.doc_to_chunk_ids = corpus.doc_to_chunk_ids
        self._log_stage("init.load_corpus", started_at, extra=f"documents={len(self.documents)} chunks={len(self.chunks)}")

    def _build_doc_index(self) -> None:
        started_at = time.perf_counter()
        doc_texts = [doc.doc_text for doc in self.documents]
        self.doc_title_tokens = [set(self.tokenize(doc.title)) for doc in self.documents]
        self.doc_dense = self._build_dense_matrix(
            texts=doc_texts,
            cache_name="docs",
            word_max_features=DOC_WORD_MAX_FEATURES,
            char_max_features=DOC_CHAR_MAX_FEATURES,
            svd_components=DOC_SVD_COMPONENTS,
            is_query_encoder=False,
            scope="doc",
        )
        self.doc_bm25 = BM25Okapi([self.tokenize(text) for text in doc_texts])
        self.doc_bge_sparse = self._init_bge_sparse_cache("docs", doc_texts) if self.is_bge_m3 and self.bge_m3_use_sparse else None
        self.doc_index = faiss.IndexFlatIP(self.doc_dense.shape[1])
        self.doc_index.add(self.doc_dense)
        self._log_stage("init.build_doc_index", started_at, extra=f"documents={len(self.documents)}")

    def _build_chunk_index(self) -> None:
        started_at = time.perf_counter()
        chunk_texts = [chunk.enriched_text for chunk in self.chunks]
        self.chunk_word_vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            max_features=CHUNK_WORD_MAX_FEATURES,
        )
        self.chunk_char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=CHUNK_CHAR_MAX_FEATURES,
        )
        word_matrix = self.chunk_word_vectorizer.fit_transform(chunk_texts)
        char_matrix = self.chunk_char_vectorizer.fit_transform(chunk_texts)
        self.chunk_sparse_matrix = hstack([word_matrix, char_matrix], format="csr", dtype=np.float32)
        self.chunk_bm25 = BM25Okapi([self.tokenize(text) for text in chunk_texts])
        self.chunk_dense = self._build_chunk_dense_matrix(chunk_texts)
        self.chunk_bge_sparse = self._init_bge_sparse_cache("chunks", chunk_texts) if self.is_bge_m3 and self.bge_m3_use_sparse else None
        self.chunk_query_tokens = [set(self.tokenize(chunk.header + " " + chunk.page_content[:250])) for chunk in self.chunks]
        self._log_stage("init.build_chunk_index", started_at, extra=f"chunks={len(self.chunks)}")

    def _build_dense_matrix(
        self,
        *,
        texts: List[str],
        cache_name: str,
        word_max_features: int,
        char_max_features: int,
        svd_components: int,
        is_query_encoder: bool,
        scope: str,
    ) -> np.ndarray:
        if self.embedding_model:
            return self._load_or_create_dense_cache(cache_name, texts, is_query=is_query_encoder)

        word_vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            max_features=word_max_features,
        )
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=char_max_features,
        )
        word_matrix = word_vectorizer.fit_transform(texts)
        char_matrix = char_vectorizer.fit_transform(texts)
        combined = hstack([word_matrix, char_matrix], format="csr", dtype=np.float32)
        n_components = max(8, min(svd_components, combined.shape[0] - 1, combined.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense = svd.fit_transform(combined).astype("float32")
        dense = normalize(dense, norm="l2")

        if scope == "doc":
            self.doc_word_vectorizer = word_vectorizer
            self.doc_char_vectorizer = char_vectorizer
            self.doc_svd = svd
        return dense

    def _build_chunk_dense_matrix(self, chunk_texts: List[str]) -> np.ndarray:
        if self.embedding_model:
            return self._load_or_create_dense_cache("chunks", chunk_texts, is_query=False)
        target_components = CHUNK_SVD_COMPONENTS or DOC_SVD_COMPONENTS
        n_components = max(
            8,
            min(target_components, self.chunk_sparse_matrix.shape[0] - 1, self.chunk_sparse_matrix.shape[1] - 1),
        )
        self.chunk_svd = TruncatedSVD(n_components=n_components, random_state=42)
        chunk_dense = self.chunk_svd.fit_transform(self.chunk_sparse_matrix).astype("float32")
        return normalize(chunk_dense, norm="l2")

    def _embed_doc_query(self, query: str) -> np.ndarray:
        if self.embedding_model:
            if self.is_bge_m3:
                return self.embedding_model.encode_dense([query])
            return self.embedding_model.encode([query], is_query=True)
        word = self.doc_word_vectorizer.transform([query])
        char = self.doc_char_vectorizer.transform([query])
        dense = self.doc_svd.transform(hstack([word, char], format="csr", dtype=np.float32)).astype("float32")
        return normalize(dense, norm="l2").astype("float32")

    def _embed_chunk_query(self, query: str):
        word = self.chunk_word_vectorizer.transform([query])
        char = self.chunk_char_vectorizer.transform([query])
        return hstack([word, char], format="csr", dtype=np.float32)

    def _embed_chunk_dense_query(self, query: str) -> np.ndarray:
        if self.embedding_model:
            if self.is_bge_m3:
                return self.embedding_model.encode_dense([query])
            return self.embedding_model.encode([query], is_query=True)
        sparse_query = self._embed_chunk_query(query)
        dense = self.chunk_svd.transform(sparse_query).astype("float32")
        return normalize(dense, norm="l2").astype("float32")

    def _cache_dir(self) -> Path:
        if not self.embedding_model_name:
            raise ValueError("Embedding cache requested without an embedding model.")
        cache_dir = EMBEDDING_CACHE_DIR / self.embedding_model_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _load_or_create_dense_cache(self, cache_name: str, texts: List[str], *, is_query: bool) -> np.ndarray:
        cache_path = self._cache_dir() / f"{cache_name}.npy"
        if cache_path.exists():
            started_at = time.perf_counter()
            values = np.load(cache_path)
            self._log_stage(f"cache.dense_load.{cache_name}", started_at, extra=f"path={cache_path.name} shape={values.shape}")
            return values
        started_at = time.perf_counter()
        if self.is_bge_m3:
            vectors = self.embedding_model.encode_dense(texts)
        else:
            vectors = self.embedding_model.encode(texts, is_query=is_query)
        np.save(cache_path, vectors)
        self._log_stage(f"cache.dense_build.{cache_name}", started_at, extra=f"path={cache_path.name} shape={vectors.shape}")
        return vectors

    def _bge_sparse_cache_path(self, cache_name: str) -> Path:
        return self._cache_dir() / f"{cache_name}_bge_sparse.npz"

    def _load_bge_sparse_cache_if_available(self, cache_name: str) -> Optional[csr_matrix]:
        cache_path = self._bge_sparse_cache_path(cache_name)
        if not cache_path.exists():
            return None
        started_at = time.perf_counter()
        values = load_npz(cache_path).tocsr()
        self._log_stage(f"cache.bge_sparse_load.{cache_name}", started_at, extra=f"path={cache_path.name} shape={values.shape}")
        return values

    def _load_or_create_bge_sparse_cache(self, cache_name: str, texts: List[str]) -> csr_matrix:
        cache_path = self._cache_dir() / f"{cache_name}_bge_sparse.npz"
        if cache_path.exists():
            started_at = time.perf_counter()
            values = load_npz(cache_path).tocsr()
            self._log_stage(f"cache.bge_sparse_load.{cache_name}", started_at, extra=f"path={cache_path.name} shape={values.shape}")
            return values
        started_at = time.perf_counter()
        values = self.embedding_model.encode_sparse(texts).tocsr()
        save_npz(cache_path, values)
        self._log_stage(f"cache.bge_sparse_build.{cache_name}", started_at, extra=f"path={cache_path.name} shape={values.shape}")
        return values

    def _init_bge_sparse_cache(self, cache_name: str, texts: List[str]) -> Optional[csr_matrix]:
        cached = self._load_bge_sparse_cache_if_available(cache_name)
        if cached is not None:
            return cached
        if self.bge_m3_eager_sparse_cache_build:
            return self._load_or_create_bge_sparse_cache(cache_name, texts)
        print(f"[stage] cache.bge_sparse_deferred.{cache_name} eager_build=False")
        return None

    def build_bge_m3_sparse_caches(self) -> None:
        if not self.is_bge_m3 or not self.bge_m3_use_sparse:
            return
        self.doc_bge_sparse = self._load_or_create_bge_sparse_cache("docs", [doc.doc_text for doc in self.documents])
        self.chunk_bge_sparse = self._load_or_create_bge_sparse_cache("chunks", [chunk.enriched_text for chunk in self.chunks])

    def route_query(self, state: RetrievalState) -> RetrievalState:
        started_at = time.perf_counter()
        query = QueryInput(query=state["query"]).query
        route = self._resolve_route(query)
        elapsed = time.perf_counter() - started_at
        self._log_stage("query.route", started_at, extra=f"query={query!r}")
        return {"route": route, "timings": {"route": elapsed}}

    def _resolve_route(self, query: str) -> RouteDecision:
        if not self.options.use_routing:
            return RouteDecision(
                theme=OTHER,
                company_size=OTHER,
                legal_role=OTHER,
                industry=OTHER,
                focus=FOCUS_GENERAL,
                keywords=self.tokenize(query)[:20],
            )
        if self.options.router_mode == "rule":
            return self.router.route_from_text(query)
        return self.router.route(query)

    def retrieve_documents(self, state: RetrievalState) -> RetrievalState:
        started_at = time.perf_counter()
        query = state["query"]
        route = state["route"]
        timings = dict(state.get("timings", {}))

        dense_ranking = self._search_dense_documents(query)
        bm25_ranking = self._search_bm25_documents(query)
        bge_sparse_ranking = self._search_bge_sparse_documents(query)
        candidate_indices = self._collect_document_candidate_indices(query, dense_ranking, bm25_ranking, bge_sparse_ranking)
        rrf_scores = reciprocal_rank_fusion(
            {
                "bge_sparse": bge_sparse_ranking,
                "bm25": bm25_ranking,
                "dense": dense_ranking,
            },
            {
                "bge_sparse": BGE_M3_DOC_RRF_SPARSE_WEIGHT,
                "bm25": DOC_RRF_BM25_WEIGHT,
                "dense": DOC_RRF_DENSE_WEIGHT,
            },
            k=RRF_K,
        )

        scored_docs = self._score_candidate_documents(
            query=query,
            route=route,
            candidate_indices=candidate_indices,
            rrf_scores=rrf_scores,
        )
        scored_docs = self._rerank_documents_with_bge_colbert(query, scored_docs)
        candidate_doc_ids = [doc_id for doc_id, _ in scored_docs[:TOP_K_DOCS]]
        timings["documents"] = time.perf_counter() - started_at
        self._log_stage("query.retrieve_documents", started_at, extra=f"query={query!r} candidates={len(candidate_doc_ids)}")
        return {"candidate_doc_ids": candidate_doc_ids, "timings": timings}

    def _search_dense_documents(self, query: str) -> List[int]:
        if not self.options.use_dense:
            return []
        query_vec = self._embed_doc_query(query)
        _, dense_indices = self.doc_index.search(query_vec, min(DOC_FAISS_K, len(self.documents)))
        return [int(idx) for idx in dense_indices[0].tolist()]

    def _search_bm25_documents(self, query: str) -> List[int]:
        if not self.options.use_bm25:
            return []
        bm25_scores = np.array(self.doc_bm25.get_scores(self.tokenize(query)), dtype="float32")
        return ranked_indices_from_scores(bm25_scores, min(DOC_FAISS_K, len(self.documents)))

    def _search_bge_sparse_documents(self, query: str) -> List[int]:
        if self.doc_bge_sparse is None or not self.is_bge_m3 or not self.options.use_bge_sparse:
            return []
        query_sparse = self.embedding_model.encode_sparse([query])
        scores = (self.doc_bge_sparse @ query_sparse.T).toarray().ravel().astype("float32")
        return ranked_indices_from_scores(scores, min(DOC_FAISS_K, len(self.documents)))

    def _collect_document_candidate_indices(
        self,
        query: str,
        dense_ranking: List[int],
        bm25_ranking: List[int],
        bge_sparse_ranking: List[int],
    ) -> List[int]:
        candidate_indices = set(dense_ranking) | set(bm25_ranking) | set(bge_sparse_ranking)
        if not self.options.use_entity_boost:
            return sorted(candidate_indices)
        query_tokens = set(self.tokenize(query))
        query_norm = normalize_name(query)
        entity_scores = []
        for idx, doc in enumerate(self.documents):
            score = 0.0
            if doc.normalized_title and (doc.normalized_title in query_norm or query_norm in doc.normalized_title):
                score += 2.5
            for company_norm in doc.normalized_company_names:
                if company_norm and company_norm in query_norm:
                    score += 3.0
                    break
            score += len(query_tokens & self.doc_title_tokens[idx]) * 0.15
            entity_scores.append((idx, score))
        entity_top_indices = [idx for idx, score in sorted(entity_scores, key=lambda item: item[1], reverse=True)[:40] if score > 0]
        candidate_indices |= set(entity_top_indices)
        return sorted(candidate_indices)

    def _score_candidate_documents(
        self,
        *,
        query: str,
        route: RouteDecision,
        candidate_indices: List[int],
        rrf_scores: Dict[int, float],
    ) -> List[tuple[str, float]]:
        query_tokens = set(self.tokenize(query))
        query_norm = normalize_name(query)
        scored_docs = []
        for idx in candidate_indices:
            doc = self.documents[idx]
            score = rrf_scores.get(idx, 0.0)
            if self.options.use_route_boost:
                score += self._document_route_bonus(doc, route)
            if self.options.use_entity_boost:
                overlap = len(query_tokens & self.doc_title_tokens[idx])
                score += min(0.30, overlap * 0.03)
                if doc.normalized_title and doc.normalized_title[:12] in query_norm:
                    score += 0.20
                for company_norm in doc.normalized_company_names:
                    if company_norm and company_norm in query_norm:
                        score += 0.45
                        break
            scored_docs.append((doc.doc_id, score))
        scored_docs.sort(key=lambda item: item[1], reverse=True)
        return scored_docs

    def _rerank_documents_with_bge_colbert(self, query: str, scored_docs: List[tuple[str, float]]) -> List[tuple[str, float]]:
        if not self.is_bge_m3 or not self.bge_m3_use_colbert or not self.options.use_bge_colbert or not scored_docs:
            return scored_docs
        rerank_count = min(BGE_M3_DOC_COLBERT_RERANK_K, len(scored_docs))
        query_vecs = self.embedding_model.encode_colbert([query])[0]
        candidate_doc_ids = [doc_id for doc_id, _ in scored_docs[:rerank_count]]
        doc_texts = [self.doc_map[doc_id].doc_text for doc_id in candidate_doc_ids]
        doc_vecs_list = self.embedding_model.encode_colbert(doc_texts)
        colbert_scores = np.array(
            [self.embedding_model.colbert_score(query_vecs, doc_vecs) for doc_vecs in doc_vecs_list],
            dtype="float32",
        )
        scaled = min_max_scale(colbert_scores)
        bonus_map = {
            doc_id: float(score) * BGE_M3_DOC_COLBERT_SCORE_WEIGHT
            for doc_id, score in zip(candidate_doc_ids, scaled)
        }
        reranked = [(doc_id, score + bonus_map.get(doc_id, 0.0)) for doc_id, score in scored_docs]
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked

    def _document_route_bonus(self, doc: DocumentRecord, route: RouteDecision) -> float:
        score = 0.0
        if route.theme != OTHER and doc.route_theme == route.theme:
            score += 0.03
        if route.legal_role != OTHER and doc.route_legal_role == route.legal_role:
            score += 0.02
        if route.industry != OTHER and doc.route_industry == route.industry:
            score += 0.015
        if route.company_size != OTHER and doc.route_company_size == route.company_size:
            score += 0.01
        return score

    def retrieve_chunks(self, state: RetrievalState) -> RetrievalState:
        started_at = time.perf_counter()
        query = state["query"]
        route = state["route"]
        timings = dict(state.get("timings", {}))
        candidate_doc_ids = state["candidate_doc_ids"][:TOP_K_DOCS]

        candidate_indices = self._collect_candidate_chunk_indices(candidate_doc_ids)
        if not candidate_indices:
            timings["chunks"] = time.perf_counter() - started_at
            return {"results": [], "timings": timings}

        sparse_query_vec = self._embed_chunk_query(query)
        dense_query_vec = self._embed_chunk_dense_query(query)
        query_tokens = self.tokenize(query)
        bm25_scores = (
            np.array(self.chunk_bm25.get_scores(query_tokens), dtype="float32")
            if self.options.use_bm25
            else np.zeros(len(self.chunks), dtype="float32")
        )
        candidate_matrix = self.chunk_sparse_matrix[candidate_indices]
        lexical_scores = (
            (candidate_matrix @ sparse_query_vec.T).toarray().ravel().astype("float32")
            if self.options.use_chunk_lexical_score
            else np.zeros(len(candidate_indices), dtype="float32")
        )
        dense_scores = (
            (self.chunk_dense[candidate_indices] @ dense_query_vec[0]).astype("float32")
            if self.options.use_dense
            else np.zeros(len(candidate_indices), dtype="float32")
        )
        bge_sparse_scores = self._score_bge_sparse_chunks(query, candidate_indices)

        dense_ranking = [candidate_indices[idx] for idx in ranked_indices_from_scores(dense_scores, min(CHUNK_FAISS_K, len(candidate_indices)))]
        bm25_ranking = [
            candidate_indices[idx]
            for idx in ranked_indices_from_scores(bm25_scores[candidate_indices], min(CHUNK_FAISS_K, len(candidate_indices)))
        ]
        bge_sparse_ranking = [
            candidate_indices[idx]
            for idx in ranked_indices_from_scores(bge_sparse_scores, min(CHUNK_FAISS_K, len(candidate_indices)))
        ]
        rrf_scores = reciprocal_rank_fusion(
            {
                "bge_sparse": bge_sparse_ranking,
                "bm25": bm25_ranking,
                "dense": dense_ranking,
            },
            {
                "bge_sparse": BGE_M3_CHUNK_RRF_SPARSE_WEIGHT,
                "bm25": CHUNK_RRF_BM25_WEIGHT,
                "dense": CHUNK_RRF_DENSE_WEIGHT,
            },
            k=RRF_K,
        )

        scored_chunks = self._score_candidate_chunks(
            query=query,
            route=route,
            candidate_doc_ids=candidate_doc_ids,
            candidate_indices=candidate_indices,
            lexical_scores=lexical_scores,
            rrf_scores=rrf_scores,
        )
        scored_chunks = self._rerank_chunks_with_bge_colbert(query, scored_chunks)
        top_chunks = self._select_top_chunks(scored_chunks)
        timings["chunks"] = time.perf_counter() - started_at
        self._log_stage("query.retrieve_chunks", started_at, extra=f"query={query!r} top_chunks={len(top_chunks)}")
        return {"results": top_chunks, "timings": timings}

    def _collect_candidate_chunk_indices(self, candidate_doc_ids: List[str]) -> List[int]:
        candidate_indices: List[int] = []
        for doc_id in candidate_doc_ids:
            for chunk_id in self.doc_to_chunk_ids.get(doc_id, []):
                candidate_indices.append(self.chunk_idx_map[chunk_id])
        return candidate_indices

    def _score_bge_sparse_chunks(self, query: str, candidate_indices: List[int]) -> np.ndarray | None:
        if self.chunk_bge_sparse is None or not self.is_bge_m3 or not self.options.use_bge_sparse:
            return None
        query_sparse = self.embedding_model.encode_sparse([query])
        candidate_sparse = self.chunk_bge_sparse[candidate_indices]
        return (candidate_sparse @ query_sparse.T).toarray().ravel().astype("float32")

    def _score_candidate_chunks(
        self,
        *,
        query: str,
        route: RouteDecision,
        candidate_doc_ids: List[str],
        candidate_indices: List[int],
        lexical_scores: np.ndarray,
        rrf_scores: Dict[int, float],
    ) -> List[tuple[ChunkRecord, float]]:
        lexical_scaled = min_max_scale(lexical_scores)
        query_norm = normalize_name(query)
        query_token_set = set(self.tokenize(query))
        doc_rank_map = {doc_id: rank for rank, doc_id in enumerate(candidate_doc_ids)}
        scored_chunks = []
        for local_idx, chunk_idx in enumerate(candidate_indices):
            chunk = self.chunks[chunk_idx]
            score = rrf_scores.get(chunk_idx, 0.0)
            score += min(0.12, float(lexical_scaled[local_idx]) * 0.12)
            if self.options.use_doc_rank_boost:
                score += max(0.0, 0.18 - doc_rank_map[chunk.doc_id] * 0.06)
            if self.options.use_route_boost:
                score += self._chunk_route_bonus(chunk, route)
            overlap = len(query_token_set & self.chunk_query_tokens[chunk_idx])
            score += min(0.18, overlap * 0.01)
            if self.options.use_chunk_structure_boost:
                score += self._chunk_focus_bonus(chunk, route)
                if chunk.normalized_title[:12] and chunk.normalized_title[:12] in query_norm:
                    score += 0.08
            scored_chunks.append((chunk, score))
        scored_chunks.sort(key=lambda item: item[1], reverse=True)
        return scored_chunks

    def _rerank_chunks_with_bge_colbert(
        self,
        query: str,
        scored_chunks: List[tuple[ChunkRecord, float]],
    ) -> List[tuple[ChunkRecord, float]]:
        if not self.is_bge_m3 or not self.bge_m3_use_colbert or not self.options.use_bge_colbert or not scored_chunks:
            return scored_chunks
        rerank_count = min(BGE_M3_CHUNK_COLBERT_RERANK_K, len(scored_chunks))
        query_vecs = self.embedding_model.encode_colbert([query])[0]
        candidate_chunks = [chunk for chunk, _ in scored_chunks[:rerank_count]]
        chunk_texts = [chunk.enriched_text for chunk in candidate_chunks]
        chunk_vecs_list = self.embedding_model.encode_colbert(chunk_texts)
        colbert_scores = np.array(
            [self.embedding_model.colbert_score(query_vecs, chunk_vecs) for chunk_vecs in chunk_vecs_list],
            dtype="float32",
        )
        scaled = min_max_scale(colbert_scores)
        bonus_map = {
            chunk.chunk_id: float(score) * BGE_M3_CHUNK_COLBERT_SCORE_WEIGHT
            for chunk, score in zip(candidate_chunks, scaled)
        }
        reranked = [(chunk, score + bonus_map.get(chunk.chunk_id, 0.0)) for chunk, score in scored_chunks]
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked

    def _chunk_route_bonus(self, chunk: ChunkRecord, route: RouteDecision) -> float:
        score = 0.0
        if route.theme != OTHER and chunk.route_theme == route.theme:
            score += 0.02
        if route.legal_role != OTHER and chunk.route_legal_role == route.legal_role:
            score += 0.01
        if route.industry != OTHER and chunk.route_industry == route.industry:
            score += 0.01
        return score

    def _chunk_focus_bonus(self, chunk: ChunkRecord, route: RouteDecision) -> float:
        score = 0.0
        if route.focus == FOCUS_ORDER and HEADER_ORDER in chunk.header:
            score += 0.04
        if route.focus == FOCUS_LAW and any(word in chunk.header for word in HEADER_LAW_WORDS):
            score += 0.035
        if route.focus == FOCUS_FACT and any(word in chunk.header for word in HEADER_FACT_WORDS):
            score += 0.035
        if "\uc774 \uc720" in chunk.header:
            score += 0.08
        if chunk.chunk_index == 1:
            score += 0.20
        if route.focus in {FOCUS_FACT, FOCUS_LAW} and "\uc774 \uc720" in chunk.header and chunk.chunk_index <= 15:
            score += 0.04
        if route.focus == FOCUS_ORDER and chunk.chunk_index >= max(1, len(self.doc_to_chunk_ids[chunk.doc_id]) - 10):
            score += 0.03
        return score

    def _select_top_chunks(self, scored_chunks: List[tuple[ChunkRecord, float]]) -> List[RetrievedChunk]:
        selected = []
        seen_ids = set()
        for chunk, score in scored_chunks:
            if chunk.chunk_id in seen_ids:
                continue
            selected.append((chunk, score))
            seen_ids.add(chunk.chunk_id)
            if len(selected) == TOP_K_CHUNKS:
                break
        return [RetrievedChunk(chunk_id=chunk.chunk_id, title=chunk.title, score=score) for chunk, score in selected]

    def _build_graph(self):
        graph = StateGraph(RetrievalState)
        graph.add_node("route_query", self.route_query)
        graph.add_node("retrieve_documents", self.retrieve_documents)
        graph.add_node("retrieve_chunks", self.retrieve_chunks)
        graph.add_edge(START, "route_query")
        graph.add_edge("route_query", "retrieve_documents")
        graph.add_edge("retrieve_documents", "retrieve_chunks")
        graph.add_edge("retrieve_chunks", END)
        return graph.compile()

    def search(self, query: str) -> List[RetrievedChunk]:
        started_at = time.perf_counter()
        result = self.graph.invoke({"query": query})
        results = result.get("results", [])
        self._print_query_timing(query=query, timings=result.get("timings", {}), started_at=started_at)
        self._log_stage("query.search_total", started_at, extra=f"query={query!r} top_chunks={len(results)}")
        return results

    def search_with_route(self, query: str, route: RouteDecision) -> List[RetrievedChunk]:
        started_at = time.perf_counter()
        state: RetrievalState = {"query": query, "route": route, "timings": {"route": 0.0}}
        state.update(self.retrieve_documents(state))
        state.update(self.retrieve_chunks(state))
        results = state.get("results", [])
        self._print_query_timing(query=query, timings=state.get("timings", {}), started_at=started_at)
        self._log_stage("query.search_total", started_at, extra=f"query={query!r} top_chunks={len(results)}")
        return results

    def _print_query_timing(self, *, query: str, timings: Dict[str, float], started_at: float) -> None:
        if not self.query_timing_log_enabled:
            return
        total_elapsed = time.perf_counter() - started_at
        route_elapsed = float(timings.get("route", 0.0))
        document_elapsed = float(timings.get("documents", 0.0))
        chunk_elapsed = float(timings.get("chunks", 0.0))
        print(
            "[timing] "
            f"query={query!r} "
            f"route={route_elapsed:.3f}s "
            f"documents={document_elapsed:.3f}s "
            f"chunks={chunk_elapsed:.3f}s "
            f"total={total_elapsed:.3f}s"
        )

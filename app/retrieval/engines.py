from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from app.preprocessing.corpus import CorpusStore
from app.retrieval.bgem3 import BGEM3HybridModel
from app.retrieval.types import QueryAnalysis, RetrievalHit
from app.utils.config import (
    resolve_chroma_index_dir,
    resolve_chroma_manifest_path,
    resolve_multivector_index_path,
    resolve_multivector_manifest_path,
    resolve_sparse_index_path,
    resolve_sparse_manifest_path,
)


INDEX_FORMAT_VERSION = 1


def build_chunk_fingerprint(chunk_ids: list[str], chunk_texts: list[str]) -> str:
    digest = hashlib.sha256()
    for chunk_id, chunk_text in zip(chunk_ids, chunk_texts):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk_text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def write_manifest(manifest_path: Path, payload: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def manifest_matches(manifest: dict, *, fingerprint: str, count: int) -> bool:
    return (
        manifest.get("format_version") == INDEX_FORMAT_VERSION
        and manifest.get("fingerprint") == fingerprint
        and int(manifest.get("chunk_count", -1)) == count
    )


class DenseSearchEngine:
    # BGE-M3 dense retrieval 경로를 초기화합니다.
    def __init__(self, corpus_store: CorpusStore, model: BGEM3HybridModel) -> None:
        self.chunks = corpus_store.chunks
        self.model = model
        self.chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        self.chunk_texts = [chunk.enriched_text for chunk in self.chunks]
        self.chunk_fingerprint = build_chunk_fingerprint(self.chunk_ids, self.chunk_texts)
        self.chroma_dir = resolve_chroma_index_dir()
        self.manifest_path = resolve_chroma_manifest_path()
        self.collection_name = "bgem3_dense_chunks"
        self.collection = self.build_collection()

    # Chroma 컬렉션을 생성하고 현재 코퍼스 기준으로 동기화합니다.
    def build_collection(self):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("dense retrieval을 ChromaDB로 사용하려면 chromadb 패키지가 필요합니다.") from exc

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        collection = client.get_or_create_collection(name=self.collection_name)

        manifest = load_manifest(self.manifest_path)
        current_count = collection.count()
        expected_count = len(self.chunks)
        if current_count != expected_count or not manifest_matches(
            manifest,
            fingerprint=self.chunk_fingerprint,
            count=expected_count,
        ):
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass
            collection = client.get_or_create_collection(name=self.collection_name)
            self.populate_collection(collection)
            write_manifest(
                self.manifest_path,
                {
                    "format_version": INDEX_FORMAT_VERSION,
                    "engine": "dense_chroma",
                    "chunk_count": expected_count,
                    "fingerprint": self.chunk_fingerprint,
                },
            )
        return collection

    # BGE-M3 dense embedding을 Chroma 컬렉션에 적재합니다.
    def populate_collection(self, collection) -> None:
        embeddings = self.model.encode_dense(self.chunk_texts).tolist()
        documents = [chunk.content for chunk in self.chunks]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "header": chunk.header,
                "section": chunk.section,
            }
            for chunk in self.chunks
        ]
        collection.add(
            ids=self.chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    # dense 질의 벡터를 생성합니다.
    def encode_query(self, analysis: QueryAnalysis) -> list[float]:
        query_matrix = self.model.encode_dense([analysis.question])
        if len(query_matrix) == 0:
            return []
        return query_matrix[0].astype("float32").tolist()

    # dense 경로로 상위 청크 후보를 반환합니다.
    def search(self, analysis: QueryAnalysis, *, candidate_doc_ids: set[str] | None, top_k: int) -> list[RetrievalHit]:
        query_vector = self.encode_query(analysis)
        if not query_vector:
            return []

        fetch_k = max(top_k * 8, 50)
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_k,
            include=["distances", "metadatas"],
        )

        hits: list[RetrievalHit] = []
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        for chunk_id, distance, metadata in zip(ids, distances, metadatas):
            if candidate_doc_ids and metadata and metadata.get("doc_id") not in candidate_doc_ids:
                continue
            similarity = 1.0 / (1.0 + float(distance))
            hits.append(RetrievalHit(chunk_id=chunk_id, score=similarity, source="dense"))
            if len(hits) == top_k:
                break
        return sorted(hits, key=lambda item: (item.score, item.chunk_id), reverse=True)[:top_k]


class SparseSearchEngine:
    # BGE-M3 sparse retrieval 경로를 초기화합니다.
    def __init__(self, corpus_store: CorpusStore, model: BGEM3HybridModel) -> None:
        self.chunks = corpus_store.chunks
        self.model = model
        self.chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        self.chunk_texts = [chunk.enriched_text for chunk in self.chunks]
        self.chunk_fingerprint = build_chunk_fingerprint(self.chunk_ids, self.chunk_texts)
        self.index_path = resolve_sparse_index_path()
        self.manifest_path = resolve_sparse_manifest_path()
        self.chunk_sparse_matrix = self.load_or_build_index()

    def load_or_build_index(self):
        from scipy.sparse import load_npz, save_npz

        manifest = load_manifest(self.manifest_path)
        if self.index_path.exists() and manifest_matches(
            manifest,
            fingerprint=self.chunk_fingerprint,
            count=len(self.chunks),
        ):
            return load_npz(self.index_path)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        sparse_matrix = self.model.encode_sparse(self.chunk_texts)
        save_npz(self.index_path, sparse_matrix)
        write_manifest(
            self.manifest_path,
            {
                "format_version": INDEX_FORMAT_VERSION,
                "engine": "sparse",
                "chunk_count": len(self.chunks),
                "fingerprint": self.chunk_fingerprint,
                "index_file": self.index_path.name,
            },
        )
        return sparse_matrix

    # sparse 질의 벡터를 생성합니다.
    def encode_query(self, analysis: QueryAnalysis):
        return self.model.encode_sparse([analysis.question])

    # sparse 쿼리와 문서 행렬의 점수를 행렬곱으로 계산합니다.
    def score_all(self, query_vector) -> np.ndarray:
        if query_vector.shape[1] == 0 or self.chunk_sparse_matrix.shape[1] == 0:
            return np.zeros((0,), dtype="float32")
        return (self.chunk_sparse_matrix @ query_vector.T).toarray().ravel().astype("float32")

    # sparse 경로로 상위 청크 후보를 반환합니다.
    def search(self, analysis: QueryAnalysis, *, candidate_doc_ids: set[str] | None, top_k: int) -> list[RetrievalHit]:
        query_vector = self.encode_query(analysis)
        scores = self.score_all(query_vector)
        hits: list[RetrievalHit] = []
        for index, score in enumerate(scores):
            chunk = self.chunks[index]
            if candidate_doc_ids and chunk.doc_id not in candidate_doc_ids:
                continue
            hits.append(RetrievalHit(chunk_id=chunk.chunk_id, score=float(score), source="sparse"))
        return sorted(hits, key=lambda item: (item.score, item.chunk_id), reverse=True)[:top_k]


class MultiVectorSearchEngine:
    # BGE-M3 multi-vector retrieval 경로를 초기화합니다.
    def __init__(self, corpus_store: CorpusStore, model: BGEM3HybridModel) -> None:
        self.chunks = corpus_store.chunks
        self.model = model
        self.chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        self.chunk_texts = [chunk.enriched_text for chunk in self.chunks]
        self.chunk_fingerprint = build_chunk_fingerprint(self.chunk_ids, self.chunk_texts)
        self.index_path = resolve_multivector_index_path()
        self.manifest_path = resolve_multivector_manifest_path()
        self.chunk_multivectors = self.load_or_build_index()

    def load_or_build_index(self) -> list[np.ndarray]:
        manifest = load_manifest(self.manifest_path)
        if self.index_path.exists() and manifest_matches(
            manifest,
            fingerprint=self.chunk_fingerprint,
            count=len(self.chunks),
        ):
            return self.load_multivectors(self.index_path)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        multivectors = self.model.encode_multivector(self.chunk_texts)
        self.save_multivectors(self.index_path, multivectors)
        write_manifest(
            self.manifest_path,
            {
                "format_version": INDEX_FORMAT_VERSION,
                "engine": "multivector",
                "chunk_count": len(self.chunks),
                "fingerprint": self.chunk_fingerprint,
                "index_file": self.index_path.name,
            },
        )
        return multivectors

    @staticmethod
    def save_multivectors(index_path: Path, vectors: list[np.ndarray]) -> None:
        lengths = np.asarray([vector.shape[0] for vector in vectors], dtype=np.int32)
        non_empty_vectors = [vector.astype("float32") for vector in vectors if vector.size > 0]
        if non_empty_vectors:
            flat_vectors = np.concatenate(non_empty_vectors, axis=0).astype("float32")
            vector_dim = int(flat_vectors.shape[1])
        else:
            vector_dim = 0
            flat_vectors = np.zeros((0, 0), dtype="float32")
        np.savez_compressed(
            index_path,
            flat_vectors=flat_vectors,
            lengths=lengths,
            vector_dim=np.asarray([vector_dim], dtype=np.int32),
        )

    @staticmethod
    def load_multivectors(index_path: Path) -> list[np.ndarray]:
        payload = np.load(index_path, allow_pickle=False)
        flat_vectors = payload["flat_vectors"].astype("float32")
        lengths = payload["lengths"].astype(np.int32)
        vector_dim = int(payload["vector_dim"][0]) if "vector_dim" in payload else 0

        vectors: list[np.ndarray] = []
        offset = 0
        for length in lengths.tolist():
            if length == 0:
                vectors.append(np.zeros((0, vector_dim), dtype="float32"))
                continue
            next_offset = offset + length
            vectors.append(flat_vectors[offset:next_offset])
            offset = next_offset
        return vectors

    # multi-vector 질의 표현을 생성합니다.
    def encode_query(self, analysis: QueryAnalysis) -> np.ndarray:
        vectors = self.model.encode_multivector([analysis.question])
        if not vectors:
            return np.zeros((0, 0), dtype="float32")
        return vectors[0]

    # 특정 청크의 multi-vector 표현을 조회합니다.
    def get_chunk_vectors(self, chunk_index: int) -> np.ndarray:
        return self.chunk_multivectors[chunk_index]

    # late interaction 점수로 전체 후보를 계산합니다.
    def score_all(self, query_vectors: np.ndarray) -> list[float]:
        scores: list[float] = []
        for doc_vectors in self.chunk_multivectors:
            scores.append(float(self.model.multivector_score(query_vectors, doc_vectors)))
        return scores

    # late interaction 점수로 상위 청크 후보를 반환합니다.
    def search(self, analysis: QueryAnalysis, *, candidate_doc_ids: set[str] | None, top_k: int) -> list[RetrievalHit]:
        query_vectors = self.encode_query(analysis)
        scores = self.score_all(query_vectors)
        hits: list[RetrievalHit] = []
        for index, score in enumerate(scores):
            chunk = self.chunks[index]
            if candidate_doc_ids and chunk.doc_id not in candidate_doc_ids:
                continue
            hits.append(RetrievalHit(chunk_id=chunk.chunk_id, score=float(score), source="multivector"))
        return sorted(hits, key=lambda item: (item.score, item.chunk_id), reverse=True)[:top_k]

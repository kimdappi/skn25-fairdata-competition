from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from app.preprocessing.corpus import CorpusStore
from app.retrieval.interfaces import (
    DenseRetrievalBackend,
    MultiVectorRetrievalBackend,
    SparseRetrievalBackend,
)
from app.retrieval.types import QueryAnalysis, RetrievalHit
from app.utils.config import (
    resolve_chroma_index_dir,
    resolve_multivector_index_path,
    resolve_sparse_index_path,
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


def sanitize_model_tag(model_dir: Path) -> str:
    return hashlib.sha1(str(model_dir.resolve()).encode("utf-8")).hexdigest()[:12]


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
    def __init__(self, corpus_store: CorpusStore, backend: DenseRetrievalBackend) -> None:
        self.chunks = corpus_store.chunks
        self.backend = backend
        self.model_tag = sanitize_model_tag(self.backend.model_dir)
        self.chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        self.chunk_texts = [chunk.enriched_text for chunk in self.chunks]
        self.chunk_fingerprint = build_chunk_fingerprint(self.chunk_ids, self.chunk_texts)
        self.chroma_dir = resolve_chroma_index_dir() / self.model_tag
        self.manifest_path = self.chroma_dir / "manifest.json"
        self.collection_name = f"bgem3_dense_chunks_{self.model_tag}"
        self.collection = self.build_collection()

    # Chroma 컬렉션을 생성하고 현재 코퍼스 기준으로 동기화합니다.
    def build_collection(self):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("dense retrieval을 ChromaDB로 사용하려면 chromadb 패키지가 필요합니다.") from exc

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.chroma_dir))

        manifest = load_manifest(self.manifest_path)
        expected_count = len(self.chunks)
        manifest_ok = manifest_matches(manifest, fingerprint=self.chunk_fingerprint, count=expected_count)

        if manifest_ok:
            try:
                collection = client.get_collection(self.collection_name)
                if collection.count() == expected_count:
                    return collection
            except Exception:
                pass

        # manifest 불일치/컬렉션 누락/개수 불일치인 경우에만 삭제 후 재생성합니다.
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:search_ef": 100},
        )
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
    # 24GB GPU에서 메모리 초과를 방지하기 위해 500개 단위로 배치 처리합니다.
    def populate_collection(self, collection, batch_size: int = 500) -> None:
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

        total = len(self.chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = self.chunk_texts[start:end]
            batch_embeddings = self.backend.encode_documents(batch_texts).tolist()
            collection.add(
                ids=self.chunk_ids[start:end],
                embeddings=batch_embeddings,
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"[DenseSearchEngine] inserted {end}/{total}")
        print(f"[DenseSearchEngine] populate_collection complete: {total} chunks")

    # dense 질의 벡터를 생성합니다.
    def encode_query(self, analysis: QueryAnalysis) -> list[float]:
        return self.backend.encode_query(analysis.question)

    # dense 경로로 상위 청크 후보를 반환합니다.
    def search(self, analysis: QueryAnalysis, *, top_k: int) -> list[RetrievalHit]:
        query_vector = self.encode_query(analysis)
        if not query_vector:
            return []

        fetch_k = max(top_k * 6, 30)
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_k,
            include=["distances", "metadatas"],
        )

        hits: list[RetrievalHit] = []
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for chunk_id, distance in zip(ids, distances):
            similarity = 1.0 / (1.0 + float(distance))
            hits.append(RetrievalHit(chunk_id=chunk_id, score=similarity, source="dense"))
            if len(hits) == top_k:
                break
        return sorted(hits, key=lambda item: (item.score, item.chunk_id), reverse=True)[:top_k]


class SparseSearchEngine:
    # learned sparse(BGE-M3) 또는 BM25 sparse retrieval 경로를 초기화합니다.
    def __init__(self, corpus_store: CorpusStore, backend: SparseRetrievalBackend) -> None:
        self.chunks = corpus_store.chunks
        self.backend = backend
        self.model_tag = sanitize_model_tag(self.backend.model_dir)
        self.chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        self.chunk_texts = [chunk.enriched_text for chunk in self.chunks]
        self.chunk_fingerprint = build_chunk_fingerprint(self.chunk_ids, self.chunk_texts)
        self.backend_tag = self.backend.__class__.__name__.lower().replace("backend", "")
        sparse_root = resolve_sparse_index_path().parent
        self.index_path = sparse_root / f"sparse_{self.backend_tag}_chunks_{self.model_tag}.npz"
        self.manifest_path = sparse_root / f"sparse_{self.backend_tag}_chunks_{self.model_tag}_manifest.json"
        self.chunk_sparse_index = self.load_or_build_index()

    def load_or_build_index(self):
        manifest = load_manifest(self.manifest_path)
        if self.index_path.exists() and manifest_matches(
            manifest,
            fingerprint=self.chunk_fingerprint,
            count=len(self.chunks),
        ):
            return self.backend.load_index(self.index_path)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        sparse_index = self.backend.encode_documents(self.chunk_texts)
        self.backend.save_index(self.index_path, sparse_index)
        write_manifest(
            self.manifest_path,
            {
                "format_version": INDEX_FORMAT_VERSION,
                "engine": f"sparse_{self.backend_tag}",
                "chunk_count": len(self.chunks),
                "fingerprint": self.chunk_fingerprint,
                "index_file": self.index_path.name,
            },
        )
        return sparse_index

    # sparse 질의 벡터를 생성합니다.
    def encode_query(self, analysis: QueryAnalysis):
        return self.backend.encode_query(analysis.question)

    # sparse 쿼리와 문서 인덱스의 점수를 backend별 방식으로 계산합니다.
    def score_all(self, query_vector) -> np.ndarray:
        return self.backend.score_all(self.chunk_sparse_index, query_vector)

    # sparse 경로로 상위 청크 후보를 반환합니다.
    def search(self, analysis: QueryAnalysis, *, top_k: int) -> list[RetrievalHit]:
        query_vector = self.encode_query(analysis)
        scores = self.score_all(query_vector)
        hits: list[RetrievalHit] = []
        for index, score in enumerate(scores):
            chunk = self.chunks[index]
            hits.append(RetrievalHit(chunk_id=chunk.chunk_id, score=float(score), source="sparse"))
        return sorted(hits, key=lambda item: (item.score, item.chunk_id), reverse=True)[:top_k]


class MultiVectorSearchEngine:
    # BGE-M3 multi-vector retrieval 경로를 초기화합니다.
    def __init__(
        self,
        corpus_store: CorpusStore,
        backend: MultiVectorRetrievalBackend,
        *,
        eager_load: bool = True,
    ) -> None:
        self.chunks = corpus_store.chunks
        self.backend = backend
        self.model_tag = sanitize_model_tag(self.backend.model_dir)
        self.chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        self.chunk_texts = [chunk.enriched_text for chunk in self.chunks]
        self.chunk_fingerprint = build_chunk_fingerprint(self.chunk_ids, self.chunk_texts)
        multivector_root = resolve_multivector_index_path().parent
        self.index_path = multivector_root / f"multivector_bgem3_chunks_{self.model_tag}.npz"
        self.manifest_path = multivector_root / f"multivector_bgem3_chunks_{self.model_tag}_manifest.json"
        self.shard_dir = multivector_root / f"multivector_bgem3_chunks_{self.model_tag}_shards"
        self.chunk_multivectors = self.load_or_build_index(eager_load=eager_load)

    def load_or_build_index(self, *, eager_load: bool = True) -> list[np.ndarray]:
        manifest = load_manifest(self.manifest_path)
        if manifest_matches(
            manifest,
            fingerprint=self.chunk_fingerprint,
            count=len(self.chunks),
        ):
            if self._manifest_uses_shards(manifest):
                return self.load_sharded_multivectors(manifest) if eager_load else []
            if self.index_path.exists():
                return self.load_multivectors(self.index_path) if eager_load else []

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.build_sharded_index()
        return self.load_sharded_multivectors(manifest) if eager_load else []

    def _manifest_uses_shards(self, manifest: dict) -> bool:
        shard_files = manifest.get("shard_files") or []
        return manifest.get("storage") == "sharded" and bool(shard_files)

    def _clear_existing_storage(self) -> None:
        if self.index_path.exists():
            self.index_path.unlink()
        if self.shard_dir.exists():
            for path in sorted(self.shard_dir.glob("*.npz")):
                path.unlink()

    def build_sharded_index(self) -> dict:
        batch_size = max(1, int(os.getenv("FAIRDATA_MULTIVECTOR_BATCH_SIZE", "32")))
        shard_size = max(batch_size, int(os.getenv("FAIRDATA_MULTIVECTOR_SHARD_SIZE", "512")))
        total = len(self.chunk_texts)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._clear_existing_storage()
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        shard_files: list[str] = []
        shard_doc_counts: list[int] = []
        shard_vectors: list[np.ndarray] = []
        shard_index = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            encoded_vectors = [
                np.asarray(vector, dtype="float32")
                for vector in self.backend.encode_documents(self.chunk_texts[start:end])
            ]
            shard_vectors.extend(encoded_vectors)
            print(f"[MultiVectorSearchEngine] encoded {end}/{total}")

            while len(shard_vectors) >= shard_size:
                current_vectors = shard_vectors[:shard_size]
                shard_path = self.shard_dir / f"shard_{shard_index:04d}.npz"
                self.save_multivectors(shard_path, current_vectors)
                shard_files.append(shard_path.name)
                shard_doc_counts.append(len(current_vectors))
                shard_vectors = shard_vectors[shard_size:]
                shard_index += 1

        if shard_vectors:
            shard_path = self.shard_dir / f"shard_{shard_index:04d}.npz"
            self.save_multivectors(shard_path, shard_vectors)
            shard_files.append(shard_path.name)
            shard_doc_counts.append(len(shard_vectors))

        manifest = {
            "format_version": INDEX_FORMAT_VERSION,
            "engine": "multivector",
            "storage": "sharded",
            "chunk_count": len(self.chunks),
            "fingerprint": self.chunk_fingerprint,
            "batch_size": batch_size,
            "shard_size": shard_size,
            "shard_dir": self.shard_dir.name,
            "shard_files": shard_files,
            "shard_doc_counts": shard_doc_counts,
        }
        write_manifest(self.manifest_path, manifest)
        return manifest

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

    def load_sharded_multivectors(self, manifest: dict) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for shard_file in manifest.get("shard_files", []):
            shard_path = self.shard_dir / shard_file
            vectors.extend(self.load_multivectors(shard_path))
        return vectors

    # multi-vector 질의 표현을 생성합니다.
    def encode_query(self, analysis: QueryAnalysis) -> np.ndarray:
        return self.backend.encode_query(analysis.question)

    # late interaction 점수로 전체 후보를 계산합니다.
    def score_all(self, query_vectors: np.ndarray) -> list[float]:
        scores: list[float] = []
        for doc_vectors in self.chunk_multivectors:
            scores.append(self.backend.score(query_vectors, doc_vectors))
        return scores

    # late interaction 점수로 상위 청크 후보를 반환합니다.
    def search(self, analysis: QueryAnalysis, *, top_k: int) -> list[RetrievalHit]:
        query_vectors = self.encode_query(analysis)
        scores = self.score_all(query_vectors)
        hits: list[RetrievalHit] = []
        for index, score in enumerate(scores):
            chunk = self.chunks[index]
            hits.append(RetrievalHit(chunk_id=chunk.chunk_id, score=float(score), source="multivector"))
        return sorted(hits, key=lambda item: (item.score, item.chunk_id), reverse=True)[:top_k]

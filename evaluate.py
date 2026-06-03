import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("STAGE_LOG_ENABLED", "false")
os.environ.setdefault("QUERY_TIMING_LOG_ENABLED", "false")
os.environ.setdefault("ROUTER_LOG_ENABLED", "false")
os.environ.setdefault("FAIRCOMP_GENERATION_MODEL", "Qwen3-8B")
os.environ.setdefault("HF_ROUTER_MODEL", "Qwen3-8B")
os.environ.setdefault("HF_ROUTER_LOCAL_FILES_ONLY", "true")
os.environ.setdefault("ROUTER_BACKEND", "hf")
os.environ.setdefault("OLLAMA_ROUTER_MODEL", "Qwen3-8B")
os.environ.setdefault("FAIRCOMP_TORCH_DTYPE", "bfloat16")

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from baseline_rag.config import MODEL_DIR, RESULT_DIR, validate_runtime_paths
from baseline_rag.experiments import RetrieverOptions
from baseline_rag.predictor import RAGPredictor, token_f1

WEIGHTS = {
    "recall_at_5": 0.35,
    "mrr": 0.15,
    "bertscore": 0.30,
    "f1": 0.20,
}


def mean(items: list[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def dataset_path() -> Path:
    explicit = os.getenv("FAIRCOMP_EVAL_DATASET_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path("from_uy/example/eval_dataset_260505.json")


def load_dataset() -> list[dict]:
    path = dataset_path()
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_active_dataset(dataset: list[dict]) -> tuple[list[dict], int, int, int]:
    limit = int(os.getenv("EVAL_LIMIT", "0"))
    sample_size = int(os.getenv("EVAL_SAMPLE_SIZE", "0"))
    sample_seed = int(os.getenv("EVAL_SAMPLE_SEED", "42"))
    active_dataset = dataset[:limit] if limit > 0 else list(dataset)
    if sample_size > 0:
        rng = random.Random(sample_seed)
        active_dataset = rng.sample(active_dataset, min(sample_size, len(active_dataset)))
    return active_dataset, limit, sample_size, sample_seed


def recall_at_5(predicted_ids: list[str], gold_set: set[str]) -> float:
    return 1.0 if any(chunk_id in gold_set for chunk_id in predicted_ids[:5]) else 0.0


def reciprocal_rank(predicted_ids: list[str], gold_set: set[str]) -> float:
    for rank, chunk_id in enumerate(predicted_ids, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


class SemanticScorer:
    def __init__(self, model_name: str) -> None:
        self.model_dir = MODEL_DIR / model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_dir, local_files_only=True).to(self.device)
        self.model.eval()

    def bertscore_f1(self, prediction: str, reference: str) -> float:
        pred_emb = self._token_embeddings(prediction)
        ref_emb = self._token_embeddings(reference)
        if pred_emb is None or ref_emb is None:
            return 0.0

        similarity = pred_emb @ ref_emb.transpose(0, 1)
        precision = similarity.max(dim=1).values.mean()
        recall = similarity.max(dim=0).values.mean()
        denom = precision + recall
        if float(denom) == 0.0:
            return 0.0
        return float((2 * precision * recall / denom).item())

    def _token_embeddings(self, text: str) -> torch.Tensor | None:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state[0]
        attention_mask = encoded["attention_mask"][0].bool()
        token_ids = encoded["input_ids"][0]
        special_mask = torch.tensor(
            [self.tokenizer.get_special_tokens_mask(token_ids.tolist(), already_has_special_tokens=True)],
            device=self.device,
            dtype=torch.bool,
        )[0]
        content_mask = attention_mask & ~special_mask
        if not bool(content_mask.any()):
            return None
        embeddings = hidden[content_mask]
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)


@dataclass(frozen=True)
class EvaluationCase:
    embedding_model_name: str
    family_label: str
    router_enabled: bool
    options: RetrieverOptions
    generation_model_name: str

    @property
    def run_name(self) -> str:
        router_label = "router_on" if self.router_enabled else "router_off"
        return f"{self.embedding_model_name}__{self.family_label}__{router_label}"

    def to_metadata(self) -> dict:
        return {
            "run_name": self.run_name,
            "embedding_model_name": self.embedding_model_name,
            "generation_model_name": self.generation_model_name,
            "family_label": self.family_label,
            "router_enabled": self.router_enabled,
            "options": self.options.to_metadata(),
        }


def discover_embedding_models() -> list[str]:
    discovered = sorted(
        path.name
        for path in MODEL_DIR.iterdir()
        if path.is_dir() and path.name.startswith("embedding_")
    )
    explicit = os.getenv("EVAL_EMBEDDING_MODELS", "").strip()
    if explicit:
        allowed = {name.strip() for name in explicit.split(",") if name.strip()}
        discovered = [name for name in discovered if name in allowed]
    return discovered


def build_case_matrix() -> list[EvaluationCase]:
    generation_model_name = os.getenv("FAIRCOMP_GENERATION_MODEL", "Qwen3-8B")
    cases: list[EvaluationCase] = []
    discovered_models = set(discover_embedding_models())
    requested_models = ["embedding_bge_m3", "embedding_ko_legal_sbert"]

    preset_matrix: dict[str, list[tuple[str, dict[str, object]]]] = {
        "embedding_bge_m3": [
            (
                "dense_only",
                {
                    "router_enabled": False,
                    "use_dense": True,
                    "use_bm25": False,
                    "use_bge_sparse": False,
                    "use_bge_colbert": False,
                },
            ),
            (
                "bm25_only",
                {
                    "router_enabled": False,
                    "use_dense": False,
                    "use_bm25": True,
                    "use_bge_sparse": False,
                    "use_bge_colbert": False,
                },
            ),
            (
                "dense_bm25_sparse_rrf",
                {
                    "router_enabled": False,
                    "use_dense": True,
                    "use_bm25": True,
                    "use_bge_sparse": True,
                    "use_bge_colbert": False,
                },
            ),
            (
                "dense_bm25_sparse_routing_rrf_colbert",
                {
                    "router_enabled": True,
                    "use_dense": True,
                    "use_bm25": True,
                    "use_bge_sparse": True,
                    "use_bge_colbert": True,
                },
            ),
        ],
        "embedding_ko_legal_sbert": [
            (
                "dense_only",
                {
                    "router_enabled": False,
                    "use_dense": True,
                    "use_bm25": False,
                    "use_bge_sparse": False,
                    "use_bge_colbert": False,
                },
            ),
            (
                "bm25_only",
                {
                    "router_enabled": False,
                    "use_dense": False,
                    "use_bm25": True,
                    "use_bge_sparse": False,
                    "use_bge_colbert": False,
                },
            ),
            (
                "dense_bm25_rrf",
                {
                    "router_enabled": False,
                    "use_dense": True,
                    "use_bm25": True,
                    "use_bge_sparse": False,
                    "use_bge_colbert": False,
                },
            ),
            (
                "dense_bm25_routing_rrf",
                {
                    "router_enabled": True,
                    "use_dense": True,
                    "use_bm25": True,
                    "use_bge_sparse": False,
                    "use_bge_colbert": False,
                },
            ),
        ],
    }

    for embedding_model_name in requested_models:
        if embedding_model_name not in discovered_models:
            continue
        for family_label, config in preset_matrix[embedding_model_name]:
            router_enabled = bool(config["router_enabled"])
            use_bm25 = bool(config["use_bm25"])
            use_bge_sparse = bool(config["use_bge_sparse"])
            options = RetrieverOptions(
                router_mode="ollama" if router_enabled else "off",
                use_dense=bool(config["use_dense"]),
                use_bm25=use_bm25,
                use_routing=router_enabled,
                use_route_boost=router_enabled,
                use_entity_boost=router_enabled,
                use_chunk_lexical_score=use_bm25 or use_bge_sparse,
                use_chunk_structure_boost=router_enabled,
                use_doc_rank_boost=True,
                use_bge_sparse=use_bge_sparse,
                use_bge_colbert=bool(config["use_bge_colbert"]),
            ).normalized(embedding_model_name)
            cases.append(
                EvaluationCase(
                    embedding_model_name=embedding_model_name,
                    family_label=family_label,
                    router_enabled=router_enabled,
                    options=options,
                    generation_model_name=generation_model_name,
                )
            )

    family_contains = os.getenv("EVAL_FAMILY_CONTAINS", "").strip()
    model_contains = os.getenv("EVAL_MODEL_CONTAINS", "").strip()
    router_only = os.getenv("EVAL_ROUTER_ONLY", "").strip().lower()

    filtered = cases
    if family_contains:
        filtered = [case for case in filtered if family_contains in case.family_label]
    if model_contains:
        filtered = [case for case in filtered if model_contains in case.embedding_model_name]
    if router_only in {"on", "off"}:
        expected = router_only == "on"
        filtered = [case for case in filtered if case.router_enabled == expected]
    return filtered


def prediction_path(case: EvaluationCase) -> Path:
    return RESULT_DIR / f"{case.run_name}_predictions.json"


def summary_path(case: EvaluationCase) -> Path:
    return RESULT_DIR / f"{case.run_name}_summary.json"


def print_selected_cases(cases: list[EvaluationCase]) -> None:
    print("Selected evaluation methods:")
    for case in cases:
        print(
            f"- {case.run_name} "
            f"(embedding={case.embedding_model_name}, "
            f"family={case.family_label}, "
            f"router={'on' if case.router_enabled else 'off'})"
        )


def evaluate_case(case: EvaluationCase, dataset: list[dict], scorer: SemanticScorer) -> tuple[dict, list[dict]]:
    active_dataset, limit, sample_size, sample_seed = load_active_dataset(dataset)
    predictor_started = time.perf_counter()
    predictor = RAGPredictor(
        embedding_model_name=case.embedding_model_name,
        generation_model_name=case.generation_model_name,
        options=case.options,
    )
    predictor_init_seconds = time.perf_counter() - predictor_started

    recalls: list[float] = []
    mrrs: list[float] = []
    bertscores: list[float] = []
    f1_scores: list[float] = []
    answer_elapsed_seconds: list[float] = []
    predictions: list[dict] = []

    query_bar = tqdm(active_dataset, desc=case.run_name[:70], ncols=160, leave=False)
    for row in query_bar:
        answer_started = time.perf_counter()
        result = predictor.predict(row["query"])
        row_answer_elapsed_seconds = time.perf_counter() - answer_started
        predicted_ids = result.retrieved_chunk_ids
        gold_ids = [item["chunk_id"] for item in row["answer_chunks"]]
        gold_set = set(gold_ids)
        reference_answer = row.get("reference_answer", "")

        row_recall = recall_at_5(predicted_ids, gold_set)
        row_mrr = reciprocal_rank(predicted_ids, gold_set)
        row_bertscore = scorer.bertscore_f1(result.answer, reference_answer) if reference_answer else 0.0
        row_f1 = token_f1(result.answer, reference_answer) if reference_answer else 0.0

        recalls.append(row_recall)
        mrrs.append(row_mrr)
        bertscores.append(row_bertscore)
        f1_scores.append(row_f1)
        answer_elapsed_seconds.append(row_answer_elapsed_seconds)

        predictions.append(
            {
                "id": row["id"],
                "query": row["query"],
                "source_doc": row["source_doc"],
                "gold_chunk_ids": gold_ids,
                "predicted_chunk_ids": predicted_ids,
                "reference_answer": reference_answer,
                "predicted_answer": result.answer,
                "recall_at_5": row_recall,
                "mrr": row_mrr,
                "bertscore": row_bertscore,
                "f1": row_f1,
                "answer_elapsed_seconds": row_answer_elapsed_seconds,
            }
        )
        query_bar.set_postfix(
            recall_at_5=f"{mean(recalls):.4f}",
            mrr=f"{mean(mrrs):.4f}",
            bertscore=f"{mean(bertscores):.4f}",
            f1=f"{mean(f1_scores):.4f}",
            avg_answer_s=f"{mean(answer_elapsed_seconds):.3f}",
        )

    summary = {
        "created_at": datetime.now().isoformat(),
        "case": case.to_metadata(),
        "dataset_path": str(dataset_path()),
        "dataset_size": len(active_dataset),
        "predictor_init_seconds": predictor_init_seconds,
        "setup": {
            "eval_limit": limit,
            "eval_sample_size": sample_size,
            "eval_sample_seed": sample_seed,
            "semantic_scorer_model": os.getenv("EVAL_SEMANTIC_SCORER_MODEL", "embedding_ko_legal_sbert"),
            "router_backend": os.getenv("ROUTER_BACKEND", "hf"),
            "router_model": os.getenv("HF_ROUTER_MODEL", "Qwen3-8B"),
        },
        "recall_at_5": mean(recalls),
        "mrr": mean(mrrs),
        "bertscore": mean(bertscores),
        "f1": mean(f1_scores),
        "avg_answer_elapsed_seconds": mean(answer_elapsed_seconds),
    }
    summary["final_score"] = sum(summary[key] * weight for key, weight in WEIGHTS.items())
    return summary, predictions


def build_result_bundle(summaries: list[dict]) -> dict:
    overall_best = max(summaries, key=lambda item: item["final_score"]) if summaries else None
    best_by_embedding: dict[str, dict] = {}
    best_by_family: dict[str, dict] = {}

    for summary in summaries:
        embedding = summary["case"]["embedding_model_name"]
        family = summary["case"]["family_label"]
        if embedding not in best_by_embedding or summary["final_score"] > best_by_embedding[embedding]["final_score"]:
            best_by_embedding[embedding] = summary
        if family not in best_by_family or summary["final_score"] > best_by_family[family]["final_score"]:
            best_by_family[family] = summary

    return {
        "created_at": datetime.now().isoformat(),
        "dataset_path": str(dataset_path()),
        "generation_model_name": os.getenv("FAIRCOMP_GENERATION_MODEL", "Qwen3-8B"),
        "semantic_scorer_model": os.getenv("EVAL_SEMANTIC_SCORER_MODEL", "embedding_ko_legal_sbert"),
        "weights": WEIGHTS,
        "case_count": len(summaries),
        "overall_best": overall_best,
        "best_by_embedding": best_by_embedding,
        "best_by_family": best_by_family,
        "cases": summaries,
    }


def main() -> None:
    validate_runtime_paths()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset()
    cases = build_case_matrix()
    if not cases:
        raise ValueError("No evaluation cases were selected. Check EVAL_* filters.")
    print_selected_cases(cases)

    scorer_model_name = os.getenv("EVAL_SEMANTIC_SCORER_MODEL", "embedding_ko_legal_sbert")
    scorer = SemanticScorer(model_name=scorer_model_name)

    summaries: list[dict] = []
    case_bar = tqdm(cases, desc="Evaluation Cases", ncols=160)
    for case in case_bar:
        started_at = time.perf_counter()
        summary, predictions = evaluate_case(case, dataset, scorer)
        summary["elapsed_seconds"] = time.perf_counter() - started_at
        summaries.append(summary)
        with open(summary_path(case), "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        with open(prediction_path(case), "w", encoding="utf-8") as file:
            json.dump(predictions, file, ensure_ascii=False, indent=2)
        case_bar.set_postfix(best_score=f"{max(item['final_score'] for item in summaries):.4f}")

    bundle = build_result_bundle(summaries)
    output_path = RESULT_DIR / "evaluation_case_matrix.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(bundle, file, ensure_ascii=False, indent=2)
    print(json.dumps(bundle["overall_best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

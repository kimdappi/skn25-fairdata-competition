from dataclasses import asdict, dataclass
import os
from typing import Dict, Iterable, List, Literal

from baseline_rag.config import EVALUATION_MODEL_NAMES

RouterMode = Literal["off", "rule", "ollama"]


@dataclass(frozen=True)
class RetrieverOptions:
    router_mode: RouterMode = "ollama"
    use_dense: bool = True
    use_bm25: bool = True
    use_routing: bool = True
    use_route_boost: bool = True
    use_entity_boost: bool = True
    use_chunk_lexical_score: bool = True
    use_chunk_structure_boost: bool = True
    use_doc_rank_boost: bool = True
    use_bge_sparse: bool = True
    use_bge_colbert: bool = True

    def normalized(self, model_name: str) -> "RetrieverOptions":
        is_bge_m3 = model_name == "embedding_bge_m3"
        use_routing = self.use_routing and self.router_mode != "off"
        router_mode: RouterMode = self.router_mode if use_routing else "off"
        return RetrieverOptions(
            router_mode=router_mode,
            use_dense=self.use_dense,
            use_bm25=self.use_bm25,
            use_routing=use_routing,
            use_route_boost=self.use_route_boost and use_routing,
            use_entity_boost=self.use_entity_boost,
            use_chunk_lexical_score=self.use_chunk_lexical_score,
            use_chunk_structure_boost=self.use_chunk_structure_boost,
            use_doc_rank_boost=self.use_doc_rank_boost,
            use_bge_sparse=self.use_bge_sparse if is_bge_m3 else False,
            use_bge_colbert=self.use_bge_colbert if is_bge_m3 else False,
        )

    def requires_any_retrieval_signal(self) -> bool:
        return self.use_dense or self.use_bm25 or self.use_bge_sparse

    def to_metadata(self) -> Dict[str, object]:
        return asdict(self)

    def slug(self) -> str:
        flags = [
            f"router-{self.router_mode}",
            f"dense-{int(self.use_dense)}",
            f"bm25-{int(self.use_bm25)}",
            f"routing-{int(self.use_routing)}",
            f"routeboost-{int(self.use_route_boost)}",
            f"entity-{int(self.use_entity_boost)}",
            f"chunklex-{int(self.use_chunk_lexical_score)}",
            f"chunkstruct-{int(self.use_chunk_structure_boost)}",
            f"docrank-{int(self.use_doc_rank_boost)}",
            f"m3sparse-{int(self.use_bge_sparse)}",
            f"m3colbert-{int(self.use_bge_colbert)}",
        ]
        return "_".join(flags)


@dataclass(frozen=True)
class ExperimentVariant:
    model_name: str
    options: RetrieverOptions

    @property
    def run_name(self) -> str:
        return f"{self.model_name}__{self.short_name}"

    @property
    def short_name(self) -> str:
        return build_variant_short_name(self.model_name, self.options)

    @property
    def group_label(self) -> str:
        return build_variant_group_label(self.model_name, self.options)

    def to_metadata(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "short_name": self.short_name,
            "group_label": self.group_label,
            "options": self.options.to_metadata(),
            "run_name": self.run_name,
        }


def build_experiment_variants(model_names: Iterable[str] | None = None) -> List[ExperimentVariant]:
    names = list(model_names or EVALUATION_MODEL_NAMES)
    variants: List[ExperimentVariant] = []
    preset = os.getenv("EVAL_VARIANT_PRESET", "paper").strip().lower()
    for model_name in names:
        for option in build_options_for_model(model_name, preset=preset):
            variants.append(ExperimentVariant(model_name=model_name, options=option.normalized(model_name)))

    unique: Dict[str, ExperimentVariant] = {}
    for variant in variants:
        unique[variant.run_name] = variant
    return list(unique.values())


def build_variant_group_label(model_name: str, options: RetrieverOptions) -> str:
    is_bge_m3 = model_name == "embedding_bge_m3"
    if is_bge_m3 and options.use_bge_colbert:
        return "bge-m3-colbert"
    if is_bge_m3 and options.use_bge_sparse:
        return "bge-m3-hybrid"
    if options.router_mode == "off" and not options.use_entity_boost and not options.use_chunk_structure_boost:
        if options.use_dense and options.use_bm25:
            return "hybrid"
        if options.use_dense:
            return "baseline-dense"
        if options.use_bm25:
            return "baseline-bm25"
    if options.router_mode in {"rule", "ollama"} and not options.use_entity_boost and not options.use_chunk_structure_boost:
        return "routing"
    if options.router_mode in {"rule", "ollama"} and options.use_entity_boost and options.use_chunk_structure_boost:
        return "full"
    return "ablation"


def build_variant_short_name(model_name: str, options: RetrieverOptions) -> str:
    router_label = {
        "off": "no-router",
        "rule": "rule-router",
        "ollama": "llm-router",
    }[options.router_mode]

    if model_name == "embedding_bge_m3":
        retrieval_label = "dense"
        if options.use_bm25 and options.use_bge_sparse:
            retrieval_label = "dense+bm25+m3sparse"
        elif options.use_bm25:
            retrieval_label = "dense+bm25"
        elif options.use_bge_sparse:
            retrieval_label = "dense+m3sparse"
        rerank_label = "+colbert" if options.use_bge_colbert else ""
        if options.router_mode == "off":
            return f"bge-m3-{retrieval_label}{rerank_label}"
        if options.use_entity_boost and options.use_chunk_structure_boost:
            return f"bge-m3-{retrieval_label}{rerank_label}-{router_label}-full"
        return f"bge-m3-{retrieval_label}{rerank_label}-{router_label}"

    if not options.use_dense and options.use_bm25:
        return "bm25-only"
    if options.use_dense and not options.use_bm25:
        return "dense-only"
    if options.use_dense and options.use_bm25 and options.router_mode == "off":
        return "dense+bm25"
    if options.use_dense and options.use_bm25 and options.router_mode != "off":
        if options.use_entity_boost and options.use_chunk_structure_boost:
            return f"dense+bm25-{router_label}-full"
        return f"dense+bm25-{router_label}"
    return options.slug()


def build_options_for_model(model_name: str, *, preset: str) -> List[RetrieverOptions]:
    if preset == "expanded":
        return build_meaningful_options_for_model(model_name)
    return build_paper_baseline_options_for_model(model_name)


def build_paper_baseline_options_for_model(model_name: str) -> List[RetrieverOptions]:
    is_bge_m3 = model_name == "embedding_bge_m3"
    options: List[RetrieverOptions] = [
        RetrieverOptions(
            router_mode="off",
            use_dense=False,
            use_bm25=True,
            use_routing=False,
            use_route_boost=False,
            use_entity_boost=False,
            use_chunk_lexical_score=True,
            use_chunk_structure_boost=False,
            use_doc_rank_boost=False,
        ),
        RetrieverOptions(
            router_mode="off",
            use_dense=True,
            use_bm25=False,
            use_routing=False,
            use_route_boost=False,
            use_entity_boost=False,
            use_chunk_lexical_score=False,
            use_chunk_structure_boost=False,
            use_doc_rank_boost=False,
        ),
        RetrieverOptions(
            router_mode="off",
            use_dense=True,
            use_bm25=True,
            use_routing=False,
            use_route_boost=False,
            use_entity_boost=False,
            use_chunk_lexical_score=True,
            use_chunk_structure_boost=False,
            use_doc_rank_boost=False,
        ),
        RetrieverOptions(
            router_mode="rule",
            use_dense=True,
            use_bm25=True,
            use_routing=True,
            use_route_boost=True,
            use_entity_boost=False,
            use_chunk_lexical_score=True,
            use_chunk_structure_boost=False,
            use_doc_rank_boost=False,
        ),
        RetrieverOptions(
            router_mode="rule",
            use_dense=True,
            use_bm25=True,
            use_routing=True,
            use_route_boost=True,
            use_entity_boost=True,
            use_chunk_lexical_score=True,
            use_chunk_structure_boost=True,
            use_doc_rank_boost=True,
        ),
        RetrieverOptions(
            router_mode="ollama",
            use_dense=True,
            use_bm25=True,
            use_routing=True,
            use_route_boost=True,
            use_entity_boost=False,
            use_chunk_lexical_score=True,
            use_chunk_structure_boost=False,
            use_doc_rank_boost=False,
        ),
        RetrieverOptions(
            router_mode="ollama",
            use_dense=True,
            use_bm25=True,
            use_routing=True,
            use_route_boost=True,
            use_entity_boost=True,
            use_chunk_lexical_score=True,
            use_chunk_structure_boost=True,
            use_doc_rank_boost=True,
        ),
    ]

    if is_bge_m3:
        options.extend(
            [
                RetrieverOptions(
                    router_mode="off",
                    use_dense=True,
                    use_bm25=False,
                    use_routing=False,
                    use_route_boost=False,
                    use_entity_boost=False,
                    use_chunk_lexical_score=False,
                    use_chunk_structure_boost=False,
                    use_doc_rank_boost=False,
                    use_bge_sparse=True,
                    use_bge_colbert=False,
                ),
                RetrieverOptions(
                    router_mode="off",
                    use_dense=True,
                    use_bm25=True,
                    use_routing=False,
                    use_route_boost=False,
                    use_entity_boost=False,
                    use_chunk_lexical_score=True,
                    use_chunk_structure_boost=False,
                    use_doc_rank_boost=False,
                    use_bge_sparse=True,
                    use_bge_colbert=False,
                ),
                RetrieverOptions(
                    router_mode="off",
                    use_dense=True,
                    use_bm25=False,
                    use_routing=False,
                    use_route_boost=False,
                    use_entity_boost=False,
                    use_chunk_lexical_score=False,
                    use_chunk_structure_boost=False,
                    use_doc_rank_boost=False,
                    use_bge_sparse=True,
                    use_bge_colbert=True,
                ),
                RetrieverOptions(
                    router_mode="rule",
                    use_dense=True,
                    use_bm25=True,
                    use_routing=True,
                    use_route_boost=True,
                    use_entity_boost=True,
                    use_chunk_lexical_score=True,
                    use_chunk_structure_boost=True,
                    use_doc_rank_boost=True,
                    use_bge_sparse=True,
                    use_bge_colbert=False,
                ),
                RetrieverOptions(
                    router_mode="ollama",
                    use_dense=True,
                    use_bm25=True,
                    use_routing=True,
                    use_route_boost=True,
                    use_entity_boost=True,
                    use_chunk_lexical_score=True,
                    use_chunk_structure_boost=True,
                    use_doc_rank_boost=True,
                    use_bge_sparse=True,
                    use_bge_colbert=False,
                ),
                RetrieverOptions(
                    router_mode="ollama",
                    use_dense=True,
                    use_bm25=True,
                    use_routing=True,
                    use_route_boost=True,
                    use_entity_boost=True,
                    use_chunk_lexical_score=True,
                    use_chunk_structure_boost=True,
                    use_doc_rank_boost=True,
                    use_bge_sparse=True,
                    use_bge_colbert=True,
                ),
            ]
        )

    return prune_meaningless_options(model_name, options)


def build_meaningful_options_for_model(model_name: str) -> List[RetrieverOptions]:
    is_bge_m3 = model_name == "embedding_bge_m3"
    options: List[RetrieverOptions] = []

    retrieval_families = [
        {"use_dense": False, "use_bm25": True, "use_bge_sparse": False, "use_bge_colbert": False},
        {"use_dense": True, "use_bm25": False, "use_bge_sparse": False, "use_bge_colbert": False},
        {"use_dense": True, "use_bm25": True, "use_bge_sparse": False, "use_bge_colbert": False},
    ]
    if is_bge_m3:
        retrieval_families.extend(
            [
                {"use_dense": True, "use_bm25": False, "use_bge_sparse": True, "use_bge_colbert": False},
                {"use_dense": True, "use_bm25": True, "use_bge_sparse": True, "use_bge_colbert": False},
                {"use_dense": True, "use_bm25": False, "use_bge_sparse": True, "use_bge_colbert": True},
                {"use_dense": True, "use_bm25": True, "use_bge_sparse": True, "use_bge_colbert": True},
            ]
        )

    router_modes: List[RouterMode] = ["off", "rule", "ollama"]
    boost_profiles = [
        {
            "use_entity_boost": False,
            "use_chunk_lexical_score": False,
            "use_chunk_structure_boost": False,
            "use_doc_rank_boost": False,
        },
        {
            "use_entity_boost": True,
            "use_chunk_lexical_score": True,
            "use_chunk_structure_boost": False,
            "use_doc_rank_boost": False,
        },
        {
            "use_entity_boost": True,
            "use_chunk_lexical_score": True,
            "use_chunk_structure_boost": True,
            "use_doc_rank_boost": True,
        },
    ]

    for family in retrieval_families:
        for router_mode in router_modes:
            for boost_profile in boost_profiles:
                options.append(
                    RetrieverOptions(
                        router_mode=router_mode,
                        use_dense=family["use_dense"],
                        use_bm25=family["use_bm25"],
                        use_routing=router_mode != "off",
                        use_route_boost=router_mode != "off",
                        use_entity_boost=boost_profile["use_entity_boost"],
                        use_chunk_lexical_score=boost_profile["use_chunk_lexical_score"],
                        use_chunk_structure_boost=boost_profile["use_chunk_structure_boost"],
                        use_doc_rank_boost=boost_profile["use_doc_rank_boost"],
                        use_bge_sparse=family["use_bge_sparse"],
                        use_bge_colbert=family["use_bge_colbert"],
                    )
                )

    return prune_meaningless_options(model_name, options)


def prune_meaningless_options(model_name: str, options: List[RetrieverOptions]) -> List[RetrieverOptions]:
    is_bge_m3 = model_name == "embedding_bge_m3"
    pruned: Dict[str, RetrieverOptions] = {}
    for option in options:
        normalized = option.normalized(model_name)
        if not normalized.requires_any_retrieval_signal():
            continue
        if normalized.use_bge_sparse and not is_bge_m3:
            continue
        if normalized.use_bge_colbert and not is_bge_m3:
            continue
        if normalized.use_bge_colbert and not normalized.use_dense:
            continue
        if normalized.use_bge_sparse and not normalized.use_dense:
            continue
        if normalized.router_mode == "off" and normalized.use_route_boost:
            continue
        if normalized.use_chunk_lexical_score and not (normalized.use_bm25 or normalized.use_bge_sparse):
            continue
        if normalized.use_chunk_structure_boost and not normalized.use_routing:
            continue
        if normalized.use_doc_rank_boost and not (normalized.use_dense or normalized.use_bm25 or normalized.use_bge_sparse):
            continue
        if normalized.use_entity_boost is False and normalized.use_route_boost is False and normalized.use_chunk_structure_boost is True:
            continue
        pruned[normalized.slug()] = normalized
    return list(pruned.values())

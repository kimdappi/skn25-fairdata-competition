from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_name: str
    header: str
    section: str
    content: str
    tokens: tuple[str, ...]


def _iter_hybrid_files(data_dir: Path) -> Iterable[Path]:
    return sorted(data_dir.glob("*_hybrid.json"))


def load_corpus(data_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in _iter_hybrid_files(data_dir):
        records = json.loads(path.read_text(encoding="utf-8"))
        doc_name = path.name.replace("_hybrid.json", "")
        for record in records:
            metadata = record.get("metadata", {})
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                continue
            header = str(metadata.get("Header", "")).strip()
            section = str(metadata.get("section", "")).strip()
            content = str(record.get("page_content", "")).strip()
            joined = " ".join(part for part in (doc_name, header, section, content) if part)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_name=doc_name,
                    header=header,
                    section=section,
                    content=content,
                    tokens=tuple(tokenize(joined)),
                )
            )
    if not chunks:
        raise RuntimeError(f"No hybrid corpus files found in {data_dir}")
    return chunks

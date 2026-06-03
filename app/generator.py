from __future__ import annotations

from app.corpus import Chunk


class GroundedGenerator:
    def generate(self, question: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "질문과 관련된 근거 청크를 찾지 못했습니다."

        lines: list[str] = []
        for chunk in chunks[:3]:
            for raw_line in chunk.content.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                lines.append(
                    f"{line} (근거: {chunk.chunk_id}, 문서: {chunk.doc_name})"
                )
                break

        if not lines:
            lines.append(
                f"관련 문서는 {chunks[0].doc_name}이며, 주요 근거 청크는 {chunks[0].chunk_id}입니다."
            )

        answer_lines = [
            "질문과 가장 관련도가 높은 의결서 청크를 기준으로 정리한 답변입니다.",
            *lines,
        ]
        return "\n".join(answer_lines)

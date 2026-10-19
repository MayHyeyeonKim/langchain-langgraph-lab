"""Optional local RAG utilities, introduced after the core workflow is understood."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedContext:
    text: str
    source: str
    page: int | None = None


def build_context(chunks: list[RetrievedContext], max_characters: int = 6_000) -> str:
    """Create a bounded, source-labelled context block for a future agent prompt."""
    selected: list[str] = []
    total = 0
    for chunk in chunks:
        labelled = f"[source: {chunk.source}{f', page {chunk.page}' if chunk.page else ''}]\n{chunk.text}"
        if total + len(labelled) > max_characters:
            break
        selected.append(labelled)
        total += len(labelled)
    return "\n\n".join(selected)

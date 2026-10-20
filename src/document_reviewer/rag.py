"""Local RAG indexing and retrieval backed by Chroma and Ollama embeddings."""

from dataclasses import dataclass
from pathlib import Path

from .schemas import SourceReference
from .settings import settings


class RagUnavailableError(RuntimeError):
    """Raised when optional RAG dependencies or the embedding service are unavailable."""


@dataclass(frozen=True)
class ReferenceDocument:
    text: str
    source: str
    page: int | None = None


@dataclass(frozen=True)
class RetrievedContext:
    text: str
    source: str
    page: int | None = None


def build_context(chunks: list[RetrievedContext], max_characters: int = 6_000) -> str:
    """Create a bounded, source-labelled context block for an agent prompt."""
    selected: list[str] = []
    total = 0
    for chunk in chunks:
        location = f", page {chunk.page}" if chunk.page else ""
        labelled = f"[source: {chunk.source}{location}]\n{chunk.text}"
        if total + len(labelled) > max_characters:
            break
        selected.append(labelled)
        total += len(labelled)
    return "\n\n".join(selected)


class LocalRagStore:
    """Persistent local reference store. Imports are lazy so core review needs no RAG packages."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        try:
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings
        except ImportError as error:
            raise RagUnavailableError("Install RAG dependencies with: uv sync --group rag") from error

        embeddings = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        self.store = Chroma(
            collection_name=collection_name or settings.rag_collection_name,
            persist_directory=str(persist_directory or settings.rag_persist_directory),
            embedding_function=embeddings,
        )

    def index(self, references: list[ReferenceDocument]) -> list[str]:
        """Chunk and persist references; returns Chroma IDs for traceability."""
        if not references:
            return []
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=150)
        documents: list[Document] = []
        for reference in references:
            metadata = {"source": reference.source}
            if reference.page is not None:
                metadata["page"] = reference.page
            documents.extend(splitter.create_documents([reference.text], metadatas=[metadata]))
        return self.store.add_documents(documents)

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedContext]:
        documents = self.store.similarity_search(query, k=k)
        return [
            RetrievedContext(
                text=document.page_content,
                source=str(document.metadata.get("source", "unknown")),
                page=document.metadata.get("page"),
            )
            for document in documents
        ]


def source_references(chunks: list[RetrievedContext]) -> list[SourceReference]:
    """Deduplicate source/page pairs for the public workflow response."""
    seen: set[tuple[str, int | None]] = set()
    sources: list[SourceReference] = []
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            sources.append(SourceReference(source=chunk.source, page=chunk.page))
            seen.add(key)
    return sources

"""Role-specific LangChain agents. Pydantic keeps their hand-offs predictable."""

from collections.abc import Callable
from typing import Protocol, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .llm import get_chat_model
from .schemas import AnalysisReport, ReviewResult

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class DocumentAgents(Protocol):
    def analyze(self, document: str, context: str = "") -> AnalysisReport: ...
    def rewrite(self, document: str, analysis: AnalysisReport, review: ReviewResult | None = None) -> str: ...
    def review(self, original: str, revised: str, context: str = "") -> ReviewResult: ...


class OllamaDocumentAgents:
    """Three narrowly scoped agents sharing one local ChatOllama instance."""

    def __init__(self, model_factory: Callable = get_chat_model) -> None:
        self.model = model_factory()

    def _structured(self, schema: type[StructuredModel], system: str, prompt: str) -> StructuredModel:
        structured_model = self.model.with_structured_output(schema)
        result = structured_model.invoke([SystemMessage(system), HumanMessage(prompt)])
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)

    def analyze(self, document: str, context: str = "") -> AnalysisReport:
        return self._structured(
            AnalysisReport,
            "You are a precise technical-document analyzer. Find concrete problems in clarity, "
            "completeness, accuracy, structure, and usability. Do not invent facts.",
            f"Analyze this technical document:\n\n{document}\n\nReference context:\n{context or 'None provided.'}",
        )

    def rewrite(
        self, document: str, analysis: AnalysisReport, review: ReviewResult | None = None
    ) -> str:
        feedback = review.model_dump_json() if review else "No previous review; address the analysis."
        response = self.model.invoke(
            [
                SystemMessage(
                    "You are a technical writer. Return only a complete revised document in Markdown. "
                    "Preserve accurate information; do not claim unprovided facts."
                ),
                HumanMessage(
                    f"Original document:\n{document}\n\nAnalysis:\n{analysis.model_dump_json()}"
                    f"\n\nReviewer feedback:\n{feedback}"
                ),
            ]
        )
        return str(response.content).strip()

    def review(self, original: str, revised: str, context: str = "") -> ReviewResult:
        return self._structured(
            ReviewResult,
            "You are a strict technical-document reviewer. Approve only if the revised document "
            "is clearer, complete enough for its stated purpose, and does not add unsupported claims.",
            f"Original:\n{original}\n\nRevised:\n{revised}\n\nReference context:\n{context or 'None provided.'}",
        )

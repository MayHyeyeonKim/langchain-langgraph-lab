"""Role-specific LangChain agents. Pydantic keeps their hand-offs predictable."""

import json
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
        contracts = {
            AnalysisReport: {
                "summary": "string",
                "issues": [
                    {
                        "title": "string",
                        "severity": "low | medium | high",
                        "evidence": "string",
                        "recommendation": "string",
                    }
                ],
            },
            ReviewResult: {
                "approved": "boolean",
                "summary": "string",
                "remaining_issues": [
                    {
                        "title": "string",
                        "severity": "low | medium | high",
                        "evidence": "string",
                        "recommendation": "string",
                    }
                ],
            },
        }
        contract = json.dumps(contracts[schema])
        structured_model = self.model.with_structured_output(schema, method="json_mode")
        result = structured_model.invoke(
            [
                SystemMessage(
                    f"{system}\n\nReturn only one JSON object with this exact shape: {contract}"
                ),
                HumanMessage(prompt),
            ]
        )
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)

    def analyze(self, document: str, context: str = "") -> AnalysisReport:
        return self._structured(
            AnalysisReport,
            "You are a precise technical-document analyzer. Find only concrete documentation "
            "problems in clarity, completeness, accuracy, structure, and usability. Do not infer "
            "or speculate about the software itself. Every issue's evidence must be a verbatim "
            "quote from the document; report an empty issues list if none are supported. Return at "
            "most two issues and keep every value concise. For a README, assess installation/setup "
            "instructions and API usage details when those topics are relevant.",
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
                    "Keep it under 200 words. Preserve accurate information; do not claim unprovided facts. "
                    "Do not add endpoint paths, HTTP methods, request bodies, dependencies, prerequisites, "
                    "or response formats unless they appear in the original document or supplied feedback. "
                    "When needed details are unknown, state that they are not specified rather than inventing them."
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
            "is clearer, complete enough for its stated purpose, and does not add unsupported claims. "
            "Reject any endpoint path, HTTP method, request body, dependency, prerequisite, response format, "
            "or other factual detail that is absent from both the original document and reference context. "
            "Every remaining issue must quote the unsupported or incomplete text verbatim.",
            f"Original:\n{original}\n\nRevised:\n{revised}\n\nReference context:\n{context or 'None provided.'}",
        )

"""Role-specific LangChain agents. Pydantic keeps their hand-offs predictable."""

import json
import re
from collections.abc import Callable
from typing import Protocol, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .llm import get_chat_model
from .schemas import AnalysisReport, Issue, ReviewResult, Severity

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)
INLINE_CODE_PATTERN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
OUTER_MARKDOWN_FENCE_PATTERN = re.compile(
    r"^```(?:markdown|md)\s*\n(?P<document>.*)\n```$", re.DOTALL | re.IGNORECASE
)


class DocumentAgents(Protocol):
    def analyze(self, document: str, context: str = "") -> AnalysisReport: ...
    def rewrite(
        self,
        document: str,
        analysis: AnalysisReport,
        review: ReviewResult | None = None,
        context: str = "",
    ) -> str: ...
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
        self,
        document: str,
        analysis: AnalysisReport,
        review: ReviewResult | None = None,
        context: str = "",
    ) -> str:
        feedback = review.model_dump_json() if review else "No previous review; address the analysis."
        response = self.model.invoke(
            [
                SystemMessage(
                    "You are a conservative technical copy editor. Return only a complete revised document "
                    "in Markdown, under 200 words. Preserve accurate information; do not add claims. "
                    "Use only facts and inline-code snippets that occur verbatim in the original document "
                    "or labelled reference context. "
                    "Reviewer feedback identifies problems but never supplies new facts. Do not add endpoint paths, "
                    "HTTP methods, request bodies, dependencies, prerequisites, commands, or response formats. "
                    "When needed details are unknown, add a plain-text TODO requesting that documentation instead."
                ),
                HumanMessage(
                    f"Original document:\n{document}\n\nAnalysis:\n{analysis.model_dump_json()}"
                    f"\n\nReviewer feedback:\n{feedback}"
                    f"\n\nReference context:\n{context or 'None provided.'}"
                ),
            ]
        )
        revised = str(response.content).strip()
        fence_match = OUTER_MARKDOWN_FENCE_PATTERN.fullmatch(revised)
        return fence_match.group("document").strip() if fence_match else revised

    def review(self, original: str, revised: str, context: str = "") -> ReviewResult:
        result = self._structured(
            ReviewResult,
            "You are a strict technical-document reviewer. Approve only if the revised document "
            "is clearer, complete enough for its stated purpose, and does not add unsupported claims. "
            "Reject any endpoint path, HTTP method, request body, dependency, prerequisite, response format, "
            "or other factual detail that is absent from both the original document and reference context. "
            "Every remaining issue must quote the unsupported or incomplete text verbatim.",
            f"Original:\n{original}\n\nRevised:\n{revised}\n\nReference context:\n{context or 'None provided.'}",
        )
        issues = [*result.remaining_issues, *self._unsupported_code_issues(original, revised, context)]
        if not issues:
            return result
        summary = result.summary
        if result.approved:
            summary = f"{summary} Approval changed to false because unresolved issues remain."
        return result.model_copy(update={"approved": False, "summary": summary, "remaining_issues": issues})

    @staticmethod
    def _unsupported_code_issues(original: str, revised: str, context: str) -> list[Issue]:
        known_text = f"{original}\n{context}"
        unsupported = {
            literal.strip()
            for literal in INLINE_CODE_PATTERN.findall(revised)
            if literal.strip() and literal.strip() not in known_text
        }
        return [
            Issue(
                title="Unsupported code detail",
                severity=Severity.HIGH,
                evidence=f"`{literal}`",
                recommendation="Remove this detail or provide it in the original document or reference context.",
            )
            for literal in sorted(unsupported)
        ]

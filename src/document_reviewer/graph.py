"""LangGraph orchestration for analyze -> write -> review with bounded retries."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import DocumentAgents, OllamaDocumentAgents
from .schemas import AnalysisReport, ReviewResult, SourceReference, WorkflowResult
from .settings import settings


class ReviewState(TypedDict, total=False):
    original_document: str
    reference_context: str
    sources: list[SourceReference]
    analysis: AnalysisReport
    revised_document: str
    review: ReviewResult
    revision_count: int
    completed: bool
    failure_reason: str | None


class ReviewWorkflow:
    """A testable workflow; pass fake agents in tests and Ollama agents in production."""

    def __init__(self, agents: DocumentAgents | None = None, max_retries: int | None = None) -> None: # TypeScript의 constructor()와 같은 역할
        self.agents = agents or OllamaDocumentAgents()
        self.max_retries = max_retries if max_retries is not None else settings.max_retries
        self.graph = self._build()

    def _build(self):
        builder = StateGraph(ReviewState)
        builder.add_node("analyze", self._analyze)
        builder.add_node("write", self._write)
        builder.add_node("review_document", self._review)
        builder.add_node("mark_incomplete", self._mark_incomplete)
        builder.add_edge(START, "analyze")
        builder.add_edge("analyze", "write")
        builder.add_edge("write", "review_document")
        builder.add_conditional_edges(
            "review_document",
            self._next_after_review,
            {"approved": END, "retry": "write", "incomplete": "mark_incomplete"},
        )
        builder.add_edge("mark_incomplete", END)
        return builder.compile()

    def _analyze(self, state: ReviewState) -> ReviewState:
        return {
            "analysis": self.agents.analyze(
                state["original_document"], state.get("reference_context", "")
            )
        }

    def _write(self, state: ReviewState) -> ReviewState:
        previous_review = state.get("review")
        return {
            "revised_document": self.agents.rewrite(
                state["original_document"],
                state["analysis"],
                previous_review,
                state.get("reference_context", ""),
            ),
            "revision_count": state.get("revision_count", 0) + 1,
        }

    def _review(self, state: ReviewState) -> ReviewState:
        result = self.agents.review(
            state["original_document"], state["revised_document"], state.get("reference_context", "")
        )
        return {"review": result, "completed": result.approved}

    def _next_after_review(self, state: ReviewState) -> str:
        if state["review"].approved:
            return "approved"
        if state["revision_count"] < self.max_retries:
            return "retry"
        return "incomplete"

    def _mark_incomplete(self, state: ReviewState) -> ReviewState:
        return {
            "completed": False,
            "failure_reason": f"Review did not pass after {state['revision_count']} revision(s).",
        }

    def run(
        self,
        document: str,
        reference_context: str = "",
        sources: list[SourceReference] | None = None,
    ) -> WorkflowResult:
        state = self.graph.invoke(
            {
                "original_document": document,
                "reference_context": reference_context,
                "sources": sources or [],
                "revision_count": 0,
            }
        )
        return WorkflowResult(
            original_document=state["original_document"],
            analysis=state["analysis"],
            revised_document=state["revised_document"],
            review=state["review"],
            revision_count=state["revision_count"],
            completed=state.get("completed", False),
            failure_reason=state.get("failure_reason"),
            sources=state.get("sources", []),
        )

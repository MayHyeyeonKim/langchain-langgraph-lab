from document_reviewer.graph import ReviewWorkflow
from document_reviewer.schemas import AnalysisReport, Issue, ReviewResult, Severity


class FakeAgents:
    def __init__(self, approvals: list[bool]) -> None:
        self.approvals = iter(approvals)
        self.writes = 0

    def analyze(self, document: str, context: str = "") -> AnalysisReport:
        return AnalysisReport(
            summary="One missing instruction.",
            issues=[Issue(title="Missing setup", severity=Severity.MEDIUM, evidence="Run", recommendation="Add command")],
        )

    def rewrite(self, document: str, analysis: AnalysisReport, review: ReviewResult | None = None) -> str:
        self.writes += 1
        return f"{document}\n\nRevision {self.writes}"

    def review(self, original: str, revised: str, context: str = "") -> ReviewResult:
        approved = next(self.approvals)
        return ReviewResult(approved=approved, summary="ok" if approved else "Needs work")


def test_workflow_retries_then_approves() -> None:
    agents = FakeAgents([False, True])
    result = ReviewWorkflow(agents=agents, max_retries=2).run("Document")
    assert result.completed is True
    assert result.revision_count == 2
    assert agents.writes == 2


def test_workflow_stops_after_retry_limit() -> None:
    result = ReviewWorkflow(agents=FakeAgents([False, False]), max_retries=2).run("Document")
    assert result.completed is False
    assert result.revision_count == 2
    assert result.failure_reason == "Review did not pass after 2 revision(s)."

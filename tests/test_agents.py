from types import SimpleNamespace

from document_reviewer.agents import OllamaDocumentAgents
from document_reviewer.schemas import AnalysisReport, Issue, ReviewResult, Severity


class RecordingModel:
    def __init__(self, content: str = "# Revised document") -> None:
        self.messages = []
        self.content = content

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.content)


class RecordingStructuredModel(RecordingModel):
    def __init__(self, result: ReviewResult | None = None) -> None:
        super().__init__()
        self.result = result or ReviewResult(approved=False, summary="Unsupported claim")

    def with_structured_output(self, schema, *, method):
        assert schema is ReviewResult
        assert method == "json_mode"
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.result


def test_writer_prompt_prohibits_unsupported_document_details() -> None:
    model = RecordingModel()
    agents = OllamaDocumentAgents(model_factory=lambda: model)

    revised = agents.rewrite("Original", AnalysisReport(summary="Needs more detail"))

    assert revised == "# Revised document"
    system_prompt = model.messages[0].content
    assert "Do not add endpoint paths" in system_prompt
    assert "dependencies" in system_prompt


def test_writer_removes_an_outer_markdown_fence() -> None:
    model = RecordingModel("```markdown\n# Revised document\n\nRun `python app.py`.\n```")
    agents = OllamaDocumentAgents(model_factory=lambda: model)

    revised = agents.rewrite("Original", AnalysisReport(summary="Needs more detail"))

    assert revised == "# Revised document\n\nRun `python app.py`."


def test_reviewer_prompt_rejects_unsupported_claims() -> None:
    model = RecordingStructuredModel()
    agents = OllamaDocumentAgents(model_factory=lambda: model)

    result = agents.review("Original", "Revised")

    assert result.approved is False
    system_prompt = model.messages[0].content
    assert "Reject any endpoint path" in system_prompt
    assert "absent from both the original document and reference context" in system_prompt


def test_reviewer_rejects_new_inline_code_and_inconsistent_approval() -> None:
    model = RecordingStructuredModel(
        ReviewResult(
            approved=True,
            summary="Looks good",
            remaining_issues=[
                Issue(
                    title="Incomplete API details",
                    severity=Severity.MEDIUM,
                    evidence="Endpoint details are not specified.",
                    recommendation="Add endpoint details.",
                )
            ],
        )
    )
    agents = OllamaDocumentAgents(model_factory=lambda: model)

    result = agents.review(
        "Run `python app.py`.",
        "Run `python app.py`. Install with `pip install -r requirements.txt`. Call `GET /tasks`.",
    )

    assert result.approved is False
    assert "Approval changed to false" in result.summary
    assert {issue.evidence for issue in result.remaining_issues} == {
        "Endpoint details are not specified.",
        "`GET /tasks`",
        "`pip install -r requirements.txt`",
    }

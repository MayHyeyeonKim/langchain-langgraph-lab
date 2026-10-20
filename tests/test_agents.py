from types import SimpleNamespace

from document_reviewer.agents import OllamaDocumentAgents
from document_reviewer.schemas import AnalysisReport, ReviewResult


class RecordingModel:
    def __init__(self) -> None:
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content="# Revised document")


class RecordingStructuredModel(RecordingModel):
    def with_structured_output(self, schema, *, method):
        assert schema is ReviewResult
        assert method == "json_mode"
        return self

    def invoke(self, messages):
        self.messages = messages
        return ReviewResult(approved=False, summary="Unsupported claim")


def test_writer_prompt_prohibits_unsupported_document_details() -> None:
    model = RecordingModel()
    agents = OllamaDocumentAgents(model_factory=lambda: model)

    revised = agents.rewrite("Original", AnalysisReport(summary="Needs more detail"))

    assert revised == "# Revised document"
    system_prompt = model.messages[0].content
    assert "Do not add endpoint paths" in system_prompt
    assert "dependencies" in system_prompt


def test_reviewer_prompt_rejects_unsupported_claims() -> None:
    model = RecordingStructuredModel()
    agents = OllamaDocumentAgents(model_factory=lambda: model)

    result = agents.review("Original", "Revised")

    assert result.approved is False
    system_prompt = model.messages[0].content
    assert "Reject any endpoint path" in system_prompt
    assert "absent from both the original document and reference context" in system_prompt

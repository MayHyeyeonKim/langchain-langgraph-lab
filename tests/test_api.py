import sys
from pathlib import Path
from unittest.mock import patch

import django
from django.test import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT / "src"))
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()


def test_api_rejects_missing_document() -> None:
    response = Client().post("/api/reviews/", {})
    assert response.status_code == 400


@patch("reviews.views.ReviewWorkflow")
def test_api_accepts_text(mock_workflow) -> None:
    from document_reviewer.schemas import AnalysisReport, ReviewResult, WorkflowResult

    mock_workflow.return_value.run.return_value = WorkflowResult(
        original_document="text",
        analysis=AnalysisReport(summary="Fine"),
        revised_document="text",
        review=ReviewResult(approved=True, summary="Fine"),
        revision_count=1,
        completed=True,
    )
    response = Client().post("/api/reviews/", {"text": "text", "max_retries": 0})
    assert response.status_code == 200
    assert response.json()["completed"] is True
    mock_workflow.assert_called_once_with(max_retries=0)

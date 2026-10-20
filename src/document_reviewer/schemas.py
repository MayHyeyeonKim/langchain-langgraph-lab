"""Validated data exchanged between agents and API clients."""

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Issue(BaseModel):
    title: str = Field(min_length=1, description="Short label for the problem")
    severity: Severity
    evidence: str = Field(min_length=1, description="Quote or precise location from the document")
    recommendation: str = Field(min_length=1)


class AnalysisReport(BaseModel):
    summary: str = Field(min_length=1)
    issues: list[Issue] = Field(default_factory=list)


class ReviewResult(BaseModel):
    approved: bool
    summary: str = Field(min_length=1)
    remaining_issues: list[Issue] = Field(default_factory=list)


class SourceReference(BaseModel): # Source Attribution(출처 표시)
    source: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


class WorkflowResult(BaseModel):
    original_document: str
    analysis: AnalysisReport
    revised_document: str
    review: ReviewResult
    revision_count: int
    completed: bool
    failure_reason: str | None = None
    sources: list[SourceReference] = Field(default_factory=list) 

"""
M6 — Structured review output schema.

All LLM review responses must conform to these Pydantic models.
`ReviewOutput` is used as the JSON response schema for Gemini's structured-
output mode and as the internal data model that flows from generation →
citation validation → markdown rendering.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class GraphCitation(BaseModel):
    """A reference to a graph entity that supports a review finding.

    `entity_uid` MUST exist in the valid entity UID set extracted from the
    current GraphRAG context.  UIDs that are not in that set are hallucinated
    and will be flagged by `validate_graph_citations`.
    """

    entity_uid: str
    entity_name: str
    entity_file: str
    relevance: str


class SimilarCodeCitation(BaseModel):
    """A reference to semantically similar code found via vector search."""

    file: str
    line: int
    similarity_score: float = 0.0
    snippet: str = ""


class Evidence(BaseModel):
    """Supporting evidence for a Finding, drawn from GraphRAG and vector search."""

    graph_refs: List[GraphCitation] = Field(default_factory=list)
    similar_code_refs: List[SimilarCodeCitation] = Field(default_factory=list)


class Finding(BaseModel):
    """A single review finding (bug, security issue, code-quality problem, etc.)."""

    severity: str           # "critical" | "high" | "medium" | "low"
    category: str           # "security" | "bug" | "performance" | "quality" | "testing"
    title: str
    file: str
    line: Optional[int] = None
    description: str
    suggestion: str
    evidence: Evidence = Field(default_factory=Evidence)
    # Set to False by validate_graph_citations() when graph_refs contain
    # UIDs that were not present in the input payload (hallucination).
    verified: bool = True


class CritiqueFinding(BaseModel):
    """Reflection-node assessment of a single Finding."""

    finding_title: str
    action: str = "keep"   # "keep" | "strengthen" | "downgrade" | "drop"
    reasoning: str = ""


class CritiqueOutput(BaseModel):
    """
    M8 — Output of the reflection node.

    The reflection node audits a ReviewOutput for hallucinated citations,
    over-flagged style nits, and missed critical issues, then returns this
    structured critique which the revise node uses to produce the final review.
    """

    overall_quality: str = "acceptable"   # "good" | "acceptable" | "poor"
    critique_findings: List[CritiqueFinding] = Field(default_factory=list)
    summary: str = ""


class ReviewOutput(BaseModel):
    """Top-level structured output from a code review LLM call."""

    summary: str
    findings: List[Finding] = Field(default_factory=list)
    overall_assessment: str = "comment"    # "approve" | "request_changes" | "comment"
    risk_level: str = "low"                # "low" | "medium" | "high" | "critical"

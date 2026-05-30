"""
Review Formatter for GitHub Comments (Phase 5.2)

This module provides markdown formatting for AI-generated code reviews to be posted on GitHub.

Design Decisions:
- Use GitHub-flavored Markdown (GFM)
- Support collapsible sections for long reviews
- Add emojis for visual clarity
- Include severity badges
- Format code blocks with syntax highlighting
- Support both summary reviews and inline comments

Key Features:
- Review summary with sections (Security, Performance, etc.)
- Issue severity badges (🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low)
- Collapsible details for verbose sections
- Code snippets with line numbers
- Actionable recommendations
- Statistics and metrics
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from .review_schema import ReviewOutput


# ========================================================================
# ENUMS & DATA CLASSES
# ========================================================================

class Severity(str, Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewEvent(str, Enum):
    """GitHub review event types"""
    COMMENT = "COMMENT"  # General comment
    APPROVE = "APPROVE"  # Approve PR
    REQUEST_CHANGES = "REQUEST_CHANGES"  # Request changes


@dataclass
class Issue:
    """Represents a code issue found in review"""
    severity: Severity
    title: str
    description: str
    file: str
    line: Optional[int] = None
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class ReviewSection:
    """A section in the review (Security, Performance, etc.)"""
    title: str
    content: str
    issues: List[Issue]
    collapsible: bool = False


@dataclass
class ReviewSummary:
    """Complete review summary"""
    overall_assessment: str
    review_strategy: str  # quick, standard, deep
    model_used: str  # gemini-flash, gemini-pro
    sections: List[ReviewSection]
    statistics: Dict[str, Any]
    recommendations: List[str]
    event_type: ReviewEvent


# ========================================================================
# EMOJI & BADGE CONSTANTS
# ========================================================================

SEVERITY_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🟢",
    Severity.INFO: "ℹ️",
}

SEVERITY_BADGE = {
    Severity.CRITICAL: "![Critical](https://img.shields.io/badge/CRITICAL-critical-critical)",
    Severity.HIGH: "![High](https://img.shields.io/badge/HIGH-high-orange)",
    Severity.MEDIUM: "![Medium](https://img.shields.io/badge/MEDIUM-medium-yellow)",
    Severity.LOW: "![Low](https://img.shields.io/badge/LOW-low-green)",
    Severity.INFO: "![Info](https://img.shields.io/badge/INFO-info-blue)",
}

SECTION_EMOJI = {
    "security": "🔒",
    "performance": "⚡",
    "bugs": "🐛",
    "quality": "✨",
    "testing": "🧪",
    "documentation": "📚",
    "architecture": "🏗️",
    "maintainability": "🔧",
}


# ========================================================================
# REVIEW FORMATTER
# ========================================================================

class ReviewFormatter:
    """
    Formats AI-generated reviews into GitHub-flavored Markdown
    
    Features:
    - Professional layout with sections
    - Severity badges and emojis
    - Collapsible details
    - Code highlighting
    - Statistics summary
    
    Usage:
        formatter = ReviewFormatter()
        markdown = formatter.format_review(review_summary)
        await github_client.post_review_comment(body=markdown)
    """
    
    def __init__(self):
        self.max_line_length = 120  # Max line length for code snippets
        self.max_issues_shown = 10  # Max issues to show before collapsing
    
    def format_review(self, review: ReviewSummary) -> str:
        """
        Format complete review as markdown
        
        Args:
            review: ReviewSummary with all review data
            
        Returns:
            str: GitHub-flavored Markdown
        """
        parts = []
        
        # Header
        parts.append(self._format_header(review))
        
        # Overall assessment
        parts.append(self._format_assessment(review.overall_assessment, review.event_type))
        
        # Statistics
        if review.statistics:
            parts.append(self._format_statistics(review.statistics))
        
        # Sections (Security, Performance, etc.)
        for section in review.sections:
            parts.append(self._format_section(section))
        
        # Recommendations
        if review.recommendations:
            parts.append(self._format_recommendations(review.recommendations))
        
        # Footer
        parts.append(self._format_footer(review))
        
        return "\n\n".join(parts)
    
    def _format_header(self, review: ReviewSummary) -> str:
        """Format review header with AI branding"""
        return f"""# 🤖 AI Code Review

> Generated using **{review.model_used}** with **{review.strategy_name(review.review_strategy)}** strategy"""
    
    def _format_assessment(self, assessment: str, event_type: ReviewEvent) -> str:
        """Format overall assessment with event icon"""
        icons = {
            ReviewEvent.APPROVE: "✅",
            ReviewEvent.REQUEST_CHANGES: "⚠️",
            ReviewEvent.COMMENT: "💬",
        }
        
        icon = icons.get(event_type, "💬")
        
        return f"""## {icon} Overall Assessment

{assessment}"""
    
    def _format_statistics(self, stats: Dict[str, Any]) -> str:
        """Format review statistics"""
        lines = [
            "## 📊 Review Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        
        # Common statistics
        stat_map = {
            "files_reviewed": ("Files Reviewed", lambda x: f"{x} files"),
            "total_additions": ("Lines Added", lambda x: f"+{x}"),
            "total_deletions": ("Lines Deleted", lambda x: f"-{x}"),
            "critical_issues": ("Critical Issues", lambda x: f"🔴 {x}"),
            "high_issues": ("High Priority", lambda x: f"🟠 {x}"),
            "medium_issues": ("Medium Priority", lambda x: f"🟡 {x}"),
            "low_issues": ("Low Priority", lambda x: f"🟢 {x}"),
            "complexity_score": ("Avg Complexity", lambda x: f"{x}/100"),
            "review_time": ("Review Time", lambda x: f"{x}s"),
        }
        
        for key, (label, formatter) in stat_map.items():
            if key in stats:
                lines.append(f"| {label} | {formatter(stats[key])} |")
        
        return "\n".join(lines)
    
    def _format_section(self, section: ReviewSection) -> str:
        """Format a review section"""
        emoji = SECTION_EMOJI.get(section.title.lower(), "📝")
        
        parts = [
            f"## {emoji} {section.title}",
            "",
            section.content,
        ]
        
        # Add issues if present
        if section.issues:
            parts.append("")
            parts.append(self._format_issues(section.issues, section.collapsible))
        
        return "\n".join(parts)
    
    def _format_issues(self, issues: List[Issue], collapsible: bool = False) -> str:
        """Format issues list"""
        if not issues:
            return ""
        
        # Sort by severity
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        sorted_issues = sorted(issues, key=lambda i: severity_order[i.severity])
        
        # Check if we need to collapse
        should_collapse = collapsible and len(issues) > self.max_issues_shown
        
        if should_collapse:
            visible_issues = sorted_issues[:self.max_issues_shown]
            hidden_issues = sorted_issues[self.max_issues_shown:]
            
            parts = ["### Issues Found", ""]
            
            # Visible issues
            for issue in visible_issues:
                parts.append(self._format_single_issue(issue))
            
            # Collapsible section for remaining issues
            parts.append("")
            parts.append("<details>")
            parts.append(f"<summary>Show {len(hidden_issues)} more issues</summary>")
            parts.append("")
            
            for issue in hidden_issues:
                parts.append(self._format_single_issue(issue))
            
            parts.append("")
            parts.append("</details>")
            
            return "\n".join(parts)
        else:
            parts = ["### Issues Found", ""]
            for issue in sorted_issues:
                parts.append(self._format_single_issue(issue))
            
            return "\n".join(parts)
    
    def _format_single_issue(self, issue: Issue) -> str:
        """Format a single issue"""
        emoji = SEVERITY_EMOJI[issue.severity]
        badge = SEVERITY_BADGE[issue.severity]
        
        parts = [
            f"#### {emoji} {issue.title}",
            "",
            f"{badge}",
            "",
        ]
        
        # File and line
        if issue.line:
            parts.append(f"**Location:** [`{issue.file}:{issue.line}`]({issue.file}#L{issue.line})")
        else:
            parts.append(f"**Location:** `{issue.file}`")
        
        parts.append("")
        
        # Description
        parts.append(f"**Issue:** {issue.description}")
        parts.append("")
        
        # Code snippet if provided
        if issue.code_snippet:
            parts.append("<details>")
            parts.append("<summary>View code</summary>")
            parts.append("")
            parts.append("```python")  # TODO: Detect language
            parts.append(issue.code_snippet)
            parts.append("```")
            parts.append("")
            parts.append("</details>")
            parts.append("")
        
        # Suggestion if provided
        if issue.suggestion:
            parts.append(f"**Suggestion:** {issue.suggestion}")
            parts.append("")
        
        parts.append("---")
        parts.append("")
        
        return "\n".join(parts)
    
    def _format_recommendations(self, recommendations: List[str]) -> str:
        """Format recommendations list"""
        parts = [
            "## 💡 Recommendations",
            "",
        ]
        
        for i, rec in enumerate(recommendations, 1):
            parts.append(f"{i}. {rec}")
        
        return "\n".join(parts)
    
    def _format_footer(self, review: ReviewSummary) -> str:
        """Format review footer"""
        return f"""---

<sub>Generated by AI Code Review System | Strategy: {review.review_strategy} | Model: {review.model_used}</sub>"""
    
    # ====================================================================
    # INLINE COMMENT FORMATTING
    # ====================================================================
    
    # ----------------------------------------------------------------
    # M6 — Structured review renderer (ReviewOutput → GFM Markdown)
    # ----------------------------------------------------------------

    def format_structured_review(self, review: "ReviewOutput") -> str:
        """
        Convert a ReviewOutput Pydantic model into GitHub-flavored Markdown.

        Findings are grouped by severity (critical → high → medium → low).
        Unverified findings (hallucinated graph citations) receive a warning
        badge so reviewers can weigh them accordingly.
        """
        from .review_schema import ReviewOutput as _RO  # noqa: F401

        _ASSESS_ICON = {
            "approve": "✅",
            "request_changes": "⚠️",
            "comment": "💬",
        }
        _SEV_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        _SEV_BADGE = {
            "critical": "![Critical](https://img.shields.io/badge/CRITICAL-critical-critical)",
            "high":     "![High](https://img.shields.io/badge/HIGH-high-orange)",
            "medium":   "![Medium](https://img.shields.io/badge/MEDIUM-medium-yellow)",
            "low":      "![Low](https://img.shields.io/badge/LOW-low-green)",
        }
        _SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        parts: List[str] = []

        # ── Header ──────────────────────────────────────────────────
        icon = _ASSESS_ICON.get(review.overall_assessment, "💬")
        parts.append("# 🤖 AI Code Review\n")
        parts.append(f"## {icon} Overall Assessment\n\n{review.summary}\n")

        # ── Statistics ──────────────────────────────────────────────
        severity_counts: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in review.findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

        unverified = sum(1 for f in review.findings if not f.verified)
        stat_rows = [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Risk Level | {review.risk_level.upper()} |",
            f"| 🔴 Critical | {severity_counts['critical']} |",
            f"| 🟠 High | {severity_counts['high']} |",
            f"| 🟡 Medium | {severity_counts['medium']} |",
            f"| 🟢 Low | {severity_counts['low']} |",
        ]
        if unverified:
            stat_rows.append(f"| ⚠️ Unverified citations | {unverified} finding(s) |")
        parts.append("## 📊 Review Statistics\n\n" + "\n".join(stat_rows) + "\n")

        # ── Findings ────────────────────────────────────────────────
        if not review.findings:
            parts.append("## ✅ No Issues Found\n\nCode looks good!\n")
        else:
            sorted_findings = sorted(
                review.findings,
                key=lambda f: _SEV_ORDER.get(f.severity, 99),
            )
            current_sev: Optional[str] = None
            finding_lines: List[str] = []

            for finding in sorted_findings:
                sev = finding.severity
                if sev != current_sev:
                    emoji = _SEV_EMOJI.get(sev, "📝")
                    finding_lines.append(f"\n## {emoji} {sev.capitalize()} Issues\n")
                    current_sev = sev

                badge = _SEV_BADGE.get(sev, "")
                unverified_note = (
                    "\n> ⚠️ **Unverified** — one or more graph citations were not found "
                    "in the GraphRAG payload and may be hallucinated. Review this finding "
                    "independently.\n"
                    if not finding.verified else ""
                )

                loc = (
                    f"[`{finding.file}:{finding.line}`]({finding.file}#L{finding.line})"
                    if finding.line else f"`{finding.file}`"
                )

                graph_evidence = ""
                if finding.evidence.graph_refs:
                    refs = "\n".join(
                        f"  - `{r.entity_name}` in `{r.entity_file}` — {r.relevance}"
                        for r in finding.evidence.graph_refs
                    )
                    graph_evidence = f"\n**Graph Evidence:**\n{refs}\n"

                similar_evidence = ""
                if finding.evidence.similar_code_refs:
                    srefs = "\n".join(
                        f"  - `{r.file}:{r.line}` (similarity: {r.similarity_score:.2f})"
                        for r in finding.evidence.similar_code_refs
                    )
                    similar_evidence = f"\n**Similar Code:**\n{srefs}\n"

                finding_lines.append(
                    f"### {badge} {finding.title}\n"
                    f"\n**Location:** {loc}"
                    f"\n**Category:** {finding.category}"
                    f"{unverified_note}"
                    f"\n\n**Issue:** {finding.description}"
                    f"\n\n**Suggestion:** {finding.suggestion}"
                    f"{graph_evidence}"
                    f"{similar_evidence}"
                    "\n---\n"
                )

            parts.append("\n".join(finding_lines))

        # ── Footer ──────────────────────────────────────────────────
        parts.append(
            "\n---\n<sub>Generated by Graph Bug AI · Structured review (M6)</sub>"
        )
        return "\n\n".join(parts)

    def format_inline_comment(self, issue: Issue) -> str:
        """
        Format issue as inline comment for specific line
        
        Args:
            issue: Issue to format
            
        Returns:
            str: Markdown for inline comment
        """
        emoji = SEVERITY_EMOJI[issue.severity]
        badge = SEVERITY_BADGE[issue.severity]
        
        parts = [
            f"### {emoji} {issue.title}",
            "",
            f"{badge}",
            "",
            issue.description,
        ]
        
        if issue.suggestion:
            parts.append("")
            parts.append("**Suggested fix:**")
            parts.append("")
            parts.append(f"```suggestion")
            parts.append(issue.suggestion)
            parts.append("```")
        
        return "\n".join(parts)
    
    # ====================================================================
    # UTILITY METHODS
    # ====================================================================
    
    @staticmethod
    def strategy_name(strategy: str) -> str:
        """Get human-readable strategy name"""
        names = {
            "quick": "Quick Review",
            "standard": "Standard Review",
            "deep": "Deep Analysis"
        }
        return names.get(strategy, strategy.title())
    
    @staticmethod
    def truncate_code(code: str, max_lines: int = 20) -> str:
        """Truncate code to max lines"""
        lines = code.split("\n")
        if len(lines) <= max_lines:
            return code
        
        visible_lines = lines[:max_lines]
        return "\n".join(visible_lines) + f"\n\n... ({len(lines) - max_lines} more lines)"


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def parse_workflow_output_to_review(workflow_state: Dict) -> ReviewSummary:
    """
    Convert workflow output to ReviewSummary
    
    This bridges Phase 4 (workflow) and Phase 5 (GitHub posting)
    
    Args:
        workflow_state: Final state from LangGraph workflow
        
    Returns:
        ReviewSummary ready for formatting
    """
    pr_context = workflow_state.get("pr_context", {})
    
    # Parse issues from context
    critical_issues = _parse_issues(pr_context.get("critical_issues", []), Severity.CRITICAL)
    high_issues = _parse_issues(pr_context.get("high_issues", []), Severity.HIGH)
    medium_issues = _parse_issues(pr_context.get("medium_issues", []), Severity.MEDIUM)
    
    # Create sections
    sections = []
    
    # Security section (if critical/high security issues exist)
    security_issues = [i for i in critical_issues + high_issues if "security" in i.description.lower()]
    if security_issues:
        sections.append(ReviewSection(
            title="Security Analysis",
            content="Critical security vulnerabilities detected that require immediate attention.",
            issues=security_issues,
            collapsible=len(security_issues) > 5
        ))
    
    # General issues section
    all_issues = critical_issues + high_issues + medium_issues
    if all_issues:
        sections.append(ReviewSection(
            title="Code Quality Issues",
            content=f"Found {len(all_issues)} issues requiring attention.",
            issues=all_issues,
            collapsible=len(all_issues) > 10
        ))
    
    # Determine event type based on issues
    if critical_issues or len(high_issues) >= 3:
        event_type = ReviewEvent.REQUEST_CHANGES
    elif not all_issues:
        event_type = ReviewEvent.APPROVE
    else:
        event_type = ReviewEvent.COMMENT
    
    # Statistics
    statistics = {
        "files_reviewed": pr_context.get("total_files", 0),
        "total_additions": pr_context.get("total_additions", 0),
        "total_deletions": pr_context.get("total_deletions", 0),
        "critical_issues": len(critical_issues),
        "high_issues": len(high_issues),
        "medium_issues": len(medium_issues),
        "complexity_score": _calculate_avg_complexity(pr_context.get("files", [])),
    }
    
    # Recommendations
    recommendations = pr_context.get("recommendations", [])
    
    return ReviewSummary(
        overall_assessment=workflow_state.get("overall_summary", "Review completed."),
        review_strategy=workflow_state.get("review_strategy", "standard"),
        model_used=workflow_state.get("selected_model", "gemini-flash"),
        sections=sections,
        statistics=statistics,
        recommendations=recommendations,
        event_type=event_type
    )


def _parse_issues(raw_issues: List[Dict], severity: Severity) -> List[Issue]:
    """Parse raw issues from workflow to Issue objects"""
    issues = []
    for raw in raw_issues:
        issues.append(Issue(
            severity=severity,
            title=raw.get("title", "Issue found"),
            description=raw.get("description", ""),
            file=raw.get("file", "unknown"),
            line=raw.get("line"),
            suggestion=raw.get("suggestion"),
            code_snippet=raw.get("code_snippet")
        ))
    return issues


def _calculate_avg_complexity(files: List[Dict]) -> int:
    """Calculate average complexity score across files"""
    if not files:
        return 0
    
    total = sum(f.get("complexity_score", 0) for f in files)
    return int(total / len(files))


# ========================================================================
# EXAMPLE USAGE
# ========================================================================

if __name__ == "__main__":
    # Example: Create sample review
    review = ReviewSummary(
        overall_assessment="This PR introduces several security vulnerabilities and code quality issues that need to be addressed before merging.",
        review_strategy="deep",
        model_used="gemini-1.5-pro",
        sections=[
            ReviewSection(
                title="Security Analysis",
                content="Found 2 critical security vulnerabilities that must be fixed.",
                issues=[
                    Issue(
                        severity=Severity.CRITICAL,
                        title="SQL Injection Vulnerability",
                        description="User input is directly interpolated into SQL query without sanitization.",
                        file="auth.py",
                        line=45,
                        suggestion="Use parameterized queries instead of string interpolation.",
                        code_snippet='query = f"SELECT * FROM users WHERE email = \'{email}\'"'
                    ),
                    Issue(
                        severity=Severity.HIGH,
                        title="Sensitive Data Exposure",
                        description="API keys are logged in plain text.",
                        file="config.py",
                        line=23,
                        suggestion="Remove sensitive data from logs or mask it."
                    )
                ]
            ),
            ReviewSection(
                title="Code Quality",
                content="Several code quality issues detected.",
                issues=[
                    Issue(
                        severity=Severity.MEDIUM,
                        title="High Cyclomatic Complexity",
                        description="Function has complexity of 45, making it hard to test and maintain.",
                        file="payment.py",
                        line=100,
                        suggestion="Refactor into smaller functions."
                    )
                ]
            )
        ],
        statistics={
            "files_reviewed": 5,
            "total_additions": 250,
            "total_deletions": 50,
            "critical_issues": 1,
            "high_issues": 1,
            "medium_issues": 3,
            "complexity_score": 42,
        },
        recommendations=[
            "Fix critical security vulnerabilities before merging",
            "Add unit tests for payment processing logic",
            "Consider refactoring high-complexity functions"
        ],
        event_type=ReviewEvent.REQUEST_CHANGES
    )
    
    # Format as markdown
    formatter = ReviewFormatter()
    markdown = formatter.format_review(review)
    
    print(markdown)

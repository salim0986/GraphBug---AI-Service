"""
M12 — Inline PR comment poster.

Posts `Finding` objects from a `ReviewOutput` as GitHub inline review comments.
Each finding that has a `line` number and is `verified` becomes a line-level
comment on the PR.

Duplicate detection: fetches existing bot comments on the PR and skips any
(file, line) pair that already has a Graph Bug AI Reviewer comment.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING, List

from .logger import setup_logger
from .review_schema import Finding, ReviewOutput

if TYPE_CHECKING:
    from .github_client import GitHubClient

logger = setup_logger(__name__)

_BOT_SIGNATURE = "Graph Bug AI Reviewer"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finding_key(finding: Finding) -> str:
    """Stable 16-char deduplication key for a finding."""
    raw = f"{finding.file}:{finding.line}:{finding.category}:{finding.title[:50]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _format_inline_body(finding: Finding) -> str:
    """Render a Finding as a GitHub inline comment body."""
    sev_emoji = {
        "critical": "🔴",
        "high":     "🟠",
        "medium":   "🟡",
        "low":      "🟢",
    }.get(finding.severity, "📝")

    lines = [
        f"{sev_emoji} **{finding.severity.upper()} — {finding.title}**",
        "",
        finding.description,
    ]
    if finding.suggestion:
        lines += ["", f"**Suggestion:** {finding.suggestion}"]
    if not finding.verified:
        lines += ["", "_⚠️ Note: graph citation could not be verified._"]
    lines += ["", f"_— {_BOT_SIGNATURE}_"]
    return "\n".join(lines)


def _existing_bot_keys(comments: list) -> set[str]:
    """Return the set of 'file:line' keys already commented by our bot."""
    keys: set[str] = set()
    for c in comments:
        if _BOT_SIGNATURE not in c.get("body", ""):
            continue
        path = c.get("path", "")
        line = c.get("line") or c.get("original_line")
        if path and line:
            keys.add(f"{path}:{line}")
    return keys


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def post_inline_findings(
    github_client: "GitHubClient",
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
    installation_id: int,
    findings: List[Finding],
    max_comments: int = 20,
) -> int:
    """
    Post inline GitHub review comments for findings with a line number.

    Skips:
    - Findings with no ``line`` set
    - Unverified findings (``finding.verified == False``)
    - Findings that duplicate an existing bot comment at the same file+line

    Returns the count of comments successfully posted.
    """
    if not findings:
        return 0

    eligible = [f for f in findings if f.line is not None and f.verified]
    if not eligible:
        logger.info("[M12] No eligible findings for inline comments")
        return 0

    # Fetch existing comments to avoid duplicates
    existing_keys: set[str] = set()
    try:
        existing = await github_client.get_pull_request_comments(
            repo_full_name, pr_number, installation_id
        )
        existing_keys = _existing_bot_keys(existing)
        if existing_keys:
            logger.info(f"[M12] {len(existing_keys)} existing bot comment(s) found — skipping duplicates")
    except Exception as exc:
        logger.warning(f"[M12] Could not fetch existing PR comments: {exc}")

    posted = 0
    for finding in eligible[:max_comments]:
        key = f"{finding.file}:{finding.line}"
        if key in existing_keys:
            logger.debug(f"[M12] Skipping duplicate {key}")
            continue

        body = _format_inline_body(finding)
        try:
            await github_client.post_inline_comment(
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                body=body,
                path=finding.file,
                line=finding.line,
                commit_id=head_sha,
                installation_id=installation_id,
                side="RIGHT",
            )
            existing_keys.add(key)
            posted += 1
            logger.info(
                f"[M12] Inline comment posted: {finding.file}:{finding.line} "
                f"[{finding.severity}] {finding.title[:40]}"
            )
            await asyncio.sleep(0.3)
        except Exception as exc:
            logger.warning(f"[M12] Failed to post inline comment at {key}: {exc}")

    logger.info(f"[M12] Posted {posted}/{len(eligible)} inline comment(s)")
    return posted


async def post_review_inline_comments(
    github_client: "GitHubClient",
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
    installation_id: int,
    final_state: dict,
    max_comments: int = 20,
) -> int:
    """
    Convenience wrapper: extract `ReviewOutput` from workflow `final_state`
    and post inline comments.  Returns 0 gracefully on any error.
    """
    try:
        struct = final_state.get("structured_review") or final_state.get("revised_review")
        if not struct:
            logger.debug("[M12] No structured_review in final_state — skipping inline comments")
            return 0
        review = ReviewOutput.model_validate(struct)
        return await post_inline_findings(
            github_client=github_client,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            head_sha=head_sha,
            installation_id=installation_id,
            findings=review.findings,
            max_comments=max_comments,
        )
    except Exception as exc:
        logger.error(f"[M12] post_review_inline_comments failed: {exc}", exc_info=True)
        return 0

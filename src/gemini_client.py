"""
Google Gemini API Client - Phase 4.3
Handles authentication, model selection, and API calls to Gemini
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, AsyncIterator, TYPE_CHECKING
import os
import re
import json
import asyncio
from dataclasses import dataclass
from google import genai
from google.genai import types
from .logger import setup_logger

if TYPE_CHECKING:
    from .review_schema import ReviewOutput
    from .llm_client import LLMClient

logger = setup_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GeminiConfig:
    """Configuration for Gemini API"""
    api_key: str
    
    # Model names
    flash_lite_model: str = "gemini-2.5-flash-lite"
    flash_model: str = "gemini-2.5-flash"
    pro_model: str = "gemini-2.5-pro"
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    
    # Rate limiting
    max_requests_per_minute: int = 60
    retry_attempts: int = 3
    retry_delay: float = 2.0
    
    # Safety settings
    enable_safety_filters: bool = True


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

class PromptTemplates:
    """Comprehensive prompt templates for code reviews"""
    
    SYSTEM_PROMPT = """You are an expert code reviewer with deep knowledge of software engineering best practices, security, performance, and maintainability.

**Your Task:**
Provide thorough, actionable code reviews with context-aware insights.

**Guidelines:**
- **Be context-aware**: Reference similar code patterns when suggesting improvements
- **Consider dependencies**: Highlight impact on related files and functions
- **Be constructive and specific** with actionable feedback
- **Prioritize**: Security vulnerabilities > Critical bugs > Performance > Quality
- **Suggest concrete improvements** with code examples
- **Identify refactoring opportunities** (consolidate duplicates, extract common patterns)
- **Format professionally** with markdown, severity badges, and clear sections

**Focus Areas:**
1. 🔒 Security vulnerabilities and dependency impact
2. 🐛 Logic errors and bugs
3. ⚡ Performance issues and optimization opportunities
4. 🏗️ Architecture and design patterns
5. 🔄 Code duplication and refactoring opportunities
6. 📝 Documentation and maintainability
7. ✅ Testing coverage recommendations
"""
    
    QUICK_REVIEW_PROMPT = """## Quick Scan: {pr_title}

**Scope:** {total_files} files | +{additions}/-{deletions}

**GraphRAG:** Entities: {entities} | Dependencies: {dependencies} | Similar: {similar_code}

**Files:** {files_summary}
**Issues:** {issues_summary}

---

## Find Problems Only

Identify actual mistakes:
🔴 Critical (security, crashes)
🟠 High (bugs, logic errors)
🟡 Medium (code smells)

Cite line numbers (L45+). Reference GraphRAG if data provided (not "None"). No generic advice.

**Output:**
### Issues
[List or "✅ None"]

### GraphRAG Matches  
[Cite 1-2 if available]
"""
    
    QUICK_SCAN_PROMPT = """## 🔎 PHASE 3: Quick Security & Critical Issue Scan

**Purpose:** Fast scan to identify CRITICAL issues and security vulnerabilities ONLY.

**File:** {filename}
**Language:** {language}
**Changes:** +{additions} -{deletions}

**Diff with Line Numbers:**
```
{diff}
```

## 🎯 YOUR TASK: Scan for CRITICAL Issues ONLY

**Focus Areas (in priority order):**

1. **🔒 Security Vulnerabilities**
   - SQL injection
   - XSS vulnerabilities
   - Authentication/authorization bypasses
   - Credential exposure
   - Insecure deserialization

2. **💥 Critical Bugs**
   - Null pointer/undefined access
   - Resource leaks (memory, connections, files)
   - Race conditions
   - Data corruption risks

3. **🚫 Immediate Blockers**
   - Breaking API changes without migration
   - Data loss scenarios
   - Production outage risks

## ⌛ SPEED REQUIREMENTS:
- This is a QUICK scan - complete in <5 seconds
- Skip non-critical issues (quality, style, minor optimizations)
- Only flag issues that are:
  - **Urgent** (must fix before merge)
  - **High impact** (security, data integrity, availability)
  - **Well-founded** (not hypothetical)

## 📝 OUTPUT FORMAT:

If critical issues found:
```
🔴 **CRITICAL ISSUES DETECTED**

🔴 **L<line>+/-**: <Brief issue title>
- **Problem:** <What's wrong with evidence from diff>
- **Impact:** <Why this is critical>
- **Fix:** <Quick suggestion>
```

If NO critical issues:
```
✅ **No critical issues detected in quick scan**

Proceed to detailed review for code quality, performance, and best practices.
```

**IMPORTANT:**
- Cite line numbers using L<num>+/- format from the diff
- Be specific - reference actual code from the diff
- Don't flag minor issues or style problems
- If unsure whether issue is critical, skip it (detailed review will catch it)
"""
    
    STANDARD_REVIEW_PROMPT = """## Standard Code Review Request

Review this pull request thoroughly.

**PR Title:** {pr_title}
**Description:** {description}

**Changes Overview:**
- Files: {total_files}
- Additions: {additions} lines
- Deletions: {deletions} lines
- Languages: {languages}
- Risk Level: {risk_level}

**Issues Detected:**
- Critical: {critical_count}
- High: {high_count}
- Medium: {medium_count}

{critical_issues}

**Impact Analysis:**
- Affected callers: {affected_callers}
- Complexity hotspots: {complexity_hotspots}
- High coupling: {coupling_files}

**Files to Review:**
{files_details}

Provide a comprehensive review including:
1. Security Analysis
2. Logic and Correctness
3. Performance Considerations
4. Code Quality
5. Testing Recommendations
6. Overall Assessment

Be specific and reference line numbers where applicable.
"""
    
    DEEP_REVIEW_PROMPT = """## Issue-Focused Code Review with GraphRAG Context

**PR:** {pr_title}
**Scope:** {total_files} files | +{additions}/-{deletions} | Risk: {risk_level}

---

## Context from GraphRAG Analysis

### 🔗 Similar Code Patterns in Codebase
{similar_code}

### 📦 Dependencies & Impact
{dependencies}

### 🏗️ Key Entities & Relationships
{entities}

### 🐛 Static Analysis Pre-Scan
- Critical Issues: {critical_issues}
- High Priority Issues: {high_issues}

---

## Files Changed (with code snippets)

{files_details}

---

## YOUR TASK: Find Issues with Evidence

You are a senior code reviewer. For EACH issue you find:

1. **Cite the line number**: `L45+` (added), `L67-` (removed)
2. **Show the problematic code**: Include the actual code snippet
3. **Explain the issue**: What's wrong and why it matters
4. **Reference GraphRAG**: If similar patterns exist in the codebase, cite them with code
5. **Suggest a fix**: Show how to fix it (code example when possible)

**CRITICAL**: You MUST include actual code snippets in your review. Do not just cite line numbers.

---

## Required Output Format

### 🔴 CRITICAL
[If found:]

**L<line>+**: <Issue title>

**Code:**
```
<actual problematic code from the diff>
```

**Issue**: <Detailed explanation of the problem>

**Evidence**: <GraphRAG reference with similar code if applicable>

**Fix**: <How to fix it, with code example if possible>

---

### 🟠 HIGH  
[Same format as above]

### 🟡 MEDIUM
[Same format as above]

### ✅ No Issues Found
[If no issues in a category, state "None - code looks good"]

---

## GraphRAG Requirement

If similar code, dependencies, or entities are provided above (not "None"), you MUST cite at least 3-5 in your findings with actual code comparisons.

**Example of GOOD usage:**

❌ **BAD**: "SQL injection at L45"

✅ **GOOD**: 
"**L45+**: SQL injection vulnerability

**Code:**
```python
query = "SELECT * FROM users WHERE id=" + user_input
```

**Issue**: Direct string concatenation enables SQL injection attacks

**Evidence**: Similar vulnerable pattern in auth.py:L234 (GraphRAG similarity: 0.89):
```python
query = f"SELECT * FROM sessions WHERE token={{user_token}}"
```
Both use unsafe string operations instead of parameterized queries.

**Fix**:
```python
query = "SELECT * FROM users WHERE id=?"
db.execute(query, [user_input])
```"

---

**Focus**: Find bugs, security issues, and mistakes. Include code context for EVERY issue.
"""
    
    FILE_REVIEW_PROMPT = """## File Review: {filename}

**Changes:** +{additions}/-{deletions} | **Language:** {language}

---

## Code Diff (with line numbers)

```{language}
{diff}
```

---

## GraphRAG Context for This File

### Similar Code in Codebase
{similar_code}

### Dependencies & Impact
{dependencies}

### Related Entities
{entities}

### Pre-Scan Issues Found
{issues}

---

## Find Issues with Code Context

Review ONLY the changes shown above. For EACH issue:

1. **Cite line number**: `L45+` or `L67-`
2. **Show the code**: Include the actual code snippet
3. **Explain the problem**: What's wrong and why
4. **Reference GraphRAG**: Cite similar patterns with code if available
5. **Suggest fix**: Provide code example when possible

**CRITICAL**: Include actual code snippets, not just line numbers.

---

## Required Format

### CRITICAL
[If found:]

**L<line>+**: <Issue title>

**Code:**
```{language}
<actual code>
```

**Issue**: <Explanation>
**Evidence**: <GraphRAG reference with code if applicable>
**Fix**: <Solution with code>

---

### HIGH
[Same format]

### MEDIUM
[Same format]

### No Issues Found
[If clean: "No issues found - code looks good"]

---

**GraphRAG Usage**: If similar code/dependencies are provided (not "None"), cite them with actual code comparisons.
"""
    
    # ----------------------------------------------------------------
    # M6 — Grounded structured prompts (JSON output + citation rules)
    # ----------------------------------------------------------------

    STRUCTURED_REVIEW_PROMPT = """## Code Review — Structured JSON Output

**PR:** {pr_title}
**Files:** {total_files} | +{additions}/-{deletions} | Risk: {risk_level}

### Valid GraphRAG Entity UIDs
Cite ONLY entity UIDs from this list.  Do NOT invent UIDs.
{entity_uids_block}

### GraphRAG Context

**Similar Code Patterns:**
{similar_code}

**Dependencies & Impact:**
{dependencies}

**Key Entities & Relationships:**
{entities}

### Static Analysis Pre-scan
{issues_summary}

### Files Changed
{files_details}

---

<reasoning>
Reason step-by-step before writing JSON (this block does not appear in output):
1. Security — SQL injection? hardcoded secrets? auth bypass? XSS? insecure deserialisation?
2. Bugs — null/undefined deref? wrong logic? off-by-one? resource leak? race condition?
3. Performance — N+1 queries? unbounded loops? blocking I/O in async context?
4. Quality — dead code? magic numbers? overly complex functions? missing error handling?
5. For each finding: which entity UID from the list is GENUINELY relevant?
   Include graph_refs only when the entity directly relates to the issue.
   When in doubt, leave graph_refs empty — an empty list is correct.
</reasoning>

Return ONLY valid JSON — no markdown fences, no explanation text outside the object:

{{{{
  "summary": "One sentence PR summary",
  "overall_assessment": "approve|request_changes|comment",
  "risk_level": "low|medium|high|critical",
  "findings": [
    {{{{
      "severity": "critical|high|medium|low",
      "category": "security|bug|performance|quality|testing",
      "title": "Short descriptive title",
      "file": "path/to/file.py",
      "line": 42,
      "description": "Detailed explanation — cite the exact line and code",
      "suggestion": "Specific fix with code example if possible",
      "evidence": {{{{
        "graph_refs": [
          {{{{
            "entity_uid": "EXACT_UID_FROM_LIST_ABOVE",
            "entity_name": "function_name",
            "entity_file": "path/to/file.py",
            "relevance": "Why this entity is relevant to the finding"
          }}}}
        ],
        "similar_code_refs": []
      }}}}
    }}}}
  ]
}}}}

GOOD citation:  entity_uid: "repo1::src/auth.py::validate_token"  (exact match from list above)
BAD citation:   entity_uid: "repo1::invented::fake_function"       (fabricated — NEVER do this)
"""

    STRUCTURED_FILE_PROMPT = """## File Review — Structured JSON Output

**File:** {filename}
**Language:** {language}
**Changes:** +{additions}/-{deletions}

### Valid GraphRAG Entity UIDs (cite ONLY these)
{entity_uids_block}

### Code Diff
```{language}
{diff}
```

### GraphRAG Context
**Similar Code:** {similar_code}
**Dependencies:** {dependencies}
**Related Entities:** {entities}

### Pre-scan Issues
{issues}

---

<reasoning>
Review ONLY the lines shown in the diff above. Reason before writing JSON:
1. What security, bug, or quality issues appear in the changed lines specifically?
2. Which entity UIDs from the list are directly relevant to each finding?
3. Assign severity: critical=sec breach/data loss, high=definite bug, medium=smell/risk, low=style.
</reasoning>

Return ONLY valid JSON — no markdown fences:

{{{{
  "summary": "One line file review summary",
  "overall_assessment": "approve|request_changes|comment",
  "risk_level": "low|medium|high|critical",
  "findings": [
    {{{{
      "severity": "critical|high|medium|low",
      "category": "security|bug|performance|quality|testing",
      "title": "Short descriptive title",
      "file": "{filename}",
      "line": 42,
      "description": "Detailed explanation with code reference",
      "suggestion": "Specific fix",
      "evidence": {{{{ "graph_refs": [], "similar_code_refs": [] }}}}
    }}}}
  ]
}}}}
"""

    AGGREGATION_PROMPT = """## Summary Review

**PR:** {pr_title}
**Files:** {files_count} | **Total Issues:** {total_issues}

**Breakdown:**
\ud83d\udd34 Critical: {critical_count}
\ud83d\udfe0 High: {high_count}
\ud83d\udfe1 Medium: {medium_count}

**Individual Reviews:**
{file_reviews}

---

## Create Brief Summary

Group by severity. Preserve GraphRAG insights from individual reviews (dependencies, similar code).

**Output:**

### \ud83d\udd34 Critical Issues Summary
[List key issues across files or "None"]

### \ud83d\udfe0 High Priority Summary
[List or "None"]

### \ud83d\udfe1 Medium Priority Summary
[List or "None"]

### Overall Assessment
[Approve / Request Changes / Comment with brief rationale]

Format as a clear, well-structured PR review comment. DO NOT discard GraphRAG context.
"""


# ============================================================================
# GEMINI CLIENT
# ============================================================================

class GeminiClient:
    """
    Client for Google Gemini API with rate limiting and error handling
    """
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or self._load_config()
        self.client = self._configure_api()
        self.templates = PromptTemplates()
        self._request_times: List[float] = []
    
    # ----------------------------------------------------------------
    # M7 — multi-provider routing
    # ----------------------------------------------------------------

    def set_llm_client(self, llm_client: "LLMClient") -> None:
        """
        Inject an LLMClient (M7).  When set, all generation is delegated to
        LLMClient instead of calling the google-genai SDK directly.  This
        makes the client provider-agnostic at runtime while preserving full
        backward compatibility for callers that never call this method.
        """
        self._llm_client: "LLMClient" = llm_client

    @staticmethod
    def _model_name_to_tier(model_name: str) -> str:
        """Map a concrete model name string to a tier label (flash/pro/thinking)."""
        lower = model_name.lower()
        if "thinking" in lower or "opus" in lower or "o1" == lower:
            return "thinking"
        if "pro" in lower or "sonnet" in lower or "large" in lower:
            return "pro"
        return "flash"

    def _load_config(self) -> GeminiConfig:
        """Load configuration from environment"""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable required"
            )
        
        return GeminiConfig(api_key=api_key)
    
    def _configure_api(self):
        """Configure the Gemini API client"""
        client = genai.Client(api_key=self.config.api_key)
        logger.info("Gemini API configured successfully")
        return client
    
    def select_model(
        self,
        total_files: int,
        total_additions: int,
        risk_level: str,
        review_strategy: str
    ) -> str:
        """
        Select appropriate Gemini model based on PR characteristics
        
        Model Selection Logic:
        - flash-lite: Quick reviews, small PRs (< 3 files, < 100 lines)
        - flash: Standard reviews, medium PRs (< 10 files, < 500 lines)
        - pro: Deep reviews, large/complex PRs or high risk
        """
        if review_strategy == "quick" and total_files <= 3 and total_additions <= 100:
            model = self.config.flash_lite_model
            logger.info(f"Selected {model} for quick review")
        
        elif review_strategy == "deep" or risk_level in ["critical", "high"]:
            model = self.config.pro_model
            logger.info(f"Selected {model} for deep/high-risk review")
        
        elif total_files > 10 or total_additions > 500:
            model = self.config.pro_model
            logger.info(f"Selected {model} for large PR")
        
        else:
            model = self.config.flash_model
            logger.info(f"Selected {model} for standard review")
        
        return model
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting for API calls"""
        import time
        
        now = time.time()
        # Keep only requests from the last minute
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self.config.max_requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                # Clean up old requests after waiting
                now = time.time()
                self._request_times = [t for t in self._request_times if now - t < 60]
        
        self._request_times.append(now)
    
    def _get_generation_config(self) -> types.GenerateContentConfig:
        """Get generation configuration"""
        return types.GenerateContentConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_output_tokens=self.config.max_output_tokens,
        )
    
    def _get_safety_settings(self) -> List[types.SafetySetting]:
        """Get safety settings"""
        if not self.config.enable_safety_filters:
            return [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE"
                ),
            ]
        
        return [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
        ]
    
    async def generate_review(
        self,
        model_name: str,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate code review using Gemini
        
        Args:
            model_name: Gemini model to use
            prompt: Review prompt
            system_instruction: System instruction (optional)
        
        Returns:
            Generated review text
        """
        # M7: delegate to LLMClient when a provider-agnostic client is injected
        if hasattr(self, "_llm_client") and self._llm_client is not None:
            tier = self._model_name_to_tier(model_name)
            return await self._llm_client.generate(tier, prompt)

        await self._enforce_rate_limit()

        system = system_instruction or self.templates.SYSTEM_PROMPT

        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(f"Generating review with {model_name} (attempt {attempt + 1})")
                
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_name,
                    contents=prompt,
                    config=self._get_generation_config()
                )
                
                if response.text:
                    logger.info(f"Review generated successfully ({len(response.text)} chars)")
                    return response.text
                else:
                    logger.warning("Empty response from Gemini")
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    return "Unable to generate review at this time."
                
            except Exception as e:
                logger.error(f"Error generating review (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max retries exceeded")
                    raise
    
    async def generate_structured_review(
        self,
        model_name: str,
        prompt: str,
    ) -> "ReviewOutput":
        """
        Generate a code review as a validated Pydantic ReviewOutput.

        Uses Gemini's JSON output mode (`response_mime_type="application/json"` +
        `response_schema`) when available.  Falls back to text-based JSON
        extraction if the API or model does not support schema-constrained output.

        Returns a ReviewOutput even on partial failure — callers can inspect
        `review.findings` to determine quality.
        """
        from .review_schema import ReviewOutput

        # M7: delegate to LLMClient when a provider-agnostic client is injected
        if hasattr(self, "_llm_client") and self._llm_client is not None:
            tier = self._model_name_to_tier(model_name)
            result = await self._llm_client.generate_structured(tier, prompt, ReviewOutput)
            return result  # type: ignore[return-value]

        await self._enforce_rate_limit()

        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(
                    f"[M6] Generating structured review with {model_name} "
                    f"(attempt {attempt + 1})"
                )
                json_config = types.GenerateContentConfig(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    max_output_tokens=self.config.max_output_tokens,
                    response_mime_type="application/json",
                    response_schema=ReviewOutput,
                )
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_name,
                    contents=prompt,
                    config=json_config,
                )
                if response.text:
                    logger.info(
                        f"[M6] Structured review generated ({len(response.text)} chars)"
                    )
                    return self._parse_review_json(response.text)

                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue

            except Exception as e:
                logger.warning(f"[M6] Structured review attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    logger.error("[M6] Max retries exceeded for structured review")

        return ReviewOutput(
            summary="Review generation failed after retries.",
            overall_assessment="comment",
            risk_level="low",
        )

    def _parse_review_json(self, text: str) -> "ReviewOutput":
        """
        Parse a JSON string into a ReviewOutput.

        Strips markdown code fences if present (some models wrap JSON in
        ```json ... ``` even when asked not to).  Returns a minimal ReviewOutput
        on any parse error so the caller always gets a usable object.
        """
        from .review_schema import ReviewOutput

        text = text.strip()
        # Strip leading ```json or ``` fence if the model added one
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
            return ReviewOutput.model_validate(data)
        except Exception as e:
            logger.warning(f"[M6] JSON parse failed: {e}. Returning fallback ReviewOutput.")
            # Preserve the raw text as the summary so we don't lose the content
            return ReviewOutput(
                summary=text[:500] if text else "Review generation produced unparseable output.",
                overall_assessment="comment",
                risk_level="low",
            )

    async def generate_structured_schema(
        self,
        model_name: str,
        prompt: str,
        schema: Any,
    ) -> Any:
        """
        M8 — Generic structured generation for any Pydantic schema.

        Routes through LLMClient when available (M7).  Falls back to the
        legacy Gemini JSON mode path for schemas other than ReviewOutput.
        Always returns a usable instance even on failure.
        """
        # M7: delegate to LLMClient
        if hasattr(self, "_llm_client") and self._llm_client is not None:
            tier = self._model_name_to_tier(model_name)
            return await self._llm_client.generate_structured(tier, prompt, schema)

        # Legacy: generate raw text, strip fences, parse
        text = await self.generate_review(model_name, prompt)
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
        try:
            data = json.loads(text)
            return schema.model_validate(data)
        except Exception as e:
            logger.warning(f"[M8] JSON parse failed for {schema.__name__}: {e}")
            try:
                return schema()
            except Exception:
                return schema(summary=text[:200])  # type: ignore[call-arg]

    async def stream_review(
        self,
        model_name: str,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream code review generation
        
        Yields chunks of generated text as they're produced
        """
        await self._enforce_rate_limit()
        
        system = system_instruction or self.templates.SYSTEM_PROMPT
        
        try:
            logger.info(f"Starting streaming review with {model_name}")
            
            response = await asyncio.to_thread(
                self.client.models.generate_content_stream,
                model=model_name,
                contents=prompt,
                config=self._get_generation_config(),
                safety_settings=self._get_safety_settings()
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            
            logger.info("Streaming review completed")
            
        except Exception as e:
            logger.error(f"Error streaming review: {e}")
            yield f"Error generating review: {str(e)}"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_gemini_client(api_key: Optional[str] = None) -> GeminiClient:
    """Factory function to create Gemini client"""
    if api_key:
        config = GeminiConfig(api_key=api_key)
        return GeminiClient(config)
    return GeminiClient()

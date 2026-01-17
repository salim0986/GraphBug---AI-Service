"""
Google Gemini API Client - Phase 4.3
Handles authentication, model selection, and API calls to Gemini
"""

from typing import Optional, List, Dict, Any, AsyncIterator
import os
import asyncio
from dataclasses import dataclass
from google import genai
from google.genai import types
from .logger import setup_logger

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
    
    QUICK_REVIEW_PROMPT = """## Quick Review Request

Perform a focused code review of this small PR.

**PR Title:** {pr_title}
**Files Changed:** {total_files}
**Lines Changed:** +{additions} -{deletions}

{description}

**Changed Files:**
{files_summary}

**Key Issues Found:**
{issues_summary}

Provide a concise review covering:
1. Critical issues (if any)
2. Code quality observations
3. Quick recommendations

Keep it brief but actionable.
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
    
    DEEP_REVIEW_PROMPT = """## Deep Code Review Request

This is a complex, high-risk PR requiring thorough analysis.

**PR Title:** {pr_title}
**Description:** {description}

**Complexity Metrics:**
- Total Files: {total_files}
- Code Changes: +{additions} -{deletions}
- Languages: {languages}
- Risk Level: **{risk_level}**
- Cyclomatic Complexity: High

**Critical Concerns:**
{critical_issues}

**High Priority Issues:**
{high_issues}

**Architecture Impact:**
- Affected Components: {affected_callers} functions
- Complexity Hotspots: {complexity_hotspots}
- Coupling Issues: {coupling_files}

**Similar Code Patterns:**
{similar_code}

**File-by-File Analysis Required:**
{files_details}

Perform an in-depth review including:
1. **Security Deep Dive** - Analyze all security implications
2. **Architecture Review** - Assess design decisions and patterns
3. **Performance Analysis** - Identify bottlenecks and inefficiencies
4. **Maintainability** - Evaluate long-term code health
5. **Testing Strategy** - Recommend comprehensive test coverage
6. **Risk Assessment** - Identify potential production issues
7. **Migration Path** - If breaking changes, suggest migration
8. **Documentation** - Assess documentation needs

Be extremely thorough. This PR requires careful scrutiny.
"""
    
    FILE_REVIEW_PROMPT = """## Review Individual File

**File:** {filename}
**Language:** {language}
**Changes:** +{additions} -{deletions}

**Static Analysis Issues Detected:**
{issues}

**🔍 Similar Code Patterns:**
{similar_code}

**📊 File Dependencies:**
{dependencies}

**Code Diff:**
```{language}
{diff}
```

**Review Instructions:**
Leverage the context above (similar code and dependencies) to provide context-aware feedback.

Provide detailed feedback on:
1. **Issues and How to Fix Them** - Reference specific line numbers and provide code examples
2. **Code Quality Improvements** - Based on similar patterns found in the codebase
3. **Best Practices** - For this {language} language
4. **Architectural Insights** - Considering the dependencies and coupling shown above
5. **Refactoring Opportunities** - Suggest consolidation if similar code is found

Format your response with:
- Clear section headers (##)
- Severity badges for issues (🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low)
- Code examples in ```{language} blocks
- Specific line references
"""
    
    AGGREGATION_PROMPT = """## Aggregate Review Results

Synthesize multiple file reviews into a cohesive overall review.

**PR Summary:**
- Title: {pr_title}
- Files Reviewed: {files_count}
- Total Issues: {total_issues}

**Individual File Reviews:**
{file_reviews}

**Overall Metrics:**
- Critical Issues: {critical_count}
- High Issues: {high_count}  
- Medium Issues: {medium_count}

Create a unified review that:
1. Summarizes key findings across all files
2. Prioritizes the most important issues
3. Identifies patterns and themes
4. Provides actionable recommendations
5. Gives an overall assessment (Approve / Request Changes / Comment)

Format as a clear, well-structured PR review comment.
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

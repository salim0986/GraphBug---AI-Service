"""
M7 — Multi-provider LLM abstraction backed by LiteLLM.

`LLMClient` is the single entry point for all AI generation in Graph Bug.
It translates Graph Bug's internal tier names (flash / pro / thinking) to
provider-specific LiteLLM model strings, so the rest of the codebase stays
provider-agnostic.

Supported providers: gemini · anthropic · openai · mistral · ollama

Usage:
    config = LLMConfig(provider="anthropic", api_key="sk-ant-...")
    client = LLMClient(config)
    text   = await client.generate(tier="pro", prompt="...")
    review = await client.generate_structured(tier="pro", prompt="...", schema=ReviewOutput)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import litellm
from pydantic import BaseModel

from .logger import setup_logger
from .review_schema import ReviewOutput

logger = setup_logger(__name__)

# Suppress verbose litellm logging unless DEBUG is explicitly requested.
litellm.set_verbose = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tier → provider-specific model map
# ---------------------------------------------------------------------------

DEFAULT_TIER_MAP: Dict[str, Dict[str, str]] = {
    "gemini": {
        "flash":    "gemini/gemini-2.5-flash",
        "pro":      "gemini/gemini-2.5-pro",
        "thinking": "gemini/gemini-2.5-flash",
    },
    "anthropic": {
        "flash":    "anthropic/claude-haiku-4-5-20251001",
        "pro":      "anthropic/claude-sonnet-4-6",
        "thinking": "anthropic/claude-opus-4-8",
    },
    "openai": {
        "flash":    "openai/gpt-4o-mini",
        "pro":      "openai/gpt-4o",
        "thinking": "openai/o1",
    },
    "mistral": {
        "flash":    "mistral/mistral-small-latest",
        "pro":      "mistral/mistral-large-latest",
        "thinking": "mistral/mistral-large-latest",
    },
    "ollama": {
        "flash":    "ollama/qwen2.5-coder:7b",
        "pro":      "ollama/qwen2.5-coder:32b",
        "thinking": "ollama/qwen2.5-coder:32b",
    },
}

# Environment variable names for each provider's API key.
_PROVIDER_ENV: Dict[str, str] = {
    "gemini":    "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "mistral":   "MISTRAL_API_KEY",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    provider: str = "gemini"
    api_key: Optional[str] = None
    tier_map: Dict[str, Dict[str, str]] = field(default_factory=lambda: DEFAULT_TIER_MAP)
    temperature: float = 0.7
    max_tokens: int = 8192
    retry_attempts: int = 3
    retry_delay: float = 2.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Multi-provider LLM client backed by LiteLLM.

    All generation goes through `generate()` (raw text) or
    `generate_structured()` (Pydantic model).  The model selected depends on
    the `tier` argument and the provider configured in `LLMConfig`.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        # Do NOT write to os.environ here — that is a global mutation and races
        # when multiple users trigger concurrent reviews with different providers.
        # Instead, pass api_key directly to each litellm.completion() call.
        # M10: cumulative usage stats for this client instance.
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.total_cost: float = 0.0

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def get_usage_stats(self) -> Dict[str, object]:
        """Return cumulative token and cost stats for this client instance."""
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.tokens_in + self.tokens_out,
            "total_cost": self.total_cost,
        }

    def resolve_model(self, tier: str) -> str:
        """Return the LiteLLM model string for the given tier."""
        provider_map = self.config.tier_map.get(self.config.provider, {})
        model = provider_map.get(tier)
        if not model:
            model = provider_map.get("pro") or next(iter(provider_map.values()), "gemini/gemini-2.5-flash")
        return model

    async def generate(self, tier: str, prompt: str) -> str:
        """
        Generate raw text for the given tier.
        Retries with exponential back-off on transient errors.
        Records token usage, cost, and a LangFuse trace on success.
        """
        model = self.resolve_model(tier)
        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(
                    f"[M7] LLMClient.generate provider={self.config.provider} "
                    f"model={model} attempt={attempt + 1}"
                )
                call_kwargs: Dict[str, object] = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                if self.config.api_key:
                    call_kwargs["api_key"] = self.config.api_key
                response = await asyncio.to_thread(litellm.completion, **call_kwargs)
                text: str = response.choices[0].message.content or ""
                logger.info(f"[M7] Generated {len(text)} chars")
                # M10: track tokens/cost and record in LangFuse
                in_tok, out_tok, cost = self._extract_usage(response)
                self.tokens_in += in_tok
                self.tokens_out += out_tok
                self.total_cost += cost
                self._log_usage(model, in_tok + out_tok)
                self._observe("generate", model, prompt, text, in_tok, out_tok,
                               tier=tier, provider=self.config.provider)
                return text
            except Exception as e:
                logger.warning(f"[M7] generate attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"[M7] Max retries exceeded for generate: {e}")
                    raise
        return ""

    async def generate_structured(
        self,
        tier: str,
        prompt: str,
        schema: Type[BaseModel] = ReviewOutput,
    ) -> BaseModel:
        """
        Generate a structured JSON response that conforms to `schema`.

        Uses LiteLLM's `response_format={"type": "json_object"}` where
        supported (OpenAI, Anthropic tool-use mode, etc.).  For providers
        that don't support JSON mode, we generate text and parse it.
        Falls back to a minimal schema instance on any error so callers
        always receive a usable object.
        """
        model = self.resolve_model(tier)
        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(
                    f"[M7] LLMClient.generate_structured provider={self.config.provider} "
                    f"model={model} schema={schema.__name__} attempt={attempt + 1}"
                )
                call_kwargs_s: Dict[str, object] = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                )
                if self.config.api_key:
                    call_kwargs_s["api_key"] = self.config.api_key
                response = await asyncio.to_thread(litellm.completion, **call_kwargs_s)
                text = response.choices[0].message.content or ""
                logger.info(f"[M7] Structured response: {len(text)} chars")
                # M10: track tokens/cost and record in LangFuse
                in_tok, out_tok, cost = self._extract_usage(response)
                self.tokens_in += in_tok
                self.tokens_out += out_tok
                self.total_cost += cost
                self._log_usage(model, in_tok + out_tok)
                self._observe("generate_structured", model, prompt, text[:500], in_tok, out_tok,
                               tier=tier, provider=self.config.provider, schema=schema.__name__)
                return self._parse_json(text, schema)
            except Exception as e:
                logger.warning(f"[M7] generate_structured attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"[M7] Max retries exceeded for generate_structured: {e}")
                    return self._fallback(schema, str(e))
        return self._fallback(schema, "Max retries exceeded")

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _extract_usage(response: object) -> tuple[int, int, float]:
        """Return (input_tokens, output_tokens, cost_usd) from a LiteLLM response."""
        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = 0.0
        try:
            cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:
            pass
        return in_tok, out_tok, cost

    @staticmethod
    def _log_usage(model: str, total_tokens: int) -> None:
        """Wire token counts into the global TokenBudget."""
        if total_tokens <= 0:
            return
        try:
            from .cost_optimizer import token_budget  # local import to avoid circular
            token_budget.log_usage(model, total_tokens)
        except Exception:
            pass

    @staticmethod
    def _observe(
        name: str,
        model: str,
        prompt: str,
        output: str,
        in_tok: int,
        out_tok: int,
        **meta: object,
    ) -> None:
        """Record the call in LangFuse (no-op when LangFuse not configured)."""
        try:
            from .observability import observe_llm_call
            observe_llm_call(name, model, prompt, output, in_tok, out_tok, **meta)
        except Exception:
            pass

    @staticmethod
    def _parse_json(text: str, schema: Type[BaseModel]) -> BaseModel:
        """Strip fences, parse JSON, validate against schema. Falls back gracefully."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
        try:
            data = json.loads(text)
            return schema.model_validate(data)
        except Exception as e:
            logger.warning(f"[M7] JSON parse failed: {e}; returning fallback")
            return LLMClient._fallback(schema, text[:500] or "Unparseable output")

    @staticmethod
    def _fallback(schema: Type[BaseModel], reason: str) -> BaseModel:
        """Return a minimal valid instance for schemas that share ReviewOutput fields."""
        try:
            return schema(  # type: ignore[call-arg]
                summary=reason,
                overall_assessment="comment",
                risk_level="low",
            )
        except Exception:
            return schema()  # type: ignore[call-arg]

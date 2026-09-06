"""
LLM client — OpenAI-compatible chat completion wrapper.

Providers:
  - stub       : offline template-based fallback (no network)
  - openrouter : OpenRouter.ai (https://openrouter.ai/api/v1)
  - groq       : Groq Cloud   (https://api.groq.com/openai/v1)
  - gemini     : Google AI    (https://generativelanguage.googleapis.com/v1beta/openai/)
  - nim        : NVIDIA NIM   (custom base_url)

The OpenAI Python SDK is used everywhere — it works with any
OpenAI-compatible endpoint via the `base_url` parameter.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """One message in a chat conversation."""
    role: str           # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass
class ChatResponse:
    """Result of a chat completion."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMClient:
    """Pluggable LLM client. Auto-selects provider from env vars.

    Resolution order for the API key:
      1. Explicit constructor argument
      2. Provider-specific env var (OPENROUTER_API_KEY, GROQ_API_KEY, ...)
      3. Generic NIM_API_KEY fallback

    Resolution order for the model:
      1. Explicit constructor argument
      2. ADVISOR_LLM_MODEL env var
      3. Provider default

    If no key is found anywhere, falls back to stub mode.
    """

    # (provider, base_url, default_model, env_var_for_key)
    PROVIDERS = {
        "stub":       ("",                       "stub-template-v1",  ""),
        "openrouter": ("https://openrouter.ai/api/v1",
                                              "meta-llama/llama-3.1-8b-instruct:free",
                                              "OPENROUTER_API_KEY"),
        "groq":       ("https://api.groq.com/openai/v1",
                                              "llama-3.1-8b-instant",
                                              "GROQ_API_KEY"),
        "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/",
                                              "gemini-1.5-flash",
                                              "GOOGLE_API_KEY"),
        "nim":        ("",                       "meta/llama-3.1-8b-instruct",
                                              "NIM_API_KEY"),
    }

    def __init__(self,
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout_s: float = 60.0):
        self.provider = (provider
                         or os.getenv("ADVISOR_LLM_PROVIDER")
                         or "stub").lower().strip()
        if self.provider not in self.PROVIDERS:
            logger.warning("Unknown LLM provider %r — falling back to stub",
                           self.provider)
            self.provider = "stub"

        default_url, default_model, env_var = self.PROVIDERS[self.provider]
        self.model = (model or os.getenv("ADVISOR_LLM_MODEL") or default_model)
        self.base_url = (base_url
                         or os.getenv(f"{self.provider.upper()}_BASE_URL")
                         or default_url)
        self.api_key = (api_key
                        or os.getenv(env_var)
                        or (os.getenv("NIM_API_KEY") if self.provider == "nim" else "")
                        or "").strip()
        self.timeout_s = timeout_s

        # Lazy-init OpenAI client
        self._client = None
        if self.provider != "stub" and self.api_key:
            try:
                from openai import OpenAI
                if self.base_url:
                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        timeout=self.timeout_s,
                    )
                else:
                    self._client = OpenAI(
                        api_key=self.api_key,
                        timeout=self.timeout_s,
                    )
            except Exception as e:
                logger.warning("Could not init OpenAI client (%s) — falling back to stub", e)
                self._client = None
                self.provider = "stub"

    @property
    def is_live(self) -> bool:
        """True if a real LLM is configured (not stub)."""
        return self.provider != "stub" and self._client is not None

    def chat(self,
             messages: list[ChatMessage],
             temperature: float = 0.2,
             max_tokens: int = 1024,
             **kwargs) -> ChatResponse:
        """Send a chat completion request.

        If provider is 'stub' or no API key is configured, returns a
        deterministic template-based answer.
        """
        if not self.is_live:
            return self._stub_response(messages)

        try:
            return self._openai_chat(messages, temperature, max_tokens, **kwargs)
        except Exception as e:
            logger.warning("LLM call failed (%s) — falling back to stub", e)
            return self._stub_response(messages)

    def _openai_chat(self, messages, temperature, max_tokens, **kwargs) -> ChatResponse:
        payload = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        msg = resp.choices[0].message
        usage = resp.usage or {}
        return ChatResponse(
            content=(msg.content or "").strip(),
            model=resp.model,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )

    def _stub_response(self, messages: list[ChatMessage]) -> ChatResponse:
        """Deterministic template-based fallback.

        Used when no API key is configured. Picks out the last user
        message and returns a canned "no LLM configured" response that
        still includes any evidence the orchestrator prepended.
        """
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        user_text = (last_user.content if last_user else "").strip()
        # Truncate very long inputs for the stub response
        snippet = user_text[:200] + ("..." if len(user_text) > 200 else "")
        content = (
            "[stub mode — no LLM API key configured]\n\n"
            f"You asked: \"{snippet}\"\n\n"
            "The rule-based multi-agent system has analyzed your query but no "
            "LLM is configured to produce a natural-language explanation. "
            "Set ADVISOR_LLM_PROVIDER (openrouter / groq / gemini / nim) and the "
            "matching API key in your environment to enable full LLM responses."
        )
        return ChatResponse(content=content, model="stub-template-v1")

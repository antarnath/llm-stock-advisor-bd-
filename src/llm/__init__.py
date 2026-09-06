"""
LLM client — OpenAI-compatible chat completion wrapper.

Supports:
  - stub       (offline; uses a deterministic template)
  - openrouter (OpenRouter.ai — many models)
  - groq       (fast inference)
  - gemini     (Google AI Studio)
  - nim        (NVIDIA NIM — local or hosted)

Selected via env: ADVISOR_LLM_PROVIDER + ADVISOR_LLM_MODEL + *_API_KEY.
All providers use the OpenAI Python SDK (openai>=1.x) with a custom
base_url where needed.
"""

from .client import LLMClient, ChatMessage, ChatResponse

__all__ = ["LLMClient", "ChatMessage", "ChatResponse"]

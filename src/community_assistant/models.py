"""Model factory. Override with COMMUNITY_MODEL (e.g. google_genai:gemini-2.5-pro).

Set COMMUNITY_MODEL_BASE_URL to route model calls through an LLM gateway or proxy.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def get_model() -> BaseChatModel:
    kwargs = {}
    if base_url := os.environ.get("COMMUNITY_MODEL_BASE_URL"):
        kwargs["base_url"] = base_url
    return init_chat_model(os.environ.get("COMMUNITY_MODEL", "anthropic:claude-sonnet-5-5"), **kwargs)

"""LangChain model factory.

Centralises creation of :class:`~langchain_core.language_models.BaseChatModel`
instances so that every agent in the graph uses a consistently configured
model.

Supported providers
-------------------
* **OpenRouter** – default; wraps OpenAI-compatible endpoint at
  ``https://openrouter.ai/api/v1``.
* **OpenAI** – direct OpenAI API.
* **Google** – Google Generative AI (Gemini).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from rohan.config.llm_settings import LLMSettings

logger = logging.getLogger(__name__)

# OpenRouter base URL
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _create_openrouter_model(
    model_name: str,
    settings: LLMSettings,
    **kwargs: object,
) -> BaseChatModel:
    """Create a model that talks to OpenRouter (OpenAI-compatible)."""
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is required when provider='openrouter'. " "Set it in your .env file or environment.")
    return ChatOpenAI(
        model=model_name,
        api_key=settings.openrouter_api_key.get_secret_value(),  # type: ignore[arg-type]
        base_url=_OPENROUTER_BASE_URL,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        **kwargs,  # type: ignore[arg-type]
    )


def _create_openai_model(
    model_name: str,
    settings: LLMSettings,
    **kwargs: object,
) -> BaseChatModel:
    """Create a direct OpenAI model."""
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required when provider='openai'. " "Set it in your .env file or environment.")
    return ChatOpenAI(
        model=model_name,
        api_key=settings.openai_api_key.get_secret_value(),  # type: ignore[arg-type]
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        **kwargs,  # type: ignore[arg-type]
    )


def _create_google_model(
    model_name: str,
    settings: LLMSettings,
    **kwargs: object,
) -> BaseChatModel:
    """Create a Google Generative AI model."""
    # Lazy import to avoid hard dependency
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError("langchain-google-genai is required for provider='google'. " "Install it with: pip install langchain-google-genai") from exc

    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required when provider='google'. " "Set it in your .env file or environment.")
    model: BaseChatModel = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key.get_secret_value(),  # type: ignore[arg-type]
        temperature=settings.temperature,
        max_output_tokens=settings.max_tokens,
        **kwargs,  # type: ignore[arg-type]
    )
    return model


_PROVIDER_FACTORIES = {
    "openrouter": _create_openrouter_model,
    "openai": _create_openai_model,
    "google": _create_google_model,
}


def create_chat_model(
    model_name: str,
    settings: LLMSettings,
    **kwargs: object,
) -> BaseChatModel:
    """Create a LangChain chat model for the given provider and model name.

    Parameters
    ----------
    model_name:
        Provider-specific model identifier
        (e.g. ``"anthropic/claude-sonnet-4"`` for OpenRouter).
    settings:
        LLM configuration settings.
    **kwargs:
        Extra keyword arguments forwarded to the underlying LangChain class.

    Returns
    -------
    BaseChatModel
        A configured, ready-to-use LangChain chat model.
    """
    factory = _PROVIDER_FACTORIES.get(settings.provider)
    if factory is None:
        raise ValueError(f"Unsupported LLM provider: {settings.provider!r}. " f"Choose from {list(_PROVIDER_FACTORIES.keys())}")
    logger.info("Creating %s model %r", settings.provider.value, model_name)
    return factory(model_name, settings, **kwargs)


@lru_cache(maxsize=8)
def _cached_settings() -> LLMSettings:
    """Load settings exactly once (cached)."""
    from rohan.config.llm_settings import LLMSettings

    return LLMSettings()


def get_codegen_model(settings: LLMSettings | None = None) -> BaseChatModel:
    """Convenience: return the code-generation model."""
    s = settings or _cached_settings()
    return create_chat_model(s.codegen_model, s)


def get_analysis_model(settings: LLMSettings | None = None) -> BaseChatModel:
    """Convenience: return the analysis / explainer model."""
    s = settings or _cached_settings()
    return create_chat_model(s.analysis_model, s)


def get_judge_model(settings: LLMSettings | None = None) -> BaseChatModel:
    """Convenience: return the convergence-judge model."""
    s = settings or _cached_settings()
    return create_chat_model(s.judge_model, s)

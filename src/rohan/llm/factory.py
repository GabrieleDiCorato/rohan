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
from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable

    from rohan.config.llm_settings import LLMSettings

logger = logging.getLogger(__name__)

# OpenRouter base URL
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _create_openrouter_model(
    model_name: str,
    settings: LLMSettings,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a model that talks to OpenRouter (OpenAI-compatible)."""
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is required when provider='openrouter'. Set it in your .env file or environment.")
    return ChatOpenAI(
        model=model_name,
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=_OPENROUTER_BASE_URL,
        temperature=kwargs.pop("temperature", settings.temperature),
        max_tokens=kwargs.pop("max_tokens", settings.max_tokens),  # pyright: ignore[reportCallIssue]
        **kwargs,
    )


def _create_openai_model(
    model_name: str,
    settings: LLMSettings,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a direct OpenAI model."""
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required when provider='openai'. Set it in your .env file or environment.")
    return ChatOpenAI(
        model=model_name,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=kwargs.pop("temperature", settings.temperature),
        max_tokens=kwargs.pop("max_tokens", settings.max_tokens),  # pyright: ignore[reportCallIssue]
        **kwargs,
    )


def _create_google_model(
    model_name: str,
    settings: LLMSettings,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a Google Generative AI model."""
    # Lazy import to avoid hard dependency
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError("langchain-google-genai is required for provider='google'. Install it with: pip install langchain-google-genai") from exc

    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required when provider='google'. Set it in your .env file or environment.")
    model: BaseChatModel = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key.get_secret_value(),
        temperature=kwargs.pop("temperature", settings.temperature),
        max_output_tokens=kwargs.pop("max_output_tokens", settings.max_tokens),
        **kwargs,
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
    **kwargs: Any,
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
        raise ValueError(f"Unsupported LLM provider: {settings.provider!r}. Choose from {list(_PROVIDER_FACTORIES.keys())}")
    logger.info("Creating %s model %r", settings.provider.value, model_name)
    return factory(model_name, settings, **kwargs)


@lru_cache(maxsize=1)
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
    """Convenience: return the convergence-judge model.

    Uses ``judge_temperature`` (default 0.0) for deterministic scoring.
    """
    s = settings or _cached_settings()
    return create_chat_model(s.judge_model, s, temperature=s.judge_temperature)


def get_planner_model(settings: LLMSettings | None = None) -> BaseChatModel:
    """Convenience: return the scenario-planner model (fast/cheap)."""
    s = settings or _cached_settings()
    return create_chat_model(s.planner_model, s)


def get_structured_model[T](
    model: BaseChatModel,
    schema: type[T],
) -> Runnable[Any, T]:
    """Bind *model* to return structured output matching *schema*.

    Uses ``method="function_calling"`` (tool-use) rather than the default
    ``json_schema`` mode.  Function calling is reliably supported across
    all providers (OpenRouter → Claude, Gemini, etc.), whereas
    ``json_schema`` is an OpenAI-specific API that proxies don't always
    honour.

    ``include_raw=True`` is used so we can detect when a model responded with
    plain text instead of invoking the tool (``parsed`` would be ``None``).
    Callers must handle ``None`` and retry as appropriate.

    Returns a :class:`~langchain_core.runnables.Runnable` whose
    ``.invoke()`` returns an instance of *schema*, or ``None`` on parse failure.
    """
    raw_runnable = model.with_structured_output(schema, method="function_calling", include_raw=True)  # type: ignore[call-overload]

    def _extract_or_log(x: dict) -> T | None:
        if x.get("parsed") is not None:
            return x["parsed"]
        parsing_error = x.get("parsing_error")
        parsing_error_name = type(parsing_error).__name__ if parsing_error is not None else "None"
        parsing_error_msg = str(parsing_error) if parsing_error is not None else "unknown error"
        raw_obj = x.get("raw")
        tool_calls = getattr(raw_obj, "tool_calls", None)
        tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
        raw_content = getattr(raw_obj, "content", "")
        raw_preview = ""
        if isinstance(raw_content, str):
            raw_preview = raw_content[:180].replace("\n", " ")
        logger.warning(
            "Structured output parse failure for %s: error_type=%s error=%s tool_calls=%d raw_preview=%r",
            schema.__name__,
            parsing_error_name,
            parsing_error_msg,
            tool_call_count,
            raw_preview,
        )
        return None

    return raw_runnable.pipe(_extract_or_log)  # type: ignore[return-value]

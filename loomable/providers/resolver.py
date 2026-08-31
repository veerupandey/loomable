"""loomable.providers.resolver - Model string shorthand resolver.

Resolves a string like ``"openai:gpt-4o-mini"`` into the appropriate provider
instance, reading API keys from environment variables automatically.

Supported formats:
- ``"openai:gpt-4o-mini"`` → OpenAIProvider(model="gpt-4o-mini")
- ``"azure:gpt-4.1-mini"`` → AzureOpenAIProvider(deployment="gpt-4.1-mini")
- ``"anthropic:claude-sonnet-4-20250514"`` → AnthropicProvider(model="...")
- ``"bedrock:anthropic.claude-3-haiku-20240307-v1:0"`` → BedrockProvider(model="...")
- ``"groq:llama-3.3-70b-versatile"`` → GroqProvider(model="...")
- ``"ollama:llama3.2"`` → OllamaProvider(model="llama3.2")
- ``"gemini:gemini-2.0-flash"`` → GeminiProvider(model="...")

When a bare model name without a colon is given, infers ``"openai:"`` prefix.
"""

from __future__ import annotations

from typing import Any


def resolve_model(model_str: str) -> Any:
    """Resolve a model string shorthand into a provider instance.

    Parameters
    ----------
    model_str:
        A string in the format ``"provider:model_name"`` or just ``"model_name"``
        (which defaults to OpenAI).

    Returns
    -------
    A ``ModelProvider`` instance configured with the resolved model and env-based auth.

    Raises
    ------
    ValueError:
        If the provider prefix is not recognized.
    """
    if ":" not in model_str:
        # Bare model name → default to OpenAI
        provider_key = "openai"
        model_name = model_str
    else:
        provider_key, model_name = model_str.split(":", 1)
        provider_key = provider_key.lower().strip()
        model_name = model_name.strip()

    if provider_key == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider(model=model_name)

    elif provider_key == "azure":
        from .openai import AzureOpenAIProvider
        return AzureOpenAIProvider(deployment=model_name)

    elif provider_key == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(model=model_name)

    elif provider_key in ("bedrock", "aws"):
        from .bedrock import BedrockProvider
        return BedrockProvider(model=model_name)

    elif provider_key == "groq":
        from .groq import GroqProvider
        return GroqProvider(model=model_name)

    elif provider_key == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(model=model_name)

    elif provider_key in ("gemini", "google"):
        from .gemini import GeminiProvider
        return GeminiProvider(model=model_name)

    else:
        raise ValueError(
            f"Unknown provider '{provider_key}' in model string '{model_str}'. "
            f"Supported: openai, azure, anthropic, bedrock, groq, ollama, gemini."
        )

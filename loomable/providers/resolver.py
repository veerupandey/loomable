"""Resolve built-in and third-party model adapters on demand."""
from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from loomable.kernel.contracts import ModelProvider

ProviderFactory = Callable[[str], ModelProvider]

_BUILTINS = {
    "openai": (".openai", "OpenAIProvider", "model"),
    "azure": (".openai", "AzureOpenAIProvider", "deployment"),
    "anthropic": (".anthropic", "AnthropicProvider", "model"),
    "bedrock": (".bedrock", "BedrockProvider", "model"),
    "groq": (".groq", "GroqProvider", "model"),
    "ollama": (".ollama", "OllamaProvider", "model"),
    "gemini": (".gemini", "GeminiProvider", "model"),
}
_ALIASES = {"aws": "bedrock", "google": "gemini"}
_FACTORIES: dict[str, ProviderFactory] = {}
_ENTRY_POINT_GROUP = "loomable.providers"


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a model-name-to-provider factory during application setup."""
    key = name.strip().lower()
    if not key or ":" in key or any(c.isspace() for c in key):
        raise ValueError("Provider names must be nonempty prefixes without spaces or colons")
    if not callable(factory):
        raise TypeError("Provider factory must be callable")
    if key in _BUILTINS or key in _ALIASES:
        raise ValueError(f"Provider '{key}' is reserved")
    previous = _FACTORIES.setdefault(key, factory)
    if previous is not factory:
        raise ValueError(f"Provider '{key}' is already registered")


def resolve_model(model_str: str) -> ModelProvider:
    """Resolve `prefix:model`. Bare model names use OpenAI."""
    prefix, separator, model = model_str.partition(":")
    if not separator:
        prefix, model = "openai", prefix
    prefix = prefix.strip().lower()
    model = model.strip()
    if not model:
        raise ValueError("Model name must be nonempty")
    prefix = _ALIASES.get(prefix, prefix)
    builtin = _BUILTINS.get(prefix)
    if builtin is not None:
        module, name, parameter = builtin
        return getattr(import_module(module, __package__), name)(**{parameter: model})

    factory = _FACTORIES.get(prefix)
    if factory is None:
        from importlib.metadata import entry_points

        plugins = tuple(entry_points(group=_ENTRY_POINT_GROUP, name=prefix))
        if len(plugins) > 1:
            raise ValueError(f"Multiple installed plugins provide '{prefix}'")
        if not plugins:
            raise ValueError(f"Unknown provider '{prefix}' in model string '{model_str}'")
        factory = plugins[0].load()
    if not callable(factory):
        raise TypeError(f"Provider plugin '{prefix}' must export a callable factory")
    provider = factory(model)
    if not callable(getattr(provider, "complete", None)):
        raise TypeError(f"Provider '{prefix}' must implement complete(request)")
    return provider

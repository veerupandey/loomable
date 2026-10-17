"""loomable.providers - Built-in, ready-to-use model providers.

This edge package ships concrete :class:`~loomable.kernel.contracts.ModelProvider`
implementations so users don't have to write their own HTTP plumbing:

- :class:`OpenAIProvider` - OpenAI and any OpenAI-compatible endpoint (vLLM,
  Together, Groq, Ollama, LM Studio, ...) via a configurable ``base_url``.
- :class:`AzureOpenAIProvider` - Azure OpenAI (deployment + api-version).
- :class:`AnthropicProvider` - Anthropic Messages API (Claude).

All providers implement the kernel ``ModelProvider`` protocol
(``async complete(request) -> ModelResponse``) and translate the
provider-agnostic :class:`~loomable.kernel.models.ModelRequest` /
:class:`~loomable.kernel.models.ModelResponse` shapes to and from each API. They
depend only on ``httpx`` (already a core dependency) and never modify
``loomable.kernel``.

Example
-------
    from loomable.agent import Agent
    from loomable.providers import OpenAIProvider

    agent = Agent(model=OpenAIProvider(model="gpt-4o-mini"))  # reads OPENAI_API_KEY
    print(agent.run("hello").output.text())"""
from __future__ import annotations

from importlib import import_module

__all__ = ['OpenAIProvider', 'AzureOpenAIProvider', 'AnthropicProvider', 'BedrockProvider', 'GeminiProvider', 'GroqProvider', 'OllamaProvider', 'Embedder', 'OpenAIEmbedder', 'AzureOpenAIEmbedder', 'GeminiEmbedder', 'HuggingFaceEmbedder', 'embed_many', 'TransientProviderError', 'PermanentProviderError', 'ResilientModel', 'RetryPolicy', 'open_vector_store', 'parse_vector_uri']

__all__ += ["GeminiMediaProvider", "register_provider"]

_EXPORTS = {
    "register_provider": (".resolver", "register_provider"),
    'OpenAIProvider': ('.openai', 'OpenAIProvider'),
    'AzureOpenAIProvider': ('.openai', 'AzureOpenAIProvider'),
    'AnthropicProvider': ('.anthropic', 'AnthropicProvider'),
    'BedrockProvider': ('.bedrock', 'BedrockProvider'),
    'GeminiProvider': ('.gemini', 'GeminiProvider'),
    'GeminiMediaProvider': ('.gemini', 'GeminiMediaProvider'),
    'GroqProvider': ('.groq', 'GroqProvider'),
    'OllamaProvider': ('.ollama', 'OllamaProvider'),
    'Embedder': ('.embedders', 'Embedder'),
    'OpenAIEmbedder': ('.embedders', 'OpenAIEmbedder'),
    'AzureOpenAIEmbedder': ('.embedders', 'AzureOpenAIEmbedder'),
    'GeminiEmbedder': ('.embedders', 'GeminiEmbedder'),
    'HuggingFaceEmbedder': ('.embedders', 'HuggingFaceEmbedder'),
    'embed_many': ('.embedders', 'embed_many'),
    'TransientProviderError': ('.errors', 'TransientProviderError'),
    'PermanentProviderError': ('.errors', 'PermanentProviderError'),
    'ResilientModel': ('.resilient', 'ResilientModel'),
    'RetryPolicy': ('.resilient', 'RetryPolicy'),
    'open_vector_store': ('.vector_store', 'open_vector_store'),
    'parse_vector_uri': ('.vector_store', 'parse_vector_uri'),
}

def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

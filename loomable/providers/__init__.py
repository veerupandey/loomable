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
    print(agent.run("hello").output.text())
"""

from .anthropic import AnthropicProvider
from .embedders import AzureOpenAIEmbedder, Embedder, OpenAIEmbedder
from .errors import PermanentProviderError, TransientProviderError
from .gemini import GeminiProvider
from .groq import GroqProvider
from .ollama import OllamaProvider
from .openai import AzureOpenAIProvider, OpenAIProvider
from .resilient import ResilientModel, RetryPolicy
from .vector_store import open_vector_store

__all__ = [
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "GroqProvider",
    "OllamaProvider",
    "Embedder",
    "OpenAIEmbedder",
    "AzureOpenAIEmbedder",
    "TransientProviderError",
    "PermanentProviderError",
    "ResilientModel",
    "RetryPolicy",
    "open_vector_store",
]

"""loomable.providers.gemini - Google Gemini provider.

Google exposes Gemini models through an OpenAI-compatible endpoint at
``https://generativelanguage.googleapis.com/v1beta/openai``. This provider
subclasses :class:`OpenAIProvider` and adapts the auth (API key as query param
or bearer token depending on the endpoint variant).

For Vertex AI, point ``base_url`` at the Vertex OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from loomable.content.capabilities import capabilities_for
from loomable.kernel.errors import ModelProviderError
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.media import Image

from ._common import _classify_http_error
from .openai import OpenAIProvider, _HTTPClientScope


class GeminiProvider(OpenAIProvider):
    """A ``ModelProvider`` for Google Gemini via the OpenAI-compatible API.

    Parameters
    ----------
    model:
        The Gemini model name (e.g. ``"gemini-2.0-flash"``, ``"gemini-1.5-pro"``).
    api_key:
        The Google AI API key. Defaults to the ``GOOGLE_API_KEY`` or
        ``GEMINI_API_KEY`` environment variable.
    base_url:
        The API base URL. Defaults to Google's OpenAI-compatible endpoint.
    """

    capabilities = capabilities_for("text+image+audio")
    _chat_audio = True

    def __init__(
        self,
        model: str = "gemini-flash-latest",
        *,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        timeout: float = 60.0,
        **kwargs,
    ) -> None:
        resolved_key = (
            api_key
            if api_key is not None
            else os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY"))
        )
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
            **kwargs,
        )

    @property
    def _provider_id(self) -> str:
        return f"gemini:{self.model}"


class GeminiMediaProvider(_HTTPClientScope):
    """Native Gemini GenerateContent for attachment analysis and image output."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        image_output: bool = False,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.capabilities = capabilities_for(
            input="full", output="text+image" if image_output else "text",
        )
        self._image_output = image_output
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._timeout = timeout
        self._client = client

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send inline text, image, audio and video to a native Gemini model."""
        if request.tools:
            raise ValueError("GeminiMediaProvider handles native media turns. Use GeminiProvider for tool calling.")
        if not self._api_key:
            raise ValueError("GeminiMediaProvider requires a Gemini API key.")
        contents = []
        system = []
        for message in request.messages:
            content = message["content"]
            parts = (
                [{"text": content}] if isinstance(content, str)
                else [self._part(part) for part in content]
            )
            if message["role"] == "system":
                system.extend(parts)
            else:
                role = "model" if message["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": parts})

        config: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens:
            config["maxOutputTokens"] = request.max_tokens
        if self._image_output:
            config["responseModalities"] = ["TEXT", "IMAGE"]
        body = {"contents": contents, "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": system}
        provider_id = f"gemini:{self.model}"
        try:
            async with self._http_client() as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                    json=body,
                    headers={"x-goog-api-key": self._api_key},
                )
                response.raise_for_status()
                data = response.json()
            candidate = data["candidates"][0]
            parts = candidate["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
            media = [
                {"media_type": part["inlineData"]["mimeType"], "data": part["inlineData"]["data"]}
                for part in parts if "inlineData" in part and not part.get("thought")
            ]
        except httpx.HTTPError as exc:
            raise _classify_http_error(provider_id, exc) from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelProviderError(provider_id) from exc
        usage = data.get("usageMetadata", {})
        return ModelResponse(content=text, metadata={"media": media, "model": self.model}, usage={
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
        })

    @staticmethod
    def _part(part: dict[str, Any]) -> dict[str, Any]:
        kind = part["type"]
        if kind == "text":
            return {"text": part["text"]}
        if kind in {"image_url", "video_url", "audio_url"}:
            url = part[kind]["url"]
            header, _, data = url.partition(",")
            if header.startswith("data:") and header.endswith(";base64") and data:
                return {"inlineData": {"mimeType": header[5:-7], "data": data}}
        raise ValueError(f"Native Gemini requires inline bytes for {kind}.")

    async def generate_image(self, prompt: str) -> Image:
        """Generate an image for use as an agent tool result."""
        from loomable.content.capabilities import from_model_response

        if not self._image_output:
            raise ValueError("Set image_output=True with a Gemini image model.")
        result = await self.complete(ModelRequest(messages=[{"role": "user", "content": prompt}]))
        for part in from_model_response(result).parts:
            if part.modality.value == "image":
                return Image.from_media_part(part)
        raise ModelProviderError(f"gemini:{self.model}")

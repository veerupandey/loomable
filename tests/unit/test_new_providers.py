"""Tests for new model providers (Groq, Ollama, Gemini)."""

from __future__ import annotations

import os

import pytest

from loomable.providers.gemini import GeminiProvider
from loomable.providers.groq import GroqProvider
from loomable.providers.ollama import OllamaProvider
from loomable.providers.openai import OpenAIProvider


class TestGroqProvider:
    """Test GroqProvider configuration and inheritance."""

    def test_default_base_url(self):
        p = GroqProvider(model="llama-3.3-70b-versatile", api_key="test")
        assert p._base_url == "https://api.groq.com/openai/v1"

    def test_inherits_openai(self):
        p = GroqProvider(api_key="test")
        assert isinstance(p, OpenAIProvider)

    def test_provider_id(self):
        p = GroqProvider(model="mixtral-8x7b-32768", api_key="test")
        assert p._provider_id == "groq:mixtral-8x7b-32768"

    def test_has_stream(self):
        p = GroqProvider(api_key="test")
        assert hasattr(p, "stream")

    def test_reads_env_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "from-env")
        p = GroqProvider()
        assert p._api_key == "from-env"

    def test_default_model(self):
        p = GroqProvider(api_key="test")
        assert p.model == "llama-3.3-70b-versatile"


class TestOllamaProvider:
    """Test OllamaProvider configuration and inheritance."""

    def test_default_base_url(self):
        p = OllamaProvider()
        assert p._base_url == "http://localhost:11434/v1"

    def test_inherits_openai(self):
        p = OllamaProvider()
        assert isinstance(p, OpenAIProvider)

    def test_provider_id(self):
        p = OllamaProvider(model="mistral")
        assert p._provider_id == "ollama:mistral"

    def test_has_stream(self):
        p = OllamaProvider()
        assert hasattr(p, "stream")

    def test_default_model(self):
        p = OllamaProvider()
        assert p.model == "llama3.2"

    def test_long_timeout(self):
        p = OllamaProvider()
        assert p._timeout == 120.0


class TestGeminiProvider:
    """Test GeminiProvider configuration and inheritance."""

    def test_default_base_url(self):
        p = GeminiProvider(api_key="test")
        assert "generativelanguage.googleapis.com" in p._base_url

    def test_inherits_openai(self):
        p = GeminiProvider(api_key="test")
        assert isinstance(p, OpenAIProvider)

    def test_provider_id(self):
        p = GeminiProvider(model="gemini-1.5-pro", api_key="test")
        assert p._provider_id == "gemini:gemini-1.5-pro"

    def test_has_stream(self):
        p = GeminiProvider(api_key="test")
        assert hasattr(p, "stream")

    def test_reads_gemini_env_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        p = GeminiProvider()
        assert p._api_key == "gemini-key"

    def test_reads_google_env_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        p = GeminiProvider()
        assert p._api_key == "google-key"

    def test_default_model(self):
        p = GeminiProvider(api_key="test")
        assert p.model == "gemini-flash-latest"

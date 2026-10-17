from loomable import Agent
from loomable.providers import OpenAIProvider, register_provider


def test_registered_prefix_builds_real_openai_adapter(monkeypatch):
    from loomable.providers import resolver

    monkeypatch.setattr(resolver, "_FACTORIES", {})
    register_provider("company-openai", OpenAIProvider)
    built = Agent(model="company-openai:gpt-4o-mini").build()
    provider = built.model_interface._providers["company-openai"]
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4o-mini"


def test_installed_entry_point_loads_real_gemini_adapter(monkeypatch, tmp_path):
    from loomable.providers import GeminiProvider

    metadata = tmp_path / "loomable_test_plugin-1.0.dist-info"
    metadata.mkdir()
    (metadata / "entry_points.txt").write_text(
        "[loomable.providers]\ncompany-gemini = loomable.providers.gemini:GeminiProvider\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    agent = Agent(model="company-gemini:gemini-flash-latest").build()
    assert isinstance(agent.model_interface._providers["company-gemini"], GeminiProvider)

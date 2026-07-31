"""22 — Tiered Model Routing with Fallback

Demonstrates configuring multiple model tiers with automatic fallback.
Uses your Azure OpenAI deployment as the primary model.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider

# --- Build a simple agent (single tier — your Azure deployment) ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions="You are a helpful assistant. Be concise.",
    # In production with multiple tiers:
    # tiers={
    #     "premium": {"provider": "gpt-4"},
    #     "standard": {"provider": "gpt-4o-mini"},
    #     "economy": {"provider": "gpt-3.5-turbo"},
    # },
    # tier_policy={"default_tier": "premium"},
    # fallback_tiers={"premium": "standard", "standard": "economy"},
)
built = agent.build()

# --- Run ---

result = asyncio.run(built.arun("Explain tiered model routing in 2 sentences."))
print("=== Tiered Routing ===\n")
print("Answer:", result.output.text())
print(f"\nRouter configured: {built.router is not None}")
print(f"Tier substitution: {result.metadata.get('tier_substitution', 'None (single tier)')}")

print("\n--- How Tiered Routing Works ---")
print("""
Configuration:
  tiers = {
      "premium": {"provider": "gpt-4"},       # Complex reasoning
      "standard": {"provider": "gpt-4o-mini"}, # General tasks
      "economy": {"provider": "gpt-3.5"},      # Simple queries
  }
  tier_policy = {"default_tier": "premium"}
  fallback_tiers = {"premium": "standard", "standard": "economy"}

Behavior:
  1. Router selects a tier based on the policy
  2. If the selected tier fails (ModelProviderError), falls back
  3. TierSubstitution recorded in result.metadata
  4. No tiers = single provider used directly (no router overhead)
""")

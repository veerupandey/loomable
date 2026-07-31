"""21 — Agent with Skills

Demonstrates loading skills from directories. Skills are self-contained
packages with SKILL.md metadata and scripts/ that provide tools.

This example shows the pattern — in production, point skills= to real
skill directories.
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool
from loomable.providers.openai import AzureOpenAIProvider


# --- Simulate a "weather lookup" skill tool ---
# In production, this would be loaded from examples/skills/weather-lookup/scripts/


@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get weather for a city (loaded from weather-lookup skill)."""
    data = {
        "paris": {"celsius": "18°C, sunny", "fahrenheit": "64°F, sunny"},
        "tokyo": {"celsius": "22°C, cloudy", "fahrenheit": "72°F, cloudy"},
        "new york": {"celsius": "12°C, rainy", "fahrenheit": "54°F, rainy"},
    }
    city_data = data.get(city.lower(), {"celsius": "20°C, clear", "fahrenheit": "68°F, clear"})
    return city_data.get(units, city_data["celsius"])


# --- Build agent ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions="You are a travel assistant. Use the weather tool to help plan trips.",
    tools=[get_weather],
    # In production with real skills:
    # skills=[Path("examples/skills")]
)

# --- Show skill structure ---

skills_dir = Path(__file__).parent / "skills"
print("=== Skills Configuration ===")
print(f"Skills directory: {skills_dir}")
if skills_dir.exists():
    for skill_path in skills_dir.iterdir():
        if skill_path.is_dir() and (skill_path / "SKILL.md").exists():
            print(f"  ✓ {skill_path.name}")

print("\nUsage: Agent(model=provider, skills=[Path('path/to/skills')])")
print("Skills load with isolation — failures don't break other tools.\n")

# --- Run ---

result = asyncio.run(agent.arun("What's the weather like in Paris and Tokyo? I'm deciding where to travel."))
print("Answer:", result.output.text())

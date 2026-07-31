"""26 — Parallel Tool Calls

When the model requests multiple tools in a single turn, loomable
dispatches them concurrently via asyncio.gather. This example shows
the model calling 3 tools at once, all executing in parallel.

You can also configure:
- tool_concurrency=N  → max N tools run at the same time (semaphore)
- tool_timeout=5.0    → each tool call has a 5s deadline
"""

import asyncio
import time
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool
from loomable.providers.openai import AzureOpenAIProvider


# --- Tools that simulate slow I/O (sleep 1s each) ---


@tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    await asyncio.sleep(1)  # Simulate network call
    data = {"paris": "22°C, sunny", "tokyo": "28°C, humid", "new york": "18°C, cloudy"}
    return data.get(city.lower(), f"{city}: 20°C, clear")


@tool
async def get_stock_price(symbol: str) -> str:
    """Get current stock price."""
    await asyncio.sleep(1)  # Simulate API call
    prices = {"AAPL": "$195.23", "GOOGL": "$142.67", "MSFT": "$415.80"}
    return prices.get(symbol.upper(), f"{symbol}: $100.00")


@tool
async def get_news(topic: str) -> str:
    """Get latest news headline for a topic."""
    await asyncio.sleep(1)  # Simulate search
    headlines = {
        "ai": "OpenAI announces GPT-5 with improved reasoning capabilities",
        "tech": "Apple unveils new M4 chip with 2x neural engine performance",
        "markets": "S&P 500 hits new all-time high amid AI optimism",
    }
    return headlines.get(topic.lower(), f"Latest on {topic}: Multiple developments reported")


# --- Build agent ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions=(
        "You are a research assistant. When the user asks for multiple pieces of "
        "information, call ALL relevant tools in a single turn (parallel). "
        "Do not call them one at a time."
    ),
    tools=[get_weather, get_stock_price, get_news],
    # Optional: cap parallelism or add timeout
    # tool_concurrency=2,  # Max 2 tools at once
    # tool_timeout=5.0,    # 5s per tool
)

# --- Run ---

print("Asking for weather + stock + news simultaneously...\n")

start = time.time()
result = asyncio.run(agent.arun(
    "Give me the weather in Paris, the AAPL stock price, and the latest AI news."
))
elapsed = time.time() - start

print(f"Answer: {result.output.text()}\n")
print(f"Tools called: {len(result.tool_activity)}")
for activity in result.tool_activity:
    print(f"  - {activity.result.content}")
print(f"\nTotal time: {elapsed:.1f}s")
print("(If tools ran sequentially it would be ~3s; parallel should be ~1s + model time)")

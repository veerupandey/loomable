---
name: weather-lookup
description: Look up current weather conditions for a given location.
---

# Weather Lookup Skill

You have access to the `get_weather` tool which returns current weather
conditions for a specified location.

## Usage

When the user asks about the weather in a specific city or region, invoke
the `get_weather` tool with the `location` argument set to the place name.

## Example

User: "What's the weather like in Vancouver?"

Action: invoke `get_weather` with `--location=Vancouver`

The tool returns a JSON object with temperature, conditions, and humidity.

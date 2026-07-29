"""Mock weather lookup script tool.

This is a representative example bundled with the weather-lookup Domain_Skill.
It echoes a synthetic weather response for any location, demonstrating that
domain capabilities are delivered as Skills (not Kernel modifications).
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Get weather for a location")
    parser.add_argument("--location", required=True, help="Location to look up")
    args = parser.parse_args()

    # Synthetic/mock response
    response = {
        "location": args.location,
        "temperature_c": 18,
        "conditions": "partly cloudy",
        "humidity_pct": 62,
    }
    print(json.dumps(response))


if __name__ == "__main__":
    main()

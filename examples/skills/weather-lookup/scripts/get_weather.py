"""Read one bundled sample weather observation."""

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", default="Vancouver")
    location = parser.parse_args().location
    if location.casefold() != "vancouver":
        print(json.dumps({"error": "The sample contains only Vancouver."}))
        return
    print(json.dumps({"location": "Vancouver", "date": "2025-06-01", "temperature_c": 18,
                      "source": "tutorial fixture"}))


if __name__ == "__main__":
    main()

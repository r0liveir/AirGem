from __future__ import annotations

import argparse
import json
import os

from airgem.core import parse_flight_request
from airgem.llm import OllamaProvider, OpenRouterProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse natural language into a flight request")
    parser.add_argument("prompt")
    parser.add_argument("--provider", choices=("ollama", "openrouter"), default=os.getenv("LLM_PROVIDER", "ollama"))
    args = parser.parse_args()

    provider = OllamaProvider() if args.provider == "ollama" else OpenRouterProvider()
    request = parse_flight_request(args.prompt, provider)
    print(json.dumps({**request.to_dict(), "missing": request.missing}, indent=2))


if __name__ == "__main__":
    main()

# AirGem

> Privacy-first, deterministic flight search and calendar-aware itinerary copilot powered by local open-weight inference and multi-objective Pareto optimization.

**Main problem**: Commercial travel platforms and cloud AI assistants create an uncomfortable trade-off: to receive context-aware trip planning, you must feed proprietary calendar entries, travel schedules, and personal habits into third-party servers where they are stored, analyzed, and (with a great probability) monetized :/

**AirGem** shifts the paradigm. It keeps all calendar parsing, constraint resolution, and trade-off evaluation inside your private infrastructure. An open-weight **Gemma** model is restricted strictly to semantic parsing and report summarization, while candidate retrieval and trade-off optimizations are handled by deterministic Python routines.

## Note: About the project

This project was thought for the DEV Side Hustle Challenge, on Hacktoberfest Open-Source AI Challenge.

Check more here: https://dev.to/devteam/join-the-hacktoberfest-open-source-ai-challenge-side-hustle-2550-in-prizes-across-18-winners-55ba?ajs_uid=01a0d318-0a78-f774-103d-7fbcd081b9c5&utm_campaign=Hacktoberfest+Registrant+DEV+Week+2+Challenge+Announcement&utm_content=Hacktoberfest+Week+1+Challenge+Announcement&utm_medium=Email&utm_source=Customer.io

---

## Architecture Overview

TODO: change this into a real image
```
                      +---------------------------------------+
                      |         Private User Input            |
                      |  - Natural Language Prompt            |
                      |  - Private Calendar File (.ics)       |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     Semantic Parsing (Gemma)          |
                      |  Converts prompt to strict JSON       |
                      |  (Runs on-prem / DigitalOcean Droplet)|
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       Deterministic Core Engine       |
                      |  1. iCal Free/Busy Slot Extraction    |
                      |  2. SerpAPI (Google Flights Engine)   |
                      |  3. Multi-Objective Pareto Filter     |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      Trade-off Synthesis (Gemma)      |
                      |  Explains non-dominated compromises   |
                      |  (e.g., price vs. schedule buffer)    |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |        Sentry Observability           |
                      |  Tracks tool errors, timeouts,        |
                      |  and schema edge cases                |
                      +---------------------------------------+

```

---

## Core Pillars

* **Zero-Leak Calendar Context:** Your schedule never leaves the machine. Calendar `.ics` processing runs locally; external search engines only receive sanitized queries like `GRU -> POA on YYYY-MM-DD`.
* **Deterministic Search & Math:** Avoids LLM arithmetic errors and phantom flight numbers. Flight availability and pricing are retrieved via SerpAPI, while scheduling and trade-off optimization are handled by standard algorithmic routines.
* **Pareto Frontier Optimization:** Instead of guessing a single "best" flight, the system finds non-dominated solutions across competing variables:
* Minimize monetary cost ($)
* Minimize travel duration (hours)
* Maximize buffer time relative to existing calendar meetings



---

## Tech Stack

* **Inference:** Gemma (Ollama / vLLM, hosted on a DigitalOcean Droplet or locally)
* **Search Engine:** SerpAPI (Google Flights Engine)
* **Application Runtime:** Python 3.12+; first slice uses standard-library validation
* **Observability:** Sentry (API latency tracking, structured validation failure alerts)
* **Web / API Layer:** Render / FastAPI

---

## Getting Started

### Prerequisites

* Python 3.12 or higher
* [uv](https://docs.astral.sh/uv/)
* Local or remote Ollama instance running Gemma:
```bash
ollama run gemma3:4b

```


* SerpAPI API Key
* Sentry DSN (optional, for observability)

### Installation

1. Install project:
```bash
uv sync
```

2. Configure environment variables:
```env
LLM_PROVIDER="ollama"
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="gemma3:4b"
OPENROUTER_API_KEY="your_openrouter_key_here"
OPENROUTER_MODEL="google/gemma-4-26b-a4b-it:free"
SERPAPI_API_KEY="your_serpapi_key_here"
SENTRY_DSN="your_sentry_dsn_here"

```



---

## Usage

Current slice parses intent, then searches exact-date flights through SerpAPI:

```bash
uv run airgem "Need cheap travel from GRU to POA on 2026-11-06, returning 2026-11-08"
uv run airgem --provider openrouter "Find the cheapest nonstop flight from GRU to POA"
# Machine-readable output:
uv run airgem --json "Find cheap flights from GRU to POA on 2026-11-06"
```

Search from Python:

```python
from datetime import date

from airgem.core import FlightRequest
from airgem.search import SerpApiProvider

provider = SerpApiProvider()
result = provider.search(
    FlightRequest("GRU", "POA", date(2026, 11, 6), date(2026, 11, 8))
)
# After selecting one returned candidate:
expanded = provider.expand_round_trip(result.outbound[0].candidate_id)
```

SerpAPI defaults: deep search, exact dates, price sorting. Round trips use native round-trip pricing. Search exposes opaque candidate IDs; `SerpApiProvider.expand_round_trip(candidate_id)` expands one selected return leg. Booking resolution is deferred until itinerary selection. Flexible dates are not supported yet. CLI is human-readable by default; use `--json` for machine-readable output.

Run offline tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v

```

### Example Execution Pipeline

1. **Natural Language Translation:** Gemma extracts explicit flight intent into structured data:
```json
{
  "origin": "GRU",
  "destination": "POA",
  "window_start": "2026-11-06T18:00:00",
  "window_end": "2026-11-08T18:00:00",
  "max_stops": 0
}

```


2. **Conflict Checking:** The engine scans `my_schedule.ics` to identify hard blocks and calculate safe arrival buffers.
3. **Candidate Fetching:** Queries SerpAPI for verified flight options matching the candidate dates.
4. **Pareto Dominance Evaluation:** Filters out candidates that are strictly worse than an alternative across all metrics.
5. **Synthesis:** Gemma presents the remaining candidates alongside clear trade-offs:
> **Option A (Lowest Cost):** R$ 420,00 — Departs 19:15, arrives 21:00. Leaves a 45-minute buffer before your dinner event.
> **Option B (Maximum Buffer):** R$ 510,00 — Departs 21:40, arrives 23:25. Adds 2.5 hours of buffer after work, with zero conflict risk.



---

## Project Structure

```
airgem/
├── src/airgem/
│   ├── llm/
│   │   ├── ollama.py       # Local Ollama adapter
│   │   ├── openrouter.py   # OpenRouter fallback adapter
│   │   └── provider.py     # Shared LLM contract + HTTP helper
│   ├── core/
│   │   ├── flight_request.py # Validated normalized request
│   │   └── parse.py           # Natural language → request
│   ├── itinerary.py            # Deterministic pairing and hard constraints
│   ├── pareto.py               # Price/duration/stops candidate filtering
│   └── search/
│       ├── provider.py     # SerpAPI/FastFlights-neutral contract
│       └── serpapi.py      # Google Flights SerpAPI adapter
├── tests/
│   └── test_slice.py
├── main.py               # CLI entrypoint
└── pyproject.toml

```

---

## License

This project is licensed under the MIT License. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.

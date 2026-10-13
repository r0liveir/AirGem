# ✈️AirGem

> A privacy-first flight-search copilot that finds flights around your real
> calendar.

AirGem turns a natural-language travel request into a small set of practical
flight options. It keeps calendar data local, uses Gemma for semantic work,
and keeps search, validation, availability checks, and trade-off filtering
deterministic.

## 🚀 Hacktoberfest Week 2: Side Hustle

AirGem is being built as a submission for the
[Hacktoberfest Open-Source AI Challenge: Week 2 — Side Hustle](https://dev.to/devteam/join-the-hacktoberfest-open-source-ai-challenge-side-hustle-2550-in-prizes-across-18-winners-55ba).

This challenge asks for a new project with open-weight or open-source AI that
solves a real problem someone could plausibly pay for.
With this, AirGem's target user is a frequent traveler, consultant, freelancer, or remote
worker who wants to find an affordable trip without manually comparing flight
times against work, classes, appointments, or family commitments.

## Why open models?

AirGem uses Gemma through Ollama by default, with OpenRouter as an optional
fallback. This keeps the model replaceable and makes local inference possible
on a developer machine or private infrastructure.

The model is deliberately constrained:

- Gemma extracts flight intent from natural language.
- Gemma selects among verified candidate IDs in a bounded agent loop.
- Python validates schemas, applies calendar constraints, pairs itineraries,
  and calculates Pareto trade-offs.
- SerpAPI retrieves flight availability.

This division avoids asking an LLM to invent flight data or perform critical
date, price, and duration arithmetic.

With local inference, calendar contents and the natural-language conversation
can stay on the user's machine instead of being sent to a model vendor. Flight
search still uses SerpAPI with sanitized route and date fields.

## Current slice

The working path is:

```text
natural-language request via CLI
        ↓
Gemma → strict FlightRequest JSON
        ↓
SerpAPI Google Flights search
        ↓
exact-date, price, stop, and calendar filters
        ↓
Pareto frontier: price, duration, stops
        ↓
bounded Gemma candidate selection
        ↓
human-readable or JSON itinerary output
```

Current functionality includes:

- One-way and native round-trip searches.
- **Exact** departure and return dates (flexible dates is still WIP)
- USD as the default currency (override it in the prompt by asking for a preferred currency).
- SerpAPI deep search sorted by price.
- Local `.ics` calendar imports (via `icalendar`).
- Timed, all-day, cancelled, transparent, and recurring calendar events (via `recurring-ical-events`).
- Configurable travel buffer before departure and after arrival (taking into account commute time).
- Calendar filtering before Pareto filtering.
- Selected round-trip outbound expansion before return selection.
- Local **Ollama Gemma** or **OpenRouter Gemma** providers.
- A bounded state-machine agent with deterministic fallback behavior.
- A small local FastAPI web UI with in-memory follow-up sessions.
- Offline unit tests for parsing, search adapters, calendar handling, itinerary
  construction, Pareto filtering, and agent behavior.

## Quick start

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) with a Gemma model, or an OpenRouter API key
- A [SerpAPI](https://serpapi.com/) API key


### Install

```bash
uv sync
```

Create a `.env` file in the project directory:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b

# Optional fallback/provider:
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free

SERPAPI_API_KEY=your_serpapi_key
```

AirGem loads `.env` automatically. Sourcing it manually is optional.

For local inference:

```bash
ollama pull gemma3:4b
ollama run gemma3:4b
```

The free OpenRouter model can be rate-limited by its upstream provider. Ollama
is the preferred development path when it is available locally.

## CLI examples

One-way search:

```bash
uv run airgem \
  "Find cheap flights from GRU to POA on 2027-01-01"
```

Round trip:

```bash
uv run airgem \
  "Find cheap flights from GRU to POA on 2027-01-01, returning 2027-01-08"
```

Use OpenRouter instead of Ollama:

```bash
uv run airgem --provider openrouter \
  "Find the cheapest nonstop flight from GRU to POA on 2027-01-01"
```

Use a local calendar export with a one-hour travel buffer:

```bash
uv run airgem \
  --calendar ./calendar.ics \
  --buffer-minutes 60 \
  "Find cheap flights from GRU to POA on 2027-01-01, returning 2027-01-08"
```

Print machine-readable output:

```bash
uv run airgem --json \
  "Find cheap flights from GRU to POA on 2027-01-01"
```

Print safe state and tool decisions:

```bash
uv run airgem --debug \
  "Find cheap flights from GRU to POA on 2027-01-01, returning 2027-01-08"
```

Debug output is intended for humans. It does not print hidden model reasoning,
API keys, or SerpAPI departure tokens. Do not combine `--debug` with machine
parsing.

## Local web UI

Run the small local web server when testing follow-up session behavior:

```bash
uv run web
```

Open <http://127.0.0.1:8000>. The page keeps an in-memory session so follow-ups
such as “try the other outbound” can expand deferred round-trip options. No
chat history or database is used; sessions disappear when the server stops.

Equivalent direct command:

```bash
uv run uvicorn airgem.web:app --reload
```

## What can the user ask for?

| User request | Example | Effect |
| --- | --- | --- |
| Route | `from GRU to POA` | Sets the origin and destination IATA codes. |
| One-way trip | `fly on 2027-01-01` | Searches only the outbound leg. |
| Round trip | `returning 2027-01-08` | Searches a native round trip and expands the selected return. |
| Maximum budget | `under 300 USD` | Removes itineraries above the total price limit. |
| Currency | `show prices in BRL` | Requests prices in a three-letter currency code. Defaults to USD. |
| Nonstop | `nonstop only` | Sets the maximum number of stops to zero. |
| Stop limit | `at most one stop` | Removes options with more than one stop. |
| Price preference | `find the cheapest` | Sorts the provider search by price and tells the selector to prefer price. |

Examples:

```bash
# Round trip, hard budget, nonstop
uv run airgem \
  "Find a nonstop round trip from GRU to POA, leaving 2027-01-01 and returning 2027-01-08, under 1500 BRL"

# One way, one-stop maximum, explicit currency
uv run airgem \
  "Find the cheapest flight from GRU to POA on 2027-01-01, at most one stop, in USD"

# Budget and schedule preference
uv run airgem \
  "Find an affordable flight from GRU to POA on 2027-01-01, under 250 USD, preferring fewer stops"
```

### Calendar constraints

Calendar availability is supplied through CLI options rather than the prompt:

```bash
uv run airgem \
  --calendar ./calendar.ics \
  --buffer-minutes 90 \
  "Find a cheap round trip from GRU to POA on 2027-01-01, returning 2027-01-08"
```

The buffer is a hard availability constraint: the itinerary must leave enough
time before departure and after arrival around every busy calendar event.

### Be explicit in the first slice

Use ISO dates such as `2027-01-01` for reproducible searches. Relative dates
such as “next weekend” are passed to Gemma with the current date, but exact
date resolution is still model-generated.

## Local calendar setup

Export a calendar from Google Calendar, Apple Calendar, Outlook, or another
calendar application as an `.ics` file. One file can contain multiple weeks
or months; there is no special weekly or monthly mode.

For a short experiment, export the date range containing the requested trip,
plus nearby events that could overlap the configured buffer. Recurring events
are expanded within the requested flight window automatically.

```bash
uv run airgem \
  --calendar ./calendar.ics \
  --buffer-minutes 60 \
  "Find cheap flights from GRU to POA on 2027-01-01, returning 2027-01-08"
```

AirGem treats an itinerary as unavailable when either leg overlaps a calendar
event after applying the buffer. The filtering happens before Pareto selection.
The current calendar layer supports:

- Timed events.
- All-day events.
- Daily and other recurring events supported by the iCalendar recurrence
  library.
- Cancelled and transparent events being ignored.

Calendar data stays local. Only sanitized flight-search fields such as airport
codes and dates are sent to SerpAPI.

The current limitation is timezone normalization between calendar locations and
flight locations. For reliable first-slice results, use calendar times matching
the local times represented by the flight search results.

## Agent and deterministic core

The agent is intentionally small and bounded. It does not run an open-ended
conversation loop.

For a round trip it performs:

1. Parse the request with Gemma.
2. Search native round-trip candidates.
3. Apply hard constraints and calendar availability checks.
4. Build the outbound Pareto set.
5. Ask Gemma to select one allowed outbound candidate.
6. Expand only that candidate's return-leg options.
7. Apply return-date and calendar checks again.
8. Ask Gemma to select one return candidate.
9. Fall back to the deterministic Pareto order if model output is invalid.

The selector must return an exact JSON object containing an allowed candidate
ID. This keeps model freedom at the decision boundary while preventing
invented flights or opaque provider tokens from entering the result.

## Project structure

```text
airgem/
├── src/airgem/
│   ├── llm/
│   │   ├── ollama.py          # Local Ollama adapter
│   │   ├── openrouter.py      # OpenRouter fallback adapter
│   │   └── provider.py        # Shared LLM contract and HTTP helper
│   ├── core/
│   │   ├── flight_request.py  # Validated normalized request
│   │   └── parse.py           # Natural language → request
│   ├── calendar.py            # .ics parsing and recurrence expansion
│   ├── application.py         # Shared CLI/web application orchestration
│   ├── agent.py               # Bounded candidate-selection state machine
│   ├── web.py                 # Minimal FastAPI server and session endpoints
│   ├── web/index.html         # Minimal local chat UI
│   ├── itinerary.py           # Pairing and hard constraints
│   ├── pareto.py              # Multi-objective candidate filtering
│   └── search/
│       ├── provider.py        # Provider-neutral flight contract
│       └── serpapi.py         # Google Flights SerpAPI adapter
├── tests/
├── README.md
└── pyproject.toml
```

## Tests

Run the offline test suite:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

The tests do not call Ollama, OpenRouter, or SerpAPI. Provider responses are
captured with fakes so the deterministic core can be tested without API costs.

## Roadmap

Near-term work:

- Mark the selected itinerary in human-readable CLI output.
- Reduce duplicate/tied candidate display.
- Improve timezone normalization for calendar and airport-local times.
- Add final Gemma trade-off synthesis for the Pareto set.
- Add retry/backoff and clearer provider error handling.

Later work:

- Google Calendar API integration as an optional adapter.
- Flexible-date search.
- Multiple-airport searches.
- Booking URL resolution after itinerary selection.
- Additional search providers.
- Hosted deployment and Sentry observability.

Google Calendar is intentionally not required for the first slice. An API
integration would add OAuth, scopes, token storage, refresh behavior, and more
privacy considerations. It can later produce the same local busy-interval
objects used by the optimizer.

## License

AirGem is released under the MIT License.

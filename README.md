# AirGem local setup

## Install

```bash
uv sync
source .env
```

Run Ollama locally, then export `SERPAPI_API_KEY` and the other values described
in `PROJECT.md`.

## Use a local calendar

Export a calendar from your calendar application as an `.ics` file. The file
can contain multiple events; it does not need to be weekly or monthly. Export
the date range that covers the flight search, plus nearby events needed for
buffer checks. A current-calendar export is enough for a short experiment.

Then pass it to the CLI:

```bash
uv run airgem \
  --calendar ./calendar.ics \
  --buffer-minutes 60 \
  "Find cheap flights from GRU to POA on 2026-11-06, returning 2026-11-08"
```

AirGem filters flights against calendar events before Pareto filtering. The
buffer applies before departure and after arrival on each leg. The first slice
supports ordinary timed and all-day `VEVENT` entries, cancelled events,
transparent events, and recurring events within the flight date window.
Timezone conversion between calendar locations and flight locations is not
implemented yet; use calendar times matching the local flight times.

Google Calendar integration can come later. It should fetch events into this
same local `BusyInterval` shape, without changing search or optimization.

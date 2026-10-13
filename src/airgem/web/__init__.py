from __future__ import annotations

import os
from html import escape
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import File, Form, FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from airgem.application import ApplicationSession, continue_session, start_session
from airgem.llm import OllamaProvider, OpenRouterProvider
from airgem.search import SerpApiProvider

app = FastAPI(title="AirGem")
_sessions: dict[str, ApplicationSession] = {}
_web_dir = Path(__file__).parent
_index_path = _web_dir / "index.html"


class SearchInput(BaseModel):
    prompt: str = Field(min_length=1)
    provider: Literal["ollama", "openrouter"] | None = None
    calendar_path: str | None = None
    buffer_minutes: int = Field(default=60, ge=0)


class MessageInput(BaseModel):
    prompt: str = Field(min_length=1)


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/ui", include_in_schema=False)
def ui_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/ui/", response_class=HTMLResponse, include_in_schema=False)
def ui_index() -> str:
    return _index_path.read_text(encoding="utf-8")


@app.post("/api/search")
def search(payload: SearchInput) -> dict[str, object]:
    try:
        session, result = start_session(
            payload.prompt,
            _llm_provider(payload.provider),
            SerpApiProvider(),
            buffer_minutes=payload.buffer_minutes,
            calendar_path=payload.calendar_path,
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    session_id = uuid4().hex
    _sessions[session_id] = session
    return {"session_id": session_id, **result}


@app.post("/api/sessions/{session_id}/messages")
def message(session_id: str, payload: MessageInput) -> dict[str, object]:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    try:
        result = continue_session(session, payload.prompt)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"session_id": session_id, **result}


@app.post("/ui/search", response_class=HTMLResponse, include_in_schema=False)
async def ui_search(
    prompt: Annotated[str, Form(min_length=1)],
    provider: Annotated[Literal["ollama", "openrouter"] | None, Form()] = None,
    buffer_minutes: Annotated[int, Form(ge=0)] = 60,
    calendar: Annotated[UploadFile | None, File()] = None,
) -> HTMLResponse:
    try:
        calendar_text = await _calendar_text(calendar)
        session, result = start_session(
            prompt,
            _llm_provider(provider),
            SerpApiProvider(),
            buffer_minutes=buffer_minutes,
            calendar_text=calendar_text,
        )
    except (RuntimeError, ValueError) as error:
        return HTMLResponse(_error_fragment(str(error)))
    session_id = uuid4().hex
    _sessions[session_id] = session
    return HTMLResponse(_workspace_fragment(session_id, result))


@app.post("/ui/sessions/{session_id}/messages", response_class=HTMLResponse, include_in_schema=False)
def ui_message(session_id: str, prompt: Annotated[str, Form(min_length=1)]) -> HTMLResponse:
    session = _sessions.get(session_id)
    if session is None:
        return HTMLResponse(_error_fragment("session not found"))
    try:
        result = continue_session(session, prompt)
    except (RuntimeError, ValueError) as error:
        return HTMLResponse(_error_fragment(str(error)))
    return HTMLResponse(_workspace_fragment(session_id, result))


def main() -> None:
    import uvicorn

    uvicorn.run("airgem.web:app", host="127.0.0.1", port=8000, reload=True)


def _llm_provider(name: str | None):
    selected = name or os.getenv("LLM_PROVIDER", "ollama")
    if selected == "ollama":
        return OllamaProvider()
    if selected == "openrouter":
        return OpenRouterProvider()
    raise ValueError(f"unknown LLM provider: {selected}")


async def _calendar_text(calendar: UploadFile | None) -> str | None:
    if calendar is None or not calendar.filename:
        return None
    contents = await calendar.read(2_000_001)
    if len(contents) > 2_000_000:
        raise ValueError("calendar file is too large")
    try:
        return contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("calendar file must be UTF-8 iCalendar data") from error


def _workspace_fragment(session_id: str, payload: dict[str, object]) -> str:
    request = payload["request"]
    assert isinstance(request, dict)
    itineraries = payload["itineraries"]
    assert isinstance(itineraries, list)
    alternatives = set(payload["alternatives"])
    cards = "".join(
        _itinerary_card(session_id, item, payload, alternatives) for item in itineraries
    )
    deferred = payload["deferred_outbound"]
    assert isinstance(deferred, list)
    deferred_html = ""
    if deferred:
        deferred_html = (
            "<h3>Other outbound candidates</h3><div class='cards'>"
            + "".join(_itinerary_card(session_id, item, payload, set()) for item in deferred)
            + "</div>"
        )
    notes = "".join(f"<li>{_text(note)}</li>" for note in payload["notes"])
    return f"""
    <div class="workspace">
      {_messages_html(payload["messages"])}
      <div class="results-header">
        <h2>{_text(request["origin"])} → {_text(request["destination"])}</h2>
        <span class="muted">{_text(request["departure_date"])}{_text(" → " + request["return_date"] if request.get("return_date") else "")}</span>
      </div>
      <h3>{"Current options" if itineraries else "No matching options"}</h3>
      {f'<div class="cards">{cards}</div>' if cards else '<p class="empty">No flights matched the current constraints.</p>'}
      {deferred_html}
      {_rationale_html(payload.get("rationale"))}
      {f'<ul class="notes">{notes}</ul>' if notes else ''}
      <section class="panel follow-up">
        <form hx-post="/ui/sessions/{_attr(session_id)}/messages" hx-target="#workspace" hx-swap="innerHTML" hx-indicator="#follow-up-searching">
          <label for="follow-up-prompt">Continue the search</label>
          <div class="follow-up-row">
            <input id="follow-up-prompt" name="prompt" required placeholder="Try ‘show the next outbound’ or ‘prefer nonstop flights’">
            <button type="submit">Ask</button>
          </div>
          <span id="follow-up-searching" class="busy muted">AirGem is checking the options…</span>
        </form>
      </section>
    </div>
    """


def _messages_html(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    items = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        label = "You" if role == "user" else "AirGem"
        css = "user-message" if role == "user" else "assistant-message"
        items.append(
            f'<div class="message {css}"><span class="message-label">{label}</span>{_text(message.get("content", ""))}</div>'
        )
    return f'<div class="chat">{"".join(items)}</div>'


def _itinerary_card(
    session_id: str,
    item: object,
    payload: dict[str, object],
    alternatives: set[object],
) -> str:
    if not isinstance(item, dict):
        return ""
    item_id = str(item.get("id", ""))
    selected = payload.get("selected")
    selected_id = selected.get("id") if isinstance(selected, dict) else None
    badge = ""
    if item_id == selected_id:
        badge = '<span class="badge">Selected</span>'
    elif item_id in alternatives:
        badge = '<span class="badge">Alternative</span>'
    actions = []
    for action in payload.get("available_actions", []):
        if isinstance(action, dict) and item_id in str(action.get("prompt", "")):
            actions.append(
                f'<form hx-post="/ui/sessions/{_attr(session_id)}/messages" hx-target="#workspace" hx-swap="innerHTML">'
                f'<input type="hidden" name="prompt" value="{_attr(action.get("prompt", ""))}">'
                f'<button class="secondary" type="submit">{_text(action.get("label", "Explore"))}</button></form>'
            )
    outbound = item.get("outbound")
    inbound = item.get("inbound")
    if not isinstance(outbound, dict):
        return ""
    airlines = [outbound.get("airline")]
    if isinstance(inbound, dict):
        airlines.append(inbound.get("airline"))
    airline = " + ".join(dict.fromkeys(str(name) for name in airlines if name)) or "Airline pending"
    return f"""
    <article class="flight{' selected' if item_id == selected_id else ''}">
      <div class="flight-title"><strong>{_text(item_id)} · {_text(airline)}</strong>{badge}</div>
      <div class="price">{_text(item.get("currency", "USD"))} {float(item.get("price", 0)):.2f}</div>
      {_leg_html("Outbound", outbound)}
      {_leg_html("Return", inbound) if isinstance(inbound, dict) else '<div class="muted">Return details pending</div>' if item.get("return_pending") else ''}
      {f'<div class="flight-actions">{"".join(actions)}</div>' if actions else ''}
    </article>
    """


def _leg_html(label: str, leg: dict[str, object]) -> str:
    return f"""
    <div class="leg">
      <strong>{_text(label)} · {_text(leg.get("origin"))} → {_text(leg.get("destination"))}</strong>
      <div class="muted">{_text(leg.get("departure"))} · {_text(leg.get("airline") or "Airline pending")}</div>
      <div class="muted">{_stops(leg.get("stops"))} · {_duration(leg.get("duration_minutes"))}</div>
    </div>
    """


def _rationale_html(rationale: object) -> str:
    if not isinstance(rationale, dict):
        return ""
    metrics = rationale.get("metrics", {})
    constraints = rationale.get("constraints", {})
    if not isinstance(metrics, dict) or not isinstance(constraints, dict):
        return ""
    facts = [
        f"Compared {rationale.get('evaluated_candidates', 0)} candidate(s)",
        f"Price: {metrics.get('currency', 'USD')} {float(metrics.get('price', 0)):.2f}",
        f"Duration: {_duration(metrics.get('duration_minutes'))}",
        f"Stops: {_stops(metrics.get('stops'))}",
    ]
    if constraints.get("max_price") is not None:
        facts.append(f"Budget limit: {metrics.get('currency', 'USD')} {float(constraints['max_price']):.2f}")
    if constraints.get("max_stops") is not None:
        facts.append(f"Stop limit: {constraints['max_stops']}")
    if constraints.get("calendar_events"):
        facts.append(f"Calendar checked with {constraints.get('buffer_minutes', 0)}-minute buffer")
    return '<section class="facts"><h3>Why this option</h3><ul>' + "".join(f"<li>{_text(fact)}</li>" for fact in facts) + "</ul></section>"


def _duration(minutes: object) -> str:
    if minutes is None:
        return "duration pending"
    total = int(minutes)
    hours, remainder = divmod(total, 60)
    return f"{hours}h {remainder:02d}m" if hours else f"{remainder}m"


def _stops(stops: object) -> str:
    if stops is None:
        return "stops pending"
    count = int(stops)
    return "nonstop" if count == 0 else f"{count} stop{'s' if count != 1 else ''}"


def _text(value: object) -> str:
    return escape(str(value))


def _attr(value: object) -> str:
    return escape(str(value), quote=True)


def _error_fragment(message: str) -> str:
    return f'<div class="error" role="alert">{_text(message)}</div>'

from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4
from urllib.parse import urlsplit
from urllib.request import Request as UrlRequest, urlopen

from fastapi import File, Form, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from airgem.application import ApplicationSession, continue_session, start_session
from airgem.llm import OllamaProvider, OpenRouterProvider
from airgem.search import BookingAction, SerpApiProvider

app = FastAPI(title="AirGem")
_SESSION_TTL_SECONDS = 30 * 60
_JOB_TTL_SECONDS = 10 * 60
_RATE_WINDOW_SECONDS = 60 * 60
_RATE_LIMIT_PER_IP = 10
_RATE_LIMIT_GLOBAL = 100
_MAX_PROMPT_LENGTH = 1_000
_MAX_FOLLOW_UP_LENGTH = 500
_MAX_CALENDAR_BYTES = 512_000
_owner_salt = os.urandom(32)
_state_lock = threading.Lock()
_rate_events: dict[str, deque[float]] = {}
_global_rate_events: deque[float] = deque()
_sessions: dict[str, "SessionEntry"] = {}
_jobs: dict[str, "WebJob"] = {}
_booking_actions: dict[str, "BookingEntry"] = {}
_job_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="airgem-web")
_web_dir = Path(__file__).parent
_index_path = _web_dir / "index.html"


@dataclass
class WebJob:
    owner: str
    expires_at: float
    status: Literal["queued", "running", "complete", "error"] = "queued"
    stage: str = "queued"
    html: str | None = None
    error: str | None = None
    completed_stages: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update_stage(self, stage: str) -> None:
        with self._lock:
            if self.stage not in {"queued", stage} and self.stage not in self.completed_stages:
                self.completed_stages.append(self.stage)
            self.status = "running"
            self.stage = stage

    def complete(self, html: str) -> None:
        with self._lock:
            if self.stage not in {"queued", "complete"} and self.stage not in self.completed_stages:
                self.completed_stages.append(self.stage)
            self.status = "complete"
            self.stage = "complete"
            self.html = html

    def fail(self, error: str) -> None:
        with self._lock:
            if self.stage not in {"queued", "error"} and self.stage not in self.completed_stages:
                self.completed_stages.append(self.stage)
            self.status = "error"
            self.stage = "error"
            self.error = error

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "status": self.status,
                "stage": self.stage,
                "completed_stages": list(self.completed_stages),
                "html": self.html,
                "error": self.error,
            }


@dataclass
class SessionEntry:
    owner: str
    session: ApplicationSession
    expires_at: float


@dataclass
class BookingEntry:
    owner: str
    action: BookingAction
    expires_at: float


_JOB_STEPS = (
    ("parsing_request", "Parse request"),
    ("loading_calendar", "Load calendar"),
    ("searching", "Find flights"),
    ("evaluating_options", "Evaluate options"),
    ("selecting_outbound", "Select outbound"),
    ("expanding_return", "Expand return"),
    ("evaluating_return_options", "Evaluate return"),
    ("selecting_return", "Select return"),
    ("fetching_booking", "Prepare booking handoff"),
)
_STAGE_DETAILS = {
    "queued": (0, "Getting ready…"),
    "parsing_request": (12, "Parsing your request…"),
    "loading_calendar": (24, "Loading your calendar locally…"),
    "searching": (42, "Finding flight options…"),
    "evaluating_options": (57, "Evaluating price, stops, and calendar fit…"),
    "selecting_outbound": (70, "Selecting the best outbound…"),
    "expanding_return": (80, "Expanding return options…"),
    "evaluating_return_options": (87, "Evaluating return options…"),
    "selecting_return": (94, "Selecting the best return…"),
    "fetching_booking": (97, "Preparing the booking handoff…"),
    "complete": (100, "Results are ready."),
    "error": (100, "AirGem could not complete the search."),
}


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=_MAX_PROMPT_LENGTH)
    serpapi_key: SecretStr = Field(min_length=8, max_length=256)
    provider: Literal["ollama", "openrouter"] | None = None
    buffer_minutes: int = Field(default=60, ge=0)


class MessageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=_MAX_FOLLOW_UP_LENGTH)


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ui", include_in_schema=False)
def ui_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/ui/", response_class=HTMLResponse, include_in_schema=False)
def ui_index() -> str:
    return _index_path.read_text(encoding="utf-8")


@app.get("/ui/jobs/{job_id}")
def ui_job_status(job_id: str, request: Request) -> dict[str, object]:
    owner = _client_owner(request)
    _cleanup_expired()
    with _state_lock:
        job = _jobs.get(job_id)
    if job is None or job.owner != owner:
        raise HTTPException(status_code=404, detail="job not found")
    payload = job.snapshot()
    stage = str(payload["stage"])
    progress, label = _STAGE_DETAILS.get(stage, _STAGE_DETAILS["queued"])
    if payload["status"] == "error":
        payload["html"] = _error_fragment(str(payload["error"] or "search failed"))
    return {**payload, "progress": progress, "label": label}


@app.post("/api/search")
def search(payload: SearchInput, request: Request) -> dict[str, object]:
    owner = _client_owner(request)
    _enforce_rate_limit(owner)
    try:
        session, result = start_session(
            payload.prompt,
            _llm_provider(payload.provider),
            SerpApiProvider(api_key=payload.serpapi_key.get_secret_value()),
            buffer_minutes=payload.buffer_minutes,
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=_public_error(error)) from error
    session_id = uuid4().hex
    _store_session(session_id, owner, session)
    return {"session_id": session_id, **result}


@app.post("/api/sessions/{session_id}/messages")
def message(session_id: str, payload: MessageInput, request: Request) -> dict[str, object]:
    owner = _client_owner(request)
    _enforce_rate_limit(owner)
    session = _owned_session(session_id, owner)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    try:
        result = continue_session(session, payload.prompt)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=_public_error(error)) from error
    return {"session_id": session_id, **result}


@app.post("/ui/search", response_class=HTMLResponse, include_in_schema=False)
async def ui_search(
    request: Request,
    prompt: Annotated[str, Form(min_length=1, max_length=_MAX_PROMPT_LENGTH)],
    serpapi_key: Annotated[str, Form(min_length=8, max_length=256)],
    provider: Annotated[Literal["ollama", "openrouter"] | None, Form()] = None,
    buffer_minutes: Annotated[int, Form(ge=0)] = 60,
    calendar: Annotated[UploadFile | None, File()] = None,
) -> HTMLResponse:
    owner = _client_owner(request)
    try:
        _enforce_rate_limit(owner)
    except HTTPException as error:
        return HTMLResponse(
            _error_fragment(str(error.detail)),
            status_code=error.status_code,
            headers=error.headers,
        )
    try:
        calendar_text = await _calendar_text(calendar)
    except (RuntimeError, ValueError) as error:
        return HTMLResponse(_error_fragment(str(error)))
    job_id = _create_job(owner)
    _job_executor.submit(
        _run_search_job,
        job_id,
        prompt,
        provider,
        serpapi_key,
        buffer_minutes,
        calendar_text,
    )
    return HTMLResponse(_job_fragment(job_id))


@app.post("/ui/sessions/{session_id}/messages", response_class=HTMLResponse, include_in_schema=False)
def ui_message(
    session_id: str,
    request: Request,
    prompt: Annotated[str, Form(min_length=1, max_length=_MAX_FOLLOW_UP_LENGTH)],
) -> HTMLResponse:
    owner = _client_owner(request)
    try:
        _enforce_rate_limit(owner)
    except HTTPException as error:
        return HTMLResponse(
            _error_fragment(str(error.detail)),
            status_code=error.status_code,
            headers=error.headers,
        )
    session = _owned_session(session_id, owner)
    if session is None:
        return HTMLResponse(_error_fragment("session not found"))
    job_id = _create_job(owner)
    _job_executor.submit(_run_message_job, job_id, session_id, session, prompt, owner)
    return HTMLResponse(_job_fragment(job_id))


@app.post("/ui/bookings/{booking_id}", response_class=HTMLResponse, include_in_schema=False)
def ui_booking(booking_id: str, request: Request) -> Response:
    owner = _client_owner(request)
    _cleanup_expired()
    with _state_lock:
        entry = _booking_actions.get(booking_id)
    if entry is None or entry.owner != owner:
        raise HTTPException(status_code=404, detail="booking action not found")
    action = entry.action
    if not _is_https_url(action.url):
        return HTMLResponse(_error_fragment("booking provider returned an unsafe URL"), status_code=502)
    if not action.post_data:
        return RedirectResponse(action.url, status_code=303)

    try:
        request = UrlRequest(
            action.url,
            data=action.post_data.encode("utf-8"),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "AirGem/1.0",
            },
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            target = response.geturl()
            if target == action.url:
                target = _meta_refresh_url(response.read(65_536)) or ""
        if not _is_https_url(target):
            raise RuntimeError("booking provider did not return a safe destination")
        return RedirectResponse(target, status_code=303)
    except Exception:
        return HTMLResponse(_error_fragment("booking handoff failed"), status_code=502)


def main() -> None:
    import uvicorn

    uvicorn.run("airgem.web:app", host="127.0.0.1", port=8000, reload=True)


def _create_job(owner: str) -> str:
    _cleanup_expired()
    job_id = uuid4().hex
    with _state_lock:
        _jobs[job_id] = WebJob(owner, time.monotonic() + _JOB_TTL_SECONDS)
    return job_id


def _run_search_job(
    job_id: str,
    prompt: str,
    provider: str | None,
    serpapi_key: str,
    buffer_minutes: int,
    calendar_text: str | None,
) -> None:
    job = _jobs[job_id]
    try:
        session, result = start_session(
            prompt,
            _llm_provider(provider),
            SerpApiProvider(api_key=serpapi_key),
            buffer_minutes=buffer_minutes,
            calendar_text=calendar_text,
            stage_callback=job.update_stage,
        )
        session_id = uuid4().hex
        _store_session(session_id, job.owner, session)
        job.complete(_workspace_fragment(session_id, result, job.owner))
    except Exception as error:
        job.fail(_public_error(error, stage=str(job.snapshot()["stage"])))


def _run_message_job(
    job_id: str,
    session_id: str,
    session: ApplicationSession,
    prompt: str,
    owner: str,
) -> None:
    job = _jobs[job_id]
    job.update_stage("selecting_outbound")
    try:
        result = continue_session(session, prompt, stage_callback=job.update_stage)
        job.complete(_workspace_fragment(session_id, result, owner))
    except Exception as error:
        job.fail(_public_error(error, stage=str(job.snapshot()["stage"])))


def _job_fragment(job_id: str) -> str:
    steps = "".join(
        f'<li data-stage="{_attr(stage)}"><span class="step-dot" aria-hidden="true"></span>{_text(label)}</li>'
        for stage, label in _JOB_STEPS
    )
    return f"""
    <section class="panel agent-job" data-job-id="{_attr(job_id)}" aria-live="polite">
      <div class="agent-spinner"><span class="spinner" aria-hidden="true"></span></div>
      <div class="agent-job-copy">
        <p class="job-kicker">AirGem is working</p>
        <p class="job-stage" data-job-stage>Getting ready…</p>
        <div class="job-progress" aria-hidden="true"><span data-job-progress style="width: 0%"></span></div>
        <ol class="job-steps">{steps}</ol>
      </div>
    </section>
    """


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
    contents = await calendar.read(_MAX_CALENDAR_BYTES + 1)
    if len(contents) > _MAX_CALENDAR_BYTES:
        raise ValueError("calendar file is too large")
    try:
        return contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("calendar file must be UTF-8 iCalendar data") from error


def _workspace_fragment(session_id: str, payload: dict[str, object], owner: str) -> str:
    session = _owned_session(session_id, owner)
    booking_action_id = (
        _register_booking_action(session.booking_action, owner)
        if session and session.booking_action
        else None
    )
    request = payload["request"]
    assert isinstance(request, dict)
    itineraries = payload["itineraries"]
    assert isinstance(itineraries, list)
    alternatives = set(payload["alternatives"])
    selected_id = payload.get("selected", {}).get("id") if isinstance(payload.get("selected"), dict) else None
    selected_item = next(
        (item for item in itineraries if isinstance(item, dict) and item.get("id") == selected_id),
        None,
    )
    other_items = [item for item in itineraries if item is not selected_item]
    selected_html = (
        '<section class="primary-option"><div class="section-label">Recommended itinerary</div>'
        f'<div class="cards featured-cards">{_itinerary_card(session_id, selected_item, payload, alternatives, booking_action_id)}</div></section>'
        if selected_item is not None
        else ""
    )
    alternatives_html = (
        '<section class="alternatives-section"><div class="section-label">Other options</div>'
        f'<div class="cards">{"".join(_itinerary_card(session_id, item, payload, alternatives, booking_action_id) for item in other_items)}</div></section>'
        if other_items
        else ""
    )
    deferred = payload["deferred_outbound"]
    assert isinstance(deferred, list)
    deferred_html = ""
    if deferred:
        deferred_html = (
            "<h3>Other outbound candidates</h3><div class='cards'>"
            + "".join(_itinerary_card(session_id, item, payload, set(), booking_action_id) for item in deferred)
            + "</div>"
        )
    notes = "".join(f"<li>{_text(note)}</li>" for note in payload["notes"])
    return f"""
    <div class="workspace" data-session-id="{_attr(session_id)}">
      {_messages_html(payload["messages"])}
      <div class="results-header">
        <h2>{_text(request["origin"])} → {_text(request["destination"])}</h2>
        <span class="muted">{_text(request["departure_date"])}{_text(" → " + request["return_date"] if request.get("return_date") else "")}</span>
      </div>
      {selected_html}{alternatives_html if itineraries else '<p class="empty">No flights matched the current constraints.</p>'}
      {deferred_html}
      {_rationale_html(payload.get("rationale"))}
      {f'<ul class="notes">{notes}</ul>' if notes else ''}
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
    booking_action_id: str | None = None,
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
    booking = ""
    if item_id == selected_id:
        if booking_action_id is not None:
            booking = (
                f'<form action="/ui/bookings/{_attr(booking_action_id)}" method="post" target="_blank">'
                '<button class="booking-button" type="submit">Open booking ↗</button></form>'
            )
        else:
            booking = _booking_link(item.get("booking_url"))
    action_html = "".join(actions) + booking
    outbound = item.get("outbound")
    inbound = item.get("inbound")
    if not isinstance(outbound, dict):
        return ""
    airlines = [outbound.get("airline")]
    if isinstance(inbound, dict):
        airlines.append(inbound.get("airline"))
    airline = " + ".join(dict.fromkeys(str(name) for name in airlines if name)) or "Airline pending"
    return f"""
    <article class="flight ticket-card{' selected' if item_id == selected_id else ''}">
      <div class="ticket-main">
        <div class="flight-title"><div><span class="ticket-kicker">{_text(item_id)}</span><strong>{_text(airline)}</strong></div>{badge}</div>
        <div class="price">{_text(item.get("currency", "USD"))} {float(item.get("price", 0)):.2f}</div>
        {_leg_html("Outbound", outbound)}
        {_leg_html("Return", inbound) if isinstance(inbound, dict) else '<div class="muted">Return details pending</div>' if item.get("return_pending") else ''}
      </div>
      <aside class="ticket-stub"><span>AirGem</span><span class="ticket-dash" aria-hidden="true"></span><span class="ticket-mark" aria-hidden="true">✦</span></aside>
      {f'<div class="flight-actions">{action_html}</div>' if action_html else ''}
    </article>
    """


def _leg_html(label: str, leg: dict[str, object]) -> str:
    departure_time, departure_date, departure_zone = _time_parts(leg.get("departure"))
    arrival_time, arrival_date, arrival_zone = _time_parts(leg.get("arrival"))
    zone_note = departure_zone if departure_zone == arrival_zone else f"{departure_zone} → {arrival_zone}"
    return f"""
    <div class="leg">
      <div class="leg-header"><strong>{_text(label)}</strong><span class="muted">{_stops(leg.get("stops"))} · {_duration(leg.get("duration_minutes"))}</span></div>
      <div class="leg-times">
        <div class="airport-time"><strong>{_text(departure_time)}</strong><span>{_text(departure_date)}</span><b>{_text(leg.get("origin"))}</b></div>
        <div class="leg-track" aria-hidden="true"><span>→</span></div>
        <div class="airport-time arrival-time"><strong>{_text(arrival_time)}</strong><span>{_text(arrival_date)}</span><b>{_text(leg.get("destination"))}</b></div>
      </div>
      <div class="timezone-note">{_text(zone_note)} · {_text(leg.get("airline") or "Airline pending")}</div>
    </div>
    """


def _time_parts(value: object) -> tuple[str, str, str]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return str(value), "", "Time as returned"
    date_label = parsed.strftime("%a, %b %d").replace(" 0", " ")
    if parsed.utcoffset() is None:
        zone_label = "Local airport time"
    else:
        offset = parsed.strftime("%z")
        zone_label = f"UTC{offset[:3]}:{offset[3:]}"
    return parsed.strftime("%H:%M"), date_label, zone_label


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


def _booking_link(value: object) -> str:
    if not isinstance(value, str):
        return ""
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.netloc:
        return ""
    return (
        f'<a class="booking-link" href="{_attr(value)}" target="_blank" '
        'rel="noopener noreferrer">Open booking ↗</a>'
    )


def _register_booking_action(action: BookingAction, owner: str) -> str:
    booking_id = uuid4().hex
    with _state_lock:
        _booking_actions[booking_id] = BookingEntry(
            owner,
            action,
            time.monotonic() + _SESSION_TTL_SECONDS,
        )
    return booking_id


def _client_owner(request: Request) -> str:
    host = request.client.host if request.client else "unknown"
    return sha256(_owner_salt + host.encode("utf-8")).hexdigest()


def _store_session(session_id: str, owner: str, session: ApplicationSession) -> None:
    _cleanup_expired()
    with _state_lock:
        _sessions[session_id] = SessionEntry(
            owner,
            session,
            time.monotonic() + _SESSION_TTL_SECONDS,
        )


def _owned_session(session_id: str, owner: str) -> ApplicationSession | None:
    _cleanup_expired()
    with _state_lock:
        entry = _sessions.get(session_id)
        if entry is None or entry.owner != owner:
            return None
        entry.expires_at = time.monotonic() + _SESSION_TTL_SECONDS
        return entry.session


def _cleanup_expired(now: float | None = None) -> None:
    current = time.monotonic() if now is None else now
    with _state_lock:
        for store in (_sessions, _jobs, _booking_actions):
            expired = [key for key, value in store.items() if value.expires_at <= current]
            for key in expired:
                del store[key]


def _enforce_rate_limit(owner: str, now: float | None = None) -> None:
    current = time.monotonic() if now is None else now
    cutoff = current - _RATE_WINDOW_SECONDS
    with _state_lock:
        while _global_rate_events and _global_rate_events[0] <= cutoff:
            _global_rate_events.popleft()
        for key, prior_events in list(_rate_events.items()):
            while prior_events and prior_events[0] <= cutoff:
                prior_events.popleft()
            if not prior_events:
                del _rate_events[key]
        events = _rate_events.setdefault(owner, deque())
        if len(events) >= _RATE_LIMIT_PER_IP or len(_global_rate_events) >= _RATE_LIMIT_GLOBAL:
            raise HTTPException(
                status_code=429,
                detail="Demo request limit reached. Try again later.",
                headers={"Retry-After": str(_RATE_WINDOW_SECONDS)},
            )
        events.append(current)
        _global_rate_events.append(current)


def _public_error(error: Exception, *, stage: str | None = None) -> str:
    message = str(error)
    if message.startswith("missing required flight fields:"):
        return message
    if stage == "parsing_request" or message.startswith(("LLM request failed", "OPENROUTER_")):
        return "The hosted AI provider failed. Check the OpenRouter configuration or try again shortly."
    if stage in {
        "searching",
        "expanding_return",
        "fetching_booking",
    } or message.startswith("SerpApi"):
        return "Flight search failed. Check your SerpApi key and available credits, then try again."
    return "Search failed while evaluating the options. Please try again shortly."


def _is_https_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parts = urlsplit(value)
    return parts.scheme == "https" and bool(parts.netloc)


class _MetaRefreshParser(HTMLParser):
    target: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag != "meta" or (values.get("http-equiv") or "").lower() != "refresh":
            return
        _, separator, directive = (values.get("content") or "").partition(";")
        name, equals, target = directive.partition("=")
        if separator and equals and name.strip().lower() == "url":
            self.target = target.strip().strip("\"'")


def _meta_refresh_url(body: bytes) -> str | None:
    parser = _MetaRefreshParser()
    parser.feed(body.decode("latin-1"))
    return parser.target


def _error_fragment(message: str) -> str:
    return f'<div class="error" role="alert">{_text(message)}</div>'

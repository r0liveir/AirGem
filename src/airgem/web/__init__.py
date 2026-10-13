from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from airgem.application import ApplicationSession, continue_session, start_session
from airgem.llm import OllamaProvider, OpenRouterProvider
from airgem.search import SerpApiProvider

app = FastAPI(title="AirGem")
_sessions: dict[str, ApplicationSession] = {}
_index_path = Path(__file__).with_name("index.html")


class SearchInput(BaseModel):
    prompt: str = Field(min_length=1)
    provider: Literal["ollama", "openrouter"] | None = None
    calendar_path: str | None = None
    buffer_minutes: int = Field(default=60, ge=0)


class MessageInput(BaseModel):
    prompt: str = Field(min_length=1)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
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

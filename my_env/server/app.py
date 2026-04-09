from __future__ import annotations

import os
import sys
import threading
from uuid import uuid4
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, Cookie, FastAPI, Header, Response
from pydantic import BaseModel, Field

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_env.environment import InboxOpsEnvironment
from my_env.models import CounterfactualAnalysis, InboxOpsAction, InboxOpsObservation, InboxOpsState
from my_env.tasks import TASKS


class ResetRequest(BaseModel):
    seed: int | None = Field(default=None, ge=0)
    task_id: str | None = Field(default=None)


class ResetResponse(BaseModel):
    observation: InboxOpsObservation
    reward: float
    done: bool
    info: dict[str, Any]


class StepRequest(BaseModel):
    action: InboxOpsAction | str | None = None


class StepResponse(BaseModel):
    observation: InboxOpsObservation
    reward: float
    done: bool
    info: dict[str, Any]


class TaskCatalogResponse(BaseModel):
    tasks: list[dict[str, Any]]


SESSION_COOKIE_NAME = "inboxops_session_id"
_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[str, InboxOpsEnvironment] = {}


def _resolve_session_id(
    session_cookie: str | None,
    session_header: str | None,
) -> str | None:
    if session_header and session_header.strip():
        return session_header.strip()
    if session_cookie and session_cookie.strip():
        return session_cookie.strip()
    return None


def _get_or_create_env(session_id: str) -> InboxOpsEnvironment:
    with _SESSION_LOCK:
        env = _SESSIONS.get(session_id)
        if env is None:
            env = InboxOpsEnvironment()
            _SESSIONS[session_id] = env
        return env


def _ensure_session(
    response: Response,
    session_cookie: str | None,
    session_header: str | None,
) -> tuple[str, InboxOpsEnvironment]:
    session_id = _resolve_session_id(session_cookie, session_header) or str(uuid4())
    env = _get_or_create_env(session_id)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
    )
    return session_id, env

app = FastAPI(
    title="InboxOps OpenEnv Server",
    version="0.1.0",
    description="Minimal deterministic OpenEnv-compatible server for InboxOps.",
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "InboxOps",
        "status": "running",
        "mode": "deterministic benchmark plus analysis tooling",
        "endpoints": ["/reset", "/step", "/state", "/health", "/metadata", "/schema", "/tasks", "/analyze/current"],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return {
        "name": "InboxOps",
        "description": "Deterministic offline task triage environment with easy, medium, and hard tasks.",
        "version": "0.1.0",
        "task_count": 3,
        "features": [
            "deterministic grading",
            "counterfactual action analysis",
            "risk-tagged task metadata",
            "openenv-compatible fastapi surface",
        ],
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action": InboxOpsAction.model_json_schema(),
        "observation": InboxOpsObservation.model_json_schema(),
        "state": InboxOpsState.model_json_schema(),
    }


@app.get("/tasks", response_model=TaskCatalogResponse)
def tasks() -> TaskCatalogResponse:
    return TaskCatalogResponse(tasks=[task.model_dump() for task in TASKS])


@app.post("/reset", response_model=ResetResponse)
def reset(
    response: Response,
    request: ResetRequest | None = None,
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    session_header: str | None = Header(default=None, alias="X-Session-Id"),
) -> ResetResponse:
    session_id, env = _ensure_session(response, session_cookie, session_header)
    observation, info = env.reset(
        seed=request.seed if request else None,
        task_id=request.task_id if request else None,
    )
    return ResetResponse(
        observation=observation,
        reward=0.0,
        done=bool(observation.get("done", False)),
        info={**info, "session_id": session_id},
    )


@app.post("/step", response_model=StepResponse)
def step(
    response: Response,
    request: StepRequest | None = None,
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    session_header: str | None = Header(default=None, alias="X-Session-Id"),
) -> StepResponse:
    _session_id, env = _ensure_session(response, session_cookie, session_header)
    raw_action = None if request is None else request.action
    observation, reward, done, info = env.step(raw_action)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=InboxOpsState)
def state(
    response: Response,
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    session_header: str | None = Header(default=None, alias="X-Session-Id"),
) -> InboxOpsState:
    _session_id, env = _ensure_session(response, session_cookie, session_header)
    return env.state()


@app.get("/analyze/current", response_model=CounterfactualAnalysis | None)
def analyze_current(
    response: Response,
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    session_header: str | None = Header(default=None, alias="X-Session-Id"),
) -> CounterfactualAnalysis | None:
    _session_id, env = _ensure_session(response, session_cookie, session_header)
    return env.counterfactual_for_current_task()


@app.post("/mcp")
def mcp(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    request_id = payload.get("id")
    method = payload.get("method")

    if method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "state",
                    "description": "Return the current InboxOps environment state",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        }
    elif method == "tools/call":
        params = payload.get("params", {})
        tool_name = params.get("name")
        session_id = str(params.get("session_id") or "mcp-default")
        env = _get_or_create_env(session_id)
        if tool_name == "state":
            result = {"content": [{"type": "text", "text": str(env.state().model_dump())}]}
        else:
            result = {"content": [{"type": "text", "text": "unknown_tool"}]}
    else:
        result = {"capabilities": {"tools": True}}

    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    resolved_port = port if port is not None else int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=resolved_port)


if __name__ == "__main__":
    main()

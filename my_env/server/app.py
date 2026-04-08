from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_env.environment import InboxOpsEnvironment
from my_env.models import CounterfactualAnalysis, InboxOpsAction, InboxOpsObservation, InboxOpsState
from my_env.tasks import TASKS


class ResetRequest(BaseModel):
    seed: int | None = Field(default=None, ge=0)


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


_ENV = InboxOpsEnvironment()

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
def reset(request: ResetRequest | None = None) -> ResetResponse:
    observation, info = _ENV.reset(seed=request.seed if request else None)
    return ResetResponse(
        observation=observation,
        reward=0.0,
        done=bool(observation.get("done", False)),
        info=info,
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest | None = None) -> StepResponse:
    raw_action = None if request is None else request.action
    observation, reward, done, info = _ENV.step(raw_action)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=InboxOpsState)
def state() -> InboxOpsState:
    return _ENV.state()


@app.get("/analyze/current", response_model=CounterfactualAnalysis | None)
def analyze_current() -> CounterfactualAnalysis | None:
    return _ENV.counterfactual_for_current_task()


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
        if tool_name == "state":
            result = {"content": [{"type": "text", "text": str(_ENV.state().model_dump())}]}
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

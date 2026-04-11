from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException

from server.environment import InboxOpsEnvironment

app = FastAPI()

# single-process environment instance for demo/Space
_ENV = InboxOpsEnvironment()


@app.get("/")
def read_root() -> dict:
    return {
        "name": "InboxOps",
        "status": "running",
        "mode": "deterministic benchmark plus analysis tooling",
        "endpoints": [
            "/reset",
            "/step",
            "/state",
            "/health",
            "/metadata",
            "/schema",
            "/tasks",
            "/analyze/current",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset")
def reset(payload: dict | None = None) -> dict:
    payload = payload or {}
    seed = payload.get("seed") if isinstance(payload.get("seed"), int) else None
    task_id = payload.get("task_id") if isinstance(payload.get("task_id"), str) else None
    try:
        observation, info = _ENV.reset(seed=seed, task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"observation": observation, "reward": 0.0, "done": False, "info": info}


@app.post("/step")
def step(payload: dict) -> dict:
    action = payload.get("action") if isinstance(payload.get("action"), (dict, str)) else None
    observation, reward, done, info = _ENV.step(action)
    return {"observation": observation, "reward": reward, "done": done, "info": info}


@app.get("/state")
def state() -> dict:
    s = _ENV.state()
    return s.model_dump()


@app.get("/tasks")
def tasks() -> dict:
    from server.tasks import TASKS

    return {"tasks": [t.model_dump() for t in TASKS]}


@app.get("/analyze/current")
def analyze_current() -> dict:
    analysis = _ENV.counterfactual_for_current_task()
    if analysis is None:
        raise HTTPException(status_code=404, detail="no active task")
    return analysis.model_dump()


def main(host: str = "0.0.0.0", port: Optional[int] = None) -> None:
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port or 7860)


if __name__ == "__main__":
    main()

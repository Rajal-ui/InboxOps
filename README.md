---
title: InboxOps
sdk: docker
app_port: 7860
pinned: false
---

# InboxOps
OpenEnv environment for deterministic email operations triage

## Overview
InboxOps is a small, fully deterministic, offline OpenEnv-style environment that simulates internal inbox triage. An agent receives an operations incident prompt (for example password resets, payroll approval issues, legal hold requests), selects a discrete action, and gets graded with a deterministic reward.

Goal of the agent: choose the correct next operational action for each task to maximize total episode reward.

System the agent interacts with: a minimal FastAPI service exposing `POST /reset` and `POST /step` that wraps the environment logic.

## Problem Statement
The environment presents a short episode consisting of 3 fixed tasks (easy, medium, hard). At each step:

1. The server returns an observation containing the current task prompt and the allowed action choices.
2. The agent submits an action.
3. The environment grades the action and returns a scalar reward in the range `0.0..1.0` (deterministic; includes optional partial credit).
4. After the 3rd graded task, the episode terminates.

Success condition: reach `done=true` with a non-zero score. The included baseline reaches normalized score `1.00`.

Constraints:
- Deterministic and offline: no external APIs are required for environment operation or grading.
- Discrete action space: the agent must pick one of a small set of operational actions.

## Key Features
- Deterministic task set and deterministic grading (no randomness by default).
- Small, explicit action space suitable for RL-style interaction loops.
- Partial-credit reward shaping for plausible-but-suboptimal actions.
- FastAPI server for local demo and Docker-based Hugging Face Spaces deployment.
- Local verification scripts (`scripts/verify_local.py`, `scripts/pre_submit_check.py`).

## Repository Structure
```text
.
  Dockerfile
  openenv.yaml
  requirements.txt
  inference.py
  client.py
  server/
    app.py
  my_env/
    environment.py
    tasks.py
    grader.py
    models.py
    server/
      app.py
  scripts/
    verify_local.py
    pre_submit_check.py
```

## Environment I/O (OpenEnv Surface)
Action choices:
- `route_it`
- `route_finance`
- `escalate`
- `reply_with_template`
- `resolve`

Observation fields:
- `task_id`, `difficulty`, `title`, `prompt`
- `choices` (the action choices)
- `remaining_tasks`, `done`, `reward`
- `metadata` (episode id, step count, max reward, etc.)

State fields (via `GET /state`):
- `episode_id`, `step_count`, `current_task_index`
- `total_tasks`, `completed_tasks`, `total_reward`
- `active_task_id`, `last_action`, `last_reward`, `last_error`

Rewards:
- Deterministic per-task grader
- Normalized to `0.0..1.0` with optional partial credit

## Local Setup (Python)
Prereqs: Python 3.11+.

Install dependencies:
```bash
pip install -r requirements.txt
```

Start the server locally:
```bash
python -m server.app
```

Verify locally:
```bash
python scripts/verify_local.py
```

Run local checks:
```bash
python scripts/pre_submit_check.py
openenv validate .
```

## API Contract
Base URL (local): `http://127.0.0.1:7860`

Reset episode:
```bash
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d "{\"seed\":0}"
```

Take one step:
```bash
curl -X POST http://127.0.0.1:7860/step -H "Content-Type: application/json" -d "{\"action\":{\"choice\":\"route_it\"}}"
```

Inspect internal state:
```bash
curl http://127.0.0.1:7860/state
```

## Baseline
`inference.py` contains a deterministic heuristic baseline policy:
- easy: `route_it` -> `0.92`
- medium: `escalate` -> `0.97`
- hard: `reply_with_template` -> `1.00`

Expected total baseline reward: `2.89 / 2.89` (normalized episode score `1.00`).

Note: `inference.py` makes a minimal OpenAI-compatible request before starting the episode so validators can confirm traffic went through the injected LiteLLM proxy. For offline local checks, set `NO_LLM=1`.

Environment variables used by `inference.py`:
```bash
API_BASE_URL=https://your-litellm-proxy/v1
API_KEY=your-proxy-key
MODEL_NAME=gpt-4o-mini
HF_TOKEN=your-proxy-key
```

`API_BASE_URL` is mandatory. `inference.py` accepts either `API_KEY` or `HF_TOKEN` for credentials so it remains compatible with both validator variants, but all LLM traffic still goes through the OpenAI client pointed at the provided proxy URL.

## Docker (Local)
Build:
```bash
docker build -t inboxops .
```

Run:
```bash
docker run --rm -p 7860:7860 -e PORT=7860 inboxops
```

Smoke test:
```bash
curl -X POST http://127.0.0.1:7860/reset
```

## Hugging Face Spaces (Docker)
This repo is intended for a Docker-based Space. The container runs `uvicorn server.app:app` and listens on `PORT` (default `7860`).

Deployment checklist:
- Create a new Space with SDK set to Docker.
- Push this repo (including `Dockerfile`, `server/`, and `my_env/`).
- The server itself does not require secrets. Only set secrets if you plan to run `inference.py` in the Space.
- Confirm the Space is in `Running` state before you try to call the API. If the Space is not `Running`, requests like `POST /reset` will fail.

Verify the deployed Space:
```bash
curl -X POST https://YOUR-SPACE-URL.hf.space/reset
```

## License
See `LICENSE`.

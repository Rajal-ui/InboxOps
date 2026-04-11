---
title: InboxOps
sdk: docker
app_port: 7860
pinned: false
---

# InboxOps

**Deterministic operations-triage benchmark for OpenEnv, FastAPI, Docker, and Hugging Face Spaces**

InboxOps is a compact but production-shaped benchmark environment where an agent must triage internal operations incidents from an enterprise inbox. Instead of generating open-ended text, the agent must choose the most appropriate next action for each operational scenario, receive a deterministic reward, and finish the episode with the highest possible score.

The project is designed to be easy to run, easy to validate, and easy to judge. It combines a realistic operations workflow with a clean API surface, deterministic grading, explainable reward shaping, and a deployment path that works locally and on Hugging Face Spaces.

## Why This Project Matters

Most internal operations work is not glamorous, but it is where organizations lose time, money, and compliance safety every day. Support queues, finance escalations, and legal retention holds all require fast and correct routing decisions. InboxOps turns that real-world decision process into a small, testable benchmark.

This makes the environment relevant for:

- AI agents that need to choose the right operational action under business constraints
- research on deterministic evaluation for workflow automation
- safe benchmarking of triage policies without external dependencies
- demos of OpenEnv-compatible environments that can be deployed and verified quickly

## Core Idea

The agent is placed in a deterministic inbox-operations setting and must decide what to do next for each task. Every task has:

- a business context
- a difficulty level
- a finite action space
- an expected best action
- optional partial-credit actions
- risk metadata such as urgency, compliance risk, and business impact

The benchmark rewards operational judgment, not prompt luck.

## Real-World Use Case

InboxOps models the first layer of operational decision-making inside an enterprise:

- Password-reset requests should be routed to IT
- payroll approval incidents should be escalated because the deadline risk is high
- legal retention holds require a compliant response before mailbox changes are made

These are small examples of a broader class of workflows that appear in IT operations, service desks, finance operations, internal tooling, GRC, and enterprise support. A benchmark like this is useful because it isolates decision quality in a way that is deterministic and auditable.

## What InboxOps Demonstrates

- Deterministic offline benchmark behavior
- enterprise-style inbox triage instead of generic toy tasks
- explicit reward shaping with partial credit for plausible but suboptimal decisions
- explainable action grading
- structured metadata for risk-aware reasoning
- OpenEnv-compatible FastAPI service
- Docker-ready deployment
- Hugging Face Spaces compatibility
- local validation scripts for reproducible testing
- a baseline `inference.py` that satisfies validator expectations while remaining robust to probe failures

## Feature Overview

### 1. Deterministic Task Environment

InboxOps ships with three curated tasks across increasing difficulty:

- `task_easy`: Password Reset Routing
- `task_medium`: Payroll Approval Incident
- `task_hard`: Mailbox Retention Hold

Each task is deterministic and replayable, which makes the environment suitable for benchmarking and comparison across policies.

### 2. Discrete, Auditable Action Space

The environment uses a small operational action set:

- `route_it`
- `route_finance`
- `escalate`
- `reply_with_template`
- `resolve`

This keeps evaluation crisp and makes failure modes easy to inspect.

### 3. Reward Shaping with Partial Credit

Task grading outputs are deterministic and normalized to the open interval `(0, 1)`.

- exact best action receives the task's full reward
- partially reasonable actions can receive partial credit
- invalid or clearly incorrect actions receive a minimal floor reward of `0.01`

This allows the benchmark to distinguish between:

- correct judgment
- plausible but incomplete judgment
- wrong judgment

### 4. Rich Operational Metadata

Each observation carries structured metadata that a serious agent can use:

- urgency
- compliance risk
- business impact
- max reward
- episode id
- step count
- semantic tags

This makes the environment more realistic than a plain classification exercise.

### 5. Counterfactual Action Analysis

The server exposes `GET /analyze/current`, which scores every valid action for the active task. This is useful for:

- debugging policies
- visualizing task difficulty
- explaining reward outcomes
- comparing alternative action choices

### 6. Introspectable API Surface

The environment provides more than just reset and step:

- `GET /`
- `GET /health`
- `GET /metadata`
- `GET /schema`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /analyze/current`
- `POST /mcp`

This makes the benchmark easier to inspect, demo, and integrate.

### 7. OpenAI-Compatible Validator Handshake

`inference.py` performs a minimal OpenAI-compatible request before starting the episode so hosted validators can confirm that the run went through the injected proxy layer. The probe is now best-effort:

- local offline runs can skip it with `NO_LLM=1`
- if the probe fails, inference still completes the benchmark episode cleanly
- warnings are emitted to `stderr` instead of crashing the run

### 8. Deployment-Ready Packaging

The project is ready for:

- local Python execution
- Docker deployment
- Hugging Face Spaces deployment
- OpenEnv validation

## Benchmark Design

### Objective

Maximize total episode reward by selecting the best operational action for each task.

### Episode Structure

- fixed-length episode with 3 tasks
- deterministic order
- one decision per task
- terminal observation after all tasks are graded

### Success Condition

Reach `done=true` and earn a non-zero normalized score.

### Baseline Score

The included deterministic baseline in [inference.py](D:\InboxOps\inference.py) achieves:

- total reward: `2.85 / 2.85`
- normalized score: `1.00`

Optimal mapping:

- easy -> `route_it`
- medium -> `escalate`
- hard -> `reply_with_template`

## Architecture

```text
Agent / Policy
  |
  v
inference.py
  |
  v
FastAPI server (see `server/`)
  |
  v
InboxOpsEnvironment
  |
  v
Task metadata + deterministic grader
```

### Main Components

- [inference.py](D:\InboxOps\inference.py): baseline episode runner and validator-compatible inference entrypoint
- [client.py](D:\InboxOps\client.py): OpenAI client builder for proxy-backed validation
- [server/app.py](D:\InboxOps\server\app.py): deployment entrypoint for the FastAPI app
- [my_env/environment.py](D:\InboxOps\my_env\environment.py): deterministic environment state machine
- [my_env/grader.py](D:\InboxOps\my_env\grader.py): exact-match, partial-credit, and invalid-action grading
- [my_env/tasks.py](D:\InboxOps\my_env\tasks.py): benchmark task catalog and valid actions
- [my_env/models.py](D:\InboxOps\my_env\models.py): pydantic models for observations, rewards, state, and analysis
- [my_env/server/app.py](D:\InboxOps\my_env\server\app.py): full API implementation
- [scripts/pre_submit_check.py](D:\InboxOps\scripts\pre_submit_check.py): local pre-submission validation
- [scripts/smoke_server.py](D:\InboxOps\scripts\smoke_server.py): end-to-end server smoke test
- [scripts/verify_local.py](D:\InboxOps\scripts\verify_local.py): local API interaction check against a running server
- [scripts/benchmark_policies.py](D:\InboxOps\scripts\benchmark_policies.py): policy comparison benchmark

## API Contract

Base URL:

- local: `http://127.0.0.1:7860`
- deployed Space: `https://YOUR-SPACE-URL.hf.space`

### Reset

```bash
curl -X POST http://127.0.0.1:7860/reset \
  -H "Content-Type: application/json" \
  -d "{\"seed\":0}"
```

### Step

```bash
curl -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d "{\"action\":{\"choice\":\"route_it\"}}"
```

### Inspect State

```bash
curl http://127.0.0.1:7860/state
```

### Inspect Task Catalog

```bash
curl http://127.0.0.1:7860/tasks
```

### Analyze All Actions for Current Task

```bash
curl http://127.0.0.1:7860/analyze/current
```

## Observation and State Design

### Observation Fields

- `task_id`
- `difficulty`
- `title`
- `prompt`
- `choices`
- `remaining_tasks`
- `done`
- `reward`
- `metadata`

### Metadata Fields

- `episode_id`
- `step_count`
- `max_reward`
- `urgency`
- `compliance_risk`
- `business_impact`
- `tags`

### State Fields

- `episode_id`
- `step_count`
- `current_task_index`
- `total_tasks`
- `completed_tasks`
- `total_reward`
- `active_task_id`
- `last_action`
- `last_reward`
- `last_error`

## Policy Benchmark Results

The repository includes multiple reference policies in [scripts/benchmark_policies.py](D:\InboxOps\scripts\benchmark_policies.py):

- `optimal`: `1.00`
- `conservative`: `0.53`
- `finance_bias`: `0.51`
- `resolve_bias`: `0.20`

This gives reviewers an immediate sense of benchmark separability and policy quality.

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python -m server.app
```

### 3. Run the Baseline Inference Locally

For offline local checks:

```bash
NO_LLM=1 API_BASE_URL=https://example.invalid/v1 API_KEY=dummy python inference.py
```

On PowerShell:

```powershell
$env:NO_LLM="1"
$env:API_BASE_URL="https://example.invalid/v1"
$env:API_KEY="dummy"
python inference.py
```

### 4. Validate the Project

```bash
python scripts/pre_submit_check.py
python scripts/smoke_server.py
python scripts/benchmark_policies.py
openenv validate .
```

Note: [scripts/verify_local.py](D:\InboxOps\scripts\verify_local.py) assumes a server is already running on `127.0.0.1:7860`.

## Environment Variables

`inference.py` uses:

```bash
API_BASE_URL=https://your-litellm-proxy/v1
API_KEY=your-proxy-key
MODEL_NAME=gpt-4o-mini
NO_LLM=1
```

Notes:

- `API_BASE_URL` is required when the validator expects OpenAI-compatible traffic
- `API_KEY` must be the injected validator key; do not fall back to another token
- `NO_LLM=1` is intended for offline local runs

## Docker

### Build

```bash
docker build -t inboxops .
```

### Run

```bash
docker run --rm -p 7860:7860 -e PORT=7860 inboxops
```

### Smoke Test

```bash
curl -X POST http://127.0.0.1:7860/reset
```

## Hugging Face Spaces

This repository is configured for a Docker-based Space.

- SDK: Docker
- app port: `7860`
- app entrypoint: `uvicorn server.app:app`

Deployment checklist:

1. Create a Hugging Face Space with Docker SDK.
2. Push this repository, including `Dockerfile`, `openenv.yaml`, `server/`, and `my_env/`.
3. Wait until the Space reaches `Running`.
4. Verify the deployment with a `POST /reset` request.

Example:

```bash
curl -X POST https://YOUR-SPACE-URL.hf.space/reset
```

## License

Released under the MIT License. See [LICENSE](D:\InboxOps\LICENSE).

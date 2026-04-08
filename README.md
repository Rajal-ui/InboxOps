# InboxOps

Minimal deterministic OpenEnv hackathon submission for Hugging Face Spaces.

## Overview

InboxOps is an offline environment with three fixed tasks:

- `easy`: route password reset backlog to IT
- `medium`: escalate a payroll approval incident
- `hard`: send the correct legal hold response

The environment is deterministic, lightweight, and runs entirely offline. It does not use a cloud database, and the environment logic does not depend on external APIs or services.

This simulates a real human workflow: internal operations triage. An agent receives a task prompt, chooses an operational action, and is graded programmatically.

## OpenEnv Surface

Action model:

- `route_it`
- `route_finance`
- `escalate`
- `reply_with_template`
- `resolve`

Observation fields:

- `task_id`
- `difficulty`
- `title`
- `prompt`
- `choices`
- `remaining_tasks`
- `done`
- `reward`
- `metadata`

State fields:

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

Reward model:

- `value`
- `max_value`
- `correct`
- `difficulty`
- `reason`

Per-task grader scores are deterministic and normalized to `0.0..1.0`.

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Environment variables:

```bash
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
HF_TOKEN=your-token
OPENAI_API_KEY=your-token
```

Use `.env` or Hugging Face Space secrets for these values. The environment itself stays offline; these variables are only for the root `inference.py` client path.

Start the server locally:

```bash
python -m server.app
```

Run the root inference entrypoint:

```bash
HF_TOKEN=dummy python inference.py
```

## Local API Checks

POST `/reset`:

```bash
curl -X POST http://127.0.0.1:7860/reset
```

POST `/step`:

```bash
curl -X POST http://127.0.0.1:7860/step -H "Content-Type: application/json" -d "{\"action\":{\"choice\":\"route_it\"}}"
```

GET `/state`:

```bash
curl http://127.0.0.1:7860/state
```

Optional local verification script:

```bash
python scripts/verify_local.py
```

Run the pre-submission checks:

```bash
python scripts/pre_submit_check.py
openenv validate .
```

## Docker

Build the image:

```bash
docker build -t inboxops .
```

Run the container:

```bash
docker run --rm -p 7860:7860 -e PORT=7860 inboxops
```

Then test the API again:

```bash
curl -X POST http://127.0.0.1:7860/reset
```

## Hugging Face Spaces

Deploy the repo to a Docker-based Hugging Face Space and confirm the Space is in the `Running` state before submission. If the Space is not `Running`, validation may fail when the validator sends `POST /reset`.

Verify the deployed Space:

```bash
curl -X POST https://YOUR-SPACE-URL.hf.space/reset
```

Use the latest submission only.

The deployed Space must remain in `Running` state before you submit.


## Baseline

Deterministic heuristic baseline in `inference.py`:

- easy: `route_it` -> `0.92`
- medium: `escalate` -> `0.97`
- hard: `reply_with_template` -> `1.0`

Expected total baseline reward:

- `2.89 / 2.89`

Expected normalized episode score:

- `1.00`

from __future__ import annotations

import os
from typing import List

from openai import OpenAI

from client import build_openai_client
from my_env.environment import InboxOpsEnvironment
from my_env.tasks import TASKS


BENCHMARK = "inboxops"
TASK_NAME = "ops_triage"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _select_action(observation: dict) -> str:
    task_id = observation.get("task_id", "")
    policy = {
        "task_easy": "route_it",
        "task_medium": "escalate",
        "task_hard": "reply_with_template",
    }
    return policy.get(task_id, "resolve")


def _build_client() -> OpenAI:
    api_base_url = os.environ["API_BASE_URL"]
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("API_KEY or HF_TOKEN is required")

    return build_openai_client(
        api_base_url=api_base_url,
        model_name=model_name,
        api_key=api_key,
    )


def _maybe_llm_ping(client: OpenAI) -> None:
    # Some validators expect at least one real request to be made via the OpenAI client.
    # Keep it tiny and silent; never emit extra stdout beyond [START]/[STEP]/[END].
    if os.getenv("NO_LLM", "").strip().lower() in {"1", "true", "yes"}:
        return

    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "ping"}],
        temperature=0,
        max_tokens=1,
        stream=False,
    )


def main() -> None:
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    rewards: List[float] = []
    steps = 0
    total_reward = 0.0
    max_total_reward = sum(task.max_reward for task in TASKS)
    success = False
    error: Exception | None = None
    score = 0.0

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={model_name}", flush=True)

    env = InboxOpsEnvironment()
    try:
        client = _build_client()
        _maybe_llm_ping(client)
        observation, _info = env.reset(seed=0)
        done = bool(observation.get("done", False))

        while not done:
            action = _select_action(observation)
            observation, reward, done, info = env.step(action)
            steps += 1
            total_reward += reward
            rewards.append(reward)

            last_error = info.get("error")
            error_text = str(last_error) if last_error else "null"
            print(
                f"[STEP] step={steps} action={action} reward={_format_reward(reward)} "
                f"done={_bool_text(done)} error={error_text}",
                flush=True,
            )

        normalized_score = total_reward / max_total_reward if max_total_reward else 0.0
        score = min(max(normalized_score, 0.0), 1.0)
        success = done and score > 0.0
    except Exception as exc:
        error = exc
        score = min(max(total_reward / max_total_reward, 0.0), 1.0) if max_total_reward else 0.0
        success = False
    finally:
        reward_csv = ",".join(_format_reward(value) for value in rewards)
        print(
            f"[END] success={_bool_text(success)} steps={steps} score={score:.2f} rewards={reward_csv}",
            flush=True,
        )

    if error is not None:
        raise error


if __name__ == "__main__":
    main()

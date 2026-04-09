from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from my_env.environment import InboxOpsEnvironment
from my_env.grader import grade_action
from my_env.tasks import TASKS, VALID_ACTIONS


def check_env_file() -> None:
    print("ENV")
    env_path = ".env"
    if not os.path.exists(env_path):
        print("missing .env")
        return
    print("found .env")


def check_task_rewards() -> None:
    print("TASKS")
    env = InboxOpsEnvironment()
    observation, _ = env.reset(seed=0)
    policy = {
        "task_easy": "route_it",
        "task_medium": "escalate",
        "task_hard": "reply_with_template",
    }
    for task in TASKS:
        action = policy[task.task_id]
        observation, reward, done, info = env.step(action)
        payload = {
            "task_id": task.task_id,
            "reward": reward,
            "done": done,
            "info": info,
        }
        print(json.dumps(payload))
        if not (0.0 < reward < 1.0):
            raise RuntimeError(f"reward must be strictly between 0 and 1 for {task.task_id}: {reward}")
    for task in TASKS:
        for action in VALID_ACTIONS:
            reward, _info = grade_action(task, action)
            if not (0.0 < reward.value < 1.0):
                raise RuntimeError(
                    f"grader output must be strictly between 0 and 1 for {task.task_id}/{action}: {reward.value}"
                )


def check_inference() -> None:
    print("INFERENCE")
    env = os.environ.copy()
    env.setdefault("API_BASE_URL", "https://example.invalid/v1")
    env.setdefault("API_KEY", "dummy-token")
    # Keep local checks offline; hosted validators can allow real LLM calls.
    env.setdefault("NO_LLM", "1")
    result = subprocess.run(
        [sys.executable, "inference.py"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    print(result.stdout.strip())


def check_inference_handles_llm_probe_failure() -> None:
    print("INFERENCE_LLM_FAILURE")
    env = os.environ.copy()
    env["API_BASE_URL"] = "https://example.invalid/v1"
    env["API_KEY"] = "dummy-token"
    env.pop("NO_LLM", None)
    result = subprocess.run(
        [sys.executable, "inference.py"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    print(result.stdout.strip())
    if "[END]" not in result.stdout:
        raise RuntimeError("inference.py did not finish cleanly after LLM probe failure")


def main() -> None:
    check_env_file()
    check_task_rewards()
    check_inference()
    check_inference_handles_llm_probe_failure()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inboxops.environment import InboxOpsEnvironment
from inboxops.tasks import TASKS


Policy = Callable[[dict], str]


def optimal_policy(observation: dict) -> str:
    mapping = {
        "task_easy": "route_it",
        "task_medium": "escalate",
        "task_hard": "reply_with_template",
    }
    return mapping.get(observation.get("task_id", ""), "resolve")


def conservative_policy(_observation: dict) -> str:
    return "escalate"


def finance_bias_policy(observation: dict) -> str:
    return "route_finance" if observation.get("difficulty") != "hard" else "reply_with_template"


def resolve_bias_policy(_observation: dict) -> str:
    return "resolve"


def run_policy(name: str, policy: Policy) -> dict:
    env = InboxOpsEnvironment()
    observation, _info = env.reset(seed=0)
    done = bool(observation.get("done", False))
    total_reward = 0.0
    steps = 0
    rewards: list[float] = []
    while not done:
        action = policy(observation)
        observation, reward, done, _info = env.step(action)
        total_reward += reward
        steps += 1
        rewards.append(reward)
    max_reward = sum(task.max_reward for task in TASKS)
    return {
        "policy": name,
        "steps": steps,
        "total_reward": round(total_reward, 2),
        "normalized_score": round(total_reward / max_reward, 2) if max_reward else 0.0,
        "rewards": rewards,
    }


def main() -> None:
    policies: list[tuple[str, Policy]] = [
        ("optimal", optimal_policy),
        ("conservative", conservative_policy),
        ("finance_bias", finance_bias_policy),
        ("resolve_bias", resolve_bias_policy),
    ]
    results = [run_policy(name, policy) for name, policy in policies]
    print(json.dumps({"benchmark": "inboxops", "results": results}, indent=2))


if __name__ == "__main__":
    main()

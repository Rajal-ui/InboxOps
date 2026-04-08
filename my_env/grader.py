from __future__ import annotations

from my_env.models import InboxOpsReward, TaskMetadata
from my_env.tasks import is_valid_action


def grade_action(task: TaskMetadata, action_value: str) -> tuple[InboxOpsReward, dict[str, object]]:
    normalized = action_value.strip().lower()

    if not is_valid_action(normalized):
        reward = InboxOpsReward(
            value=0.0,
            max_value=task.max_reward,
            correct=False,
            difficulty=task.difficulty,
            reason="invalid_action",
        )
        return reward, {
            "task_id": task.task_id,
            "expected_action": task.expected_action,
            "received_action": normalized,
            "error": "invalid_action",
        }

    if normalized == task.expected_action:
        reward = InboxOpsReward(
            value=task.max_reward,
            max_value=task.max_reward,
            correct=True,
            difficulty=task.difficulty,
            reason="exact_match",
        )
        return reward, {
            "task_id": task.task_id,
            "expected_action": task.expected_action,
            "received_action": normalized,
            "grade": "correct",
        }

    partial_value = task.partial_credit.get(normalized)
    if partial_value is not None:
        reward = InboxOpsReward(
            value=partial_value,
            max_value=task.max_reward,
            correct=False,
            difficulty=task.difficulty,
            reason="partial_credit",
        )
        return reward, {
            "task_id": task.task_id,
            "expected_action": task.expected_action,
            "received_action": normalized,
            "grade": "partial",
        }

    reward = InboxOpsReward(
        value=0.0,
        max_value=task.max_reward,
        correct=False,
        difficulty=task.difficulty,
        reason="incorrect_action",
    )
    return reward, {
        "task_id": task.task_id,
        "expected_action": task.expected_action,
        "received_action": normalized,
        "grade": "incorrect",
    }

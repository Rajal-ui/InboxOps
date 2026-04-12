from __future__ import annotations

from typing import Any

from inboxops.grader import MIN_REWARD_FLOOR, grade_action
from inboxops.tasks import TASKS


_TASK_BY_ID = {task.task_id: task for task in TASKS}


def _extract_task_id(*args: Any, **kwargs: Any) -> str | None:
    for value in args:
        if isinstance(value, str) and value in _TASK_BY_ID:
            return value
        if isinstance(value, dict):
            task_id = value.get("task_id") or value.get("id")
            if isinstance(task_id, str) and task_id in _TASK_BY_ID:
                return task_id
    task_id = kwargs.get("task_id") or kwargs.get("id")
    if isinstance(task_id, str) and task_id in _TASK_BY_ID:
        return task_id
    task = kwargs.get("task")
    if isinstance(task, dict):
        nested_task_id = task.get("task_id") or task.get("id")
        if isinstance(nested_task_id, str) and nested_task_id in _TASK_BY_ID:
            return nested_task_id
    return None


def _extract_action(*args: Any, **kwargs: Any) -> str | None:
    direct_action = kwargs.get("action")
    if isinstance(direct_action, str):
        return direct_action
    if isinstance(direct_action, dict):
        choice = direct_action.get("choice")
        if isinstance(choice, str):
            return choice

    trajectory = kwargs.get("trajectory")
    if isinstance(trajectory, list):
        for item in reversed(trajectory):
            if not isinstance(item, dict):
                continue
            action = item.get("action")
            if isinstance(action, str):
                return action
            if isinstance(action, dict):
                choice = action.get("choice")
                if isinstance(choice, str):
                    return choice

    for value in args:
        if isinstance(value, dict):
            action = value.get("action")
            if isinstance(action, str):
                return action
            if isinstance(action, dict):
                choice = action.get("choice")
                if isinstance(choice, str):
                    return choice
    return None


def grade_task(*args: Any, **kwargs: Any) -> float:
    """
    Validator-friendly task grader entrypoint.

    The hosted validator may call this with varying argument shapes, so this
    function tolerates flexible inputs and always returns a score in (0, 1).
    """

    task_id = _extract_task_id(*args, **kwargs)
    if task_id is None:
        return MIN_REWARD_FLOOR

    task = _TASK_BY_ID[task_id]
    action = _extract_action(*args, **kwargs)
    if action is None:
        return task.max_reward

    reward, _info = grade_action(task, action)
    return reward.value


def grade_task_easy(*args: Any, **kwargs: Any) -> float:
    kwargs["task_id"] = "task_easy"
    return grade_task(*args, **kwargs)


def grade_task_medium(*args: Any, **kwargs: Any) -> float:
    kwargs["task_id"] = "task_medium"
    return grade_task(*args, **kwargs)


def grade_task_hard(*args: Any, **kwargs: Any) -> float:
    kwargs["task_id"] = "task_hard"
    return grade_task(*args, **kwargs)

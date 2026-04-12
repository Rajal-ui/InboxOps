from __future__ import annotations

from typing import Any

from server.grader import MIN_REWARD_FLOOR, grade_action
from server.tasks import TASKS


_TASK_BY_ID = {task.task_id: task for task in TASKS}


def _clamp(value: float) -> float:
    # Hackathon requirement: 0.0 and 1.0 are invalid.
    return max(0.01, min(0.99, value))


def _extract_task_id(instance: Any, *args, **kwargs) -> str | None:
    # Try keywords first
    tid = kwargs.get("task_id") or kwargs.get("id")
    if tid:
        return tid

    # Try raw dictionary as first arg
    if args and isinstance(args[0], dict):
        return args[0].get("task_id") or args[0].get("id")

    # Try looking for a .task_id or .id attribute on the first arg (State object)
    if args and hasattr(args[0], "task_id"):
        return getattr(args[0], "task_id")
    if args and hasattr(args[0], "active_task_id"):
        return getattr(args[0], "active_task_id")

    return None


class EasyGrader:
    def grade(self, env: Any, *args, **kwargs) -> float:
        task = _TASK_BY_ID["task_easy"]
        # Extract action choice from kwargs or args
        action_value = kwargs.get("action")
        if action_value is None and args:
            # Action is often passed as a second arg if the first is state
            if len(args) > 1:
                action_value = args[1]
            elif isinstance(args[0], (str, dict)):
                action_value = args[0]

        reward, _info = grade_action(task, action_value)
        return _clamp(reward.value)


class MediumGrader:
    def grade(self, env: Any, *args, **kwargs) -> float:
        task = _TASK_BY_ID["task_medium"]
        action_value = kwargs.get("action")
        if action_value is None and args:
            if len(args) > 1:
                action_value = args[1]
            elif isinstance(args[0], (str, dict)):
                action_value = args[0]

        reward, _info = grade_action(task, action_value)
        return _clamp(reward.value)


class HardGrader:
    def grade(self, env: Any, *args, **kwargs) -> float:
        task = _TASK_BY_ID["task_hard"]
        action_value = kwargs.get("action")
        if action_value is None and args:
            if len(args) > 1:
                action_value = args[1]
            elif isinstance(args[0], (str, dict)):
                action_value = args[0]

        reward, _info = grade_action(task, action_value)
        return _clamp(reward.value)


class MultiGrader:
    """Dispatches to the correct grader based on task discovery."""

    def __init__(self):
        self.easy = EasyGrader()
        self.medium = MediumGrader()
        self.hard = HardGrader()

    def grade(self, env: Any, *args, **kwargs) -> float:
        task_id = _extract_task_id(None, *args, **kwargs)
        if task_id == "task_easy":
            return self.easy.grade(env, *args, **kwargs)
        elif task_id == "task_medium":
            return self.medium.grade(env, *args, **kwargs)
        elif task_id == "task_hard":
            return self.hard.grade(env, *args, **kwargs)
        return _clamp(MIN_REWARD_FLOOR)


# Also expose as functions for potential direct calls
def grade_task_easy(env, *args, **kwargs) -> float:
    return EasyGrader().grade(env, *args, **kwargs)


def grade_task_medium(env, *args, **kwargs) -> float:
    return MediumGrader().grade(env, *args, **kwargs)


def grade_task_hard(env, *args, **kwargs) -> float:
    return HardGrader().grade(env, *args, **kwargs)


def grade_task(env, *args, **kwargs) -> float:
    return MultiGrader().grade(env, *args, **kwargs)

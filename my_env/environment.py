from __future__ import annotations

from typing import Any

from my_env.grader import grade_action
from my_env.models import (
    CounterfactualActionScore,
    CounterfactualAnalysis,
    InboxOpsAction,
    InboxOpsObservation,
    InboxOpsState,
    InboxOpsStepResult,
    TaskMetadata,
)
from my_env.tasks import TASKS, VALID_ACTIONS


class InboxOpsEnvironment:
    """Small deterministic environment with three graded tasks."""

    def __init__(self) -> None:
        self._tasks: tuple[TaskMetadata, ...] = TASKS
        self._reset_counter = 0
        self._state = self._initial_state(seed=0)
        self._done = False

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._reset_counter += 1
        resolved_seed = 0 if seed is None else seed
        self._state = self._initial_state(seed=resolved_seed)
        self._done = False
        observation = self._build_observation(reward_value=0.0)
        info = {
            "message": "environment_reset",
            "seed": resolved_seed,
            "task_count": len(self._tasks),
        }
        return observation.model_dump(), info

    def step(
        self, action: InboxOpsAction | dict[str, Any] | str | None
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._done:
            observation = self._terminal_observation()
            return observation.model_dump(), 0.0, True, {"error": "episode_already_finished"}

        task = self.current_task()
        if task is None:
            self._done = True
            observation = self._terminal_observation()
            return observation.model_dump(), 0.0, True, {"error": "no_active_task"}

        normalized_action = self._normalize_action(action)
        reward, info = grade_action(task, normalized_action)

        next_index = self._state.current_task_index + 1
        done = next_index >= len(self._tasks)
        error_value = info.get("error")
        last_error = error_value if isinstance(error_value, str) else None

        self._state = InboxOpsState(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count + 1,
            current_task_index=next_index,
            total_tasks=len(self._tasks),
            completed_tasks=self._state.completed_tasks + 1,
            total_reward=round(self._state.total_reward + reward.value, 2),
            active_task_id=None if done else self._tasks[next_index].task_id,
            last_action=normalized_action,
            last_reward=reward.value,
            last_error=last_error,
        )
        self._done = done

        observation = self._terminal_observation() if done else self._build_observation(reward_value=reward.value)
        result = InboxOpsStepResult(
            observation=observation,
            reward=reward,
            done=done,
            info={
                **info,
                "difficulty": task.difficulty,
                "completed_tasks": self._state.completed_tasks,
                "total_reward": self._state.total_reward,
            },
        )
        return (
            result.observation.model_dump(),
            result.reward.value,
            result.done,
            result.info,
        )

    def state(self) -> InboxOpsState:
        return self._state

    def counterfactual_for_current_task(self) -> CounterfactualAnalysis | None:
        task = self.current_task()
        if task is None:
            return None
        scores = []
        for action in VALID_ACTIONS:
            reward, _info = grade_action(task, action)
            scores.append(
                CounterfactualActionScore(
                    action=action,
                    reward=reward.value,
                    correct=reward.correct,
                    reason=reward.reason,
                )
            )
        return CounterfactualAnalysis(
            task_id=task.task_id,
            title=task.title,
            expected_action=task.expected_action,
            scores=scores,
        )

    def current_task(self) -> TaskMetadata | None:
        if self._done or self._state.current_task_index >= len(self._tasks):
            return None
        return self._tasks[self._state.current_task_index]

    def _initial_state(self, seed: int) -> InboxOpsState:
        return InboxOpsState(
            episode_id=f"inboxops-{self._reset_counter + 1:04d}-seed-{seed}",
            step_count=0,
            current_task_index=0,
            total_tasks=len(self._tasks),
            completed_tasks=0,
            total_reward=0.0,
            active_task_id=self._tasks[0].task_id,
            last_action=None,
            last_reward=0.0,
            last_error=None,
        )

    def _normalize_action(self, action: InboxOpsAction | dict[str, Any] | str | None) -> str:
        if isinstance(action, InboxOpsAction):
            return action.choice.strip().lower()
        if isinstance(action, dict):
            raw_choice = action.get("choice", "")
            return str(raw_choice).strip().lower()
        if action is None:
            return ""
        return str(action).strip().lower()

    def _build_observation(self, reward_value: float) -> InboxOpsObservation:
        task = self.current_task()
        if task is None:
            return self._terminal_observation()
        return InboxOpsObservation(
            task_id=task.task_id,
            difficulty=task.difficulty,
            title=task.title,
            prompt=task.prompt,
            choices=list(VALID_ACTIONS),
            remaining_tasks=len(self._tasks) - self._state.current_task_index,
            done=False,
            reward=reward_value,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "max_reward": task.max_reward,
                "urgency": task.urgency,
                "compliance_risk": task.compliance_risk,
                "business_impact": task.business_impact,
                "tags": task.tags,
            },
        )

    def _terminal_observation(self) -> InboxOpsObservation:
        return InboxOpsObservation(
            task_id="terminal",
            difficulty=None,
            title="Episode Complete",
            prompt="All tasks have been graded.",
            choices=[],
            remaining_tasks=0,
            done=True,
            reward=self._state.last_reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "total_reward": self._state.total_reward,
            },
        )

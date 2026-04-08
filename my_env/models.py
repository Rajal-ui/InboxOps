from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Difficulty = Literal["easy", "medium", "hard"]
ActionChoice = Literal[
    "route_it",
    "route_finance",
    "escalate",
    "reply_with_template",
    "resolve",
]


class TaskMetadata(BaseModel):
    task_id: str = Field(..., description="Stable task identifier")
    difficulty: Difficulty = Field(..., description="Task difficulty bucket")
    title: str = Field(..., description="Short task name")
    prompt: str = Field(..., description="Task prompt shown to the agent")
    expected_action: ActionChoice = Field(..., description="Ground-truth action for grading")
    max_reward: float = Field(..., description="Maximum reward available for this task")
    partial_credit: dict[ActionChoice, float] = Field(
        default_factory=dict,
        description="Optional deterministic partial-credit actions scored between 0.0 and 1.0",
    )


class InboxOpsAction(BaseModel):
    choice: str = Field(..., description="Action selected by the agent")
    metadata: dict[str, Any] = Field(default_factory=dict)


class InboxOpsReward(BaseModel):
    value: float = Field(..., description="Reward returned for the last action")
    max_value: float = Field(..., description="Maximum reward available for the task")
    correct: bool = Field(..., description="Whether the selected action was correct")
    difficulty: Difficulty = Field(..., description="Difficulty level for the graded task")
    reason: str = Field(..., description="Short deterministic grading explanation")


class InboxOpsObservation(BaseModel):
    task_id: str = Field(default="", description="Current task identifier")
    difficulty: Difficulty | None = Field(default=None, description="Current task difficulty")
    title: str = Field(default="", description="Current task title")
    prompt: str = Field(default="", description="Task prompt shown to the agent")
    choices: list[ActionChoice] = Field(default_factory=list, description="Available actions")
    remaining_tasks: int = Field(default=0, description="Tasks remaining in the episode")
    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: float | None = Field(default=None, description="Latest reward signal")
    metadata: dict[str, Any] = Field(default_factory=dict)


class InboxOpsStepResult(BaseModel):
    observation: InboxOpsObservation
    reward: InboxOpsReward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class InboxOpsState(BaseModel):
    episode_id: str | None = Field(default=None, description="Episode identifier")
    step_count: int = Field(default=0, description="Number of steps taken so far")
    current_task_index: int = Field(default=0, description="Zero-based active task index")
    total_tasks: int = Field(default=0, description="Total number of tasks in the episode")
    completed_tasks: int = Field(default=0, description="Tasks already graded")
    total_reward: float = Field(default=0.0, description="Accumulated episode reward")
    active_task_id: str | None = Field(default=None, description="Current active task id")
    last_action: str | None = Field(default=None, description="Most recent action value")
    last_reward: float = Field(default=0.0, description="Most recent scalar reward")
    last_error: str | None = Field(default=None, description="Most recent safe error message")

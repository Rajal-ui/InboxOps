from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


Difficulty = Literal["easy", "medium", "hard"]
Urgency = Literal["low", "medium", "high"]
RiskLevel = Literal["low", "medium", "high"]
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
    urgency: Urgency = Field(..., description="Operational urgency of the task")
    compliance_risk: RiskLevel = Field(..., description="Risk of violating policy or legal process")
    business_impact: RiskLevel = Field(..., description="Business impact if the task is mishandled")
    tags: list[str] = Field(default_factory=list, description="Extra descriptors for analytics and demos")
    expected_action: ActionChoice = Field(..., description="Ground-truth action for grading")
    max_reward: float = Field(..., description="Maximum reward available for this task")
    grader: str | None = Field(default=None, description="Optional grader entrypoint for validator task discovery")
    partial_credit: dict[ActionChoice, float] = Field(
        default_factory=dict,
        description="Optional deterministic partial-credit actions scored between 0.0 and 1.0",
    )


class InboxOpsAction(Action):
    choice: str = Field(..., description="Action selected by the agent")


class InboxOpsReward(BaseModel):
    value: float = Field(..., description="Reward returned for the last action")
    max_value: float = Field(..., description="Maximum reward available for the task")
    correct: bool = Field(..., description="Whether the selected action was correct")
    difficulty: Difficulty = Field(..., description="Difficulty level for the graded task")
    reason: str = Field(..., description="Short deterministic grading explanation")


class InboxOpsObservation(Observation):
    task_id: str = Field(default="", description="Current task identifier")
    difficulty: Difficulty | None = Field(default=None, description="Current task difficulty")
    title: str = Field(default="", description="Current task title")
    prompt: str = Field(default="", description="Task prompt shown to the agent")
    choices: list[ActionChoice] = Field(default_factory=list, description="Available actions")
    remaining_tasks: int = Field(default=0, description="Tasks remaining in the episode")


class InboxOpsStepResult(BaseModel):
    observation: InboxOpsObservation
    reward: InboxOpsReward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class InboxOpsState(State):
    current_task_index: int = Field(default=0, description="Zero-based active task index")
    total_tasks: int = Field(default=0, description="Total number of tasks in the episode")
    completed_tasks: int = Field(default=0, description="Tasks already graded")
    total_reward: float = Field(default=0.0, description="Accumulated episode reward")
    active_task_id: str | None = Field(default=None, description="Current active task id")
    last_action: str | None = Field(default=None, description="Most recent action value")
    last_reward: float = Field(default=0.0, description="Most recent scalar reward")
    last_error: str | None = Field(default=None, description="Most recent safe error message")


class CounterfactualActionScore(BaseModel):
    action: ActionChoice
    reward: float
    correct: bool
    reason: str


class CounterfactualAnalysis(BaseModel):
    task_id: str
    title: str
    expected_action: ActionChoice
    scores: list[CounterfactualActionScore]

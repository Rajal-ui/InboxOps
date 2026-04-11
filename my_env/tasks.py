from __future__ import annotations

from typing import TypeGuard

from my_env.models import ActionChoice, TaskMetadata


class InboxOpsTask(TaskMetadata):
    def grade(self, action_value: str) -> float:
        from my_env.grader import grade_action

        reward, _info = grade_action(self, action_value)
        return reward.value


VALID_ACTIONS: tuple[ActionChoice, ...] = (
    "route_it",
    "route_finance",
    "escalate",
    "reply_with_template",
    "resolve",
)


TASKS: tuple[InboxOpsTask, ...] = (
    InboxOpsTask(
        task_id="task_easy",
        difficulty="easy",
        title="Password Reset Routing",
        prompt=(
            "An internal inbox contains employee password reset requests. "
            "Choose the best next action for the queue."
        ),
        urgency="low",
        compliance_risk="low",
        business_impact="medium",
        tags=["identity", "routing", "service-desk"],
        expected_action="route_it",
        max_reward=0.92,
        grader="my_env.task_graders:grade_task_easy",
        partial_credit={
            "resolve": 0.21,
        },
    ),
    InboxOpsTask(
        task_id="task_medium",
        difficulty="medium",
        title="Payroll Approval Incident",
        prompt=(
            "Finance reports that payroll approvals are blocked before a payroll deadline. "
            "Choose the best next action."
        ),
        urgency="high",
        compliance_risk="medium",
        business_impact="high",
        tags=["finance", "deadline", "escalation"],
        expected_action="escalate",
        max_reward=0.97,
        grader="my_env.task_graders:grade_task_medium",
        partial_credit={
            "route_finance": 0.48,
            "resolve": 0.19,
        },
    ),
    InboxOpsTask(
        task_id="task_hard",
        difficulty="hard",
        title="Mailbox Retention Hold",
        prompt=(
            "Legal requires a compliant hold response before any mailbox change is made. "
            "Choose the best next action."
        ),
        urgency="high",
        compliance_risk="high",
        business_impact="high",
        tags=["legal", "retention", "compliance"],
        expected_action="reply_with_template",
        max_reward=0.96,
        grader="my_env.task_graders:grade_task_hard",
        partial_credit={
            "escalate": 0.52,
            "resolve": 0.17,
        },
    ),
)


def is_valid_action(value: str) -> TypeGuard[ActionChoice]:
    return value in VALID_ACTIONS

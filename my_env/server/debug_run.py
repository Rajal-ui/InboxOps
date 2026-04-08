from __future__ import annotations

from pprint import pprint

from my_env.models import InboxOpsAction
from my_env.environment import InboxOpsEnvironment


def main() -> None:
    env = InboxOpsEnvironment()
    observation, info = env.reset(seed=0)
    print("RESET")
    pprint(observation)
    pprint(info)

    for choice in ["route_finance", "escalate", "reply_with_template"]:
        observation, reward, done, step_info = env.step(InboxOpsAction(choice=choice))
        print("STEP")
        pprint(observation)
        print(reward)
        print(done)
        pprint(step_info)
        if done:
            break


if __name__ == "__main__":
    main()

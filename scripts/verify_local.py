from __future__ import annotations

import json
import urllib.request


BASE_URL = "http://127.0.0.1:7860"


def _post(path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    reset_result = _post("/reset")
    print("RESET")
    print(json.dumps(reset_result, indent=2))

    step_result = _post("/step", {"action": {"choice": "route_it"}})
    print("STEP")
    print(json.dumps(step_result, indent=2))

    state_result = _get("/state")
    print("STATE")
    print(json.dumps(state_result, indent=2))


if __name__ == "__main__":
    main()

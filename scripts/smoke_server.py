from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _get_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _post_json(url: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def main() -> int:
    port = _free_port()
    env = os.environ.copy()
    env["PORT"] = str(port)

    proc = subprocess.Popen(
        [sys.executable, "-m", "server.app"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"

        # Wait for server to start.
        deadline = time.time() + 15
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                _ = _get_json(f"{base}/health", timeout=1.0)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.2)
        if last_err is not None:
            raise RuntimeError(f"server did not become healthy: {last_err}")

        health = _get_json(f"{base}/health")
        print("HEALTH", json.dumps(health, sort_keys=True))

        reset = _post_json(f"{base}/reset", {"seed": 0})
        print("RESET", json.dumps(reset, sort_keys=True)[:500])

        step = _post_json(f"{base}/step", {"action": {"choice": "route_it"}})
        print("STEP", json.dumps(step, sort_keys=True)[:500])

        state = _get_json(f"{base}/state")
        print("STATE", json.dumps(state, sort_keys=True)[:500])

        return 0
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"http error: {exc.code} {exc.reason}") from exc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        if proc.stdout is not None:
            out = proc.stdout.read().strip()
            if out:
                print("SERVER_LOGS_BEGIN")
                print(out[-2000:])
                print("SERVER_LOGS_END")


if __name__ == "__main__":
    raise SystemExit(main())


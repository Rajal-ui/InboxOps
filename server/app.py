from __future__ import annotations

from fastapi import FastAPI
from typing import Optional

app = FastAPI()


@app.get("/")
def read_root() -> dict:
    return {"status": "ok", "service": "InboxOps"}


def main(host: str = "0.0.0.0", port: Optional[int] = None) -> None:
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port or 8000)


if __name__ == "__main__":
    main()

from __future__ import annotations

from my_env.server.app import app as app
from my_env.server.app import main as _main


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    _main(host=host, port=port)


if __name__ == "__main__":
    main()

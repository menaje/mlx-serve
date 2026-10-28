"""Run an app-owned server only while its local management agent exists."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os

import uvicorn

from mlx_serve.config import settings


async def _serve(host: str, port: int, parent_pid: int) -> None:
    # Propagate app ownership to isolated workers so they use the same log policy.
    os.environ["MLX_SERVE_APP_OWNED"] = "1"
    config = uvicorn.Config(
        "mlx_serve.server:app", host=host, port=port, access_log=False, proxy_headers=False,
        log_level=settings.log_level.lower(),
    )
    server = uvicorn.Server(config)

    async def monitor_parent() -> None:
        while not server.should_exit:
            if os.getppid() != parent_pid:
                server.should_exit = True
                return
            await asyncio.sleep(1)

    watcher = asyncio.create_task(monitor_parent())
    try:
        await server.serve()
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(_serve(args.host, args.port, args.parent))


if __name__ == "__main__":
    main()

"""Disposable process running the same validated model preparation as the CLI."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import threading


async def run(parent: int, job_id: str) -> int:
    from mlx_serve.core.model_manager import model_manager
    from mlx_serve.core.retrieval_prepare_worker import RETRIEVAL_PROFILES

    params = json.loads(sys.stdin.readline())
    os.environ["MLX_SERVE_PREPARE_JOB_ID"] = job_id
    current = asyncio.current_task()
    assert current is not None

    async def watch_parent() -> None:
        while True:
            await asyncio.sleep(1)
            if os.getppid() != parent:
                current.cancel()
                return

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, current.cancel)
    watcher = asyncio.create_task(watch_parent())
    outcome = 1
    try:
        async for event in model_manager.pull_model(
            hf_repo=params["repo"],
            model_type=RETRIEVAL_PROFILES.get(params["repo"], "decision"),
            model_name=params.get("name"),
            revision=params.get("revision"),
            keep_original=params.get("keep_original", False),
        ):
            print(json.dumps(event, ensure_ascii=False), flush=True)
            if event.get("status") == "success":
                outcome = 0
        return outcome
    except asyncio.CancelledError:
        return 130
    finally:
        watcher.cancel()
        loop.remove_signal_handler(signal.SIGTERM)


def _enforce_parent_lifetime(parent: int, finished: threading.Event) -> None:
    """Bound orphan cleanup even when a download thread cannot be cancelled."""
    while not finished.wait(1):
        if os.getppid() == parent:
            continue
        pid = os.getpid()
        # Only signal the group when the job runner owns it. Otherwise leave
        # unrelated processes alone and stop the runner itself.
        target = os.killpg if os.getpgrp() == pid else os.kill
        with contextlib.suppress(ProcessLookupError, PermissionError):
            target(pid, signal.SIGTERM)
        if not finished.wait(10):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                target(pid, signal.SIGKILL)
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--parent", type=int, required=True)
    args = parser.parse_args()
    finished = threading.Event()
    threading.Thread(
        target=_enforce_parent_lifetime, args=(args.parent, finished), daemon=True,
    ).start()
    try:
        raise SystemExit(asyncio.run(run(args.parent, args.job_id)))
    except BrokenPipeError:
        raise SystemExit(1) from None
    finally:
        finished.set()


if __name__ == "__main__":
    main()

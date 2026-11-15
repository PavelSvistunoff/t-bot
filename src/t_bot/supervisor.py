import asyncio
import fcntl
import os
import signal
import sys
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from t_bot.config import Settings
from t_bot.errors import BotError

RESTART_EXIT_CODE = 75


@contextmanager
def process_lock(path: Path) -> Iterator[None]:
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BotError("Інший екземпляр уже працює.") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


async def supervise(settings: Settings) -> int:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    failures: deque[float] = deque()
    with process_lock(settings.data_dir / "supervisor.lock"):
        while not stop.is_set():
            environment = os.environ.copy()
            environment["T_BOT_SUPERVISED"] = "1"
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "t_bot",
                "worker",
                env=environment,
            )
            started = time.monotonic()
            wait_child = asyncio.create_task(child.wait())
            wait_stop = asyncio.create_task(stop.wait())
            try:
                await asyncio.wait([wait_child, wait_stop], return_when=asyncio.FIRST_COMPLETED)
                if stop.is_set() and child.returncode is None:
                    child.terminate()
                    try:
                        async with asyncio.timeout(settings.shutdown_timeout_seconds + 20):
                            await wait_child
                    except TimeoutError:
                        child.kill()
                        await wait_child
                    return 0
                code = await wait_child
                if code == 0:
                    return 0
                now = time.monotonic()
                if now - started < settings.restart_cooldown_seconds:
                    failures.append(now)
                while failures and now - failures[0] > 300:
                    failures.popleft()
                if len(failures) >= 5:
                    from t_bot.store import Store

                    store = Store(settings.data_dir / "state.sqlite3")
                    pending = store.get("restart_pending")
                    if pending:
                        store.notify(pending["chat_id"], "Перезапуск не вдався: перевірте журнали.")
                    store.set("worker_state", {"ready": False, "reason": "restart_limit"})
                    store.close()
                    return 1
                # Backoff also bounds repeated requested restarts after early startup failures.
                try:
                    async with asyncio.timeout(min(2 ** len(failures), 30)):
                        await stop.wait()
                except TimeoutError:
                    continue
            finally:
                wait_stop.cancel()
                await asyncio.gather(wait_stop, return_exceptions=True)

    return 0


def stop_supervisor(settings: Settings) -> None:
    path = settings.data_dir / "supervisor.lock"
    if not path.exists():
        raise BotError("Supervisor не запущено.")
    with path.open("r+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid = int(handle.read())
            os.kill(pid, signal.SIGTERM)
            return
        raise BotError("Supervisor не запущено.")

import asyncio
import logging
from collections import defaultdict

from t_bot.config import Settings
from t_bot.errors import BotError
from t_bot.runtime import AgentRuntime
from t_bot.store import Store

logger = logging.getLogger(__name__)


class Jobs:
    def __init__(self, settings: Settings, store: Store, runtime: AgentRuntime):
        self.settings = settings
        self.store = store
        self.runtime = runtime
        self.queue: asyncio.Queue[int] = asyncio.Queue(maxsize=settings.queue_size)
        self.locks: dict[tuple[int, int], asyncio.Lock] = defaultdict(asyncio.Lock)
        self.running: dict[int, asyncio.Task] = {}
        self.workers: list[asyncio.Task] = []
        self.accepting = True

    def start(self) -> None:
        self.store.recover()
        self.workers = [
            asyncio.create_task(self.worker(), name=f"worker-{index}")
            for index in range(self.settings.max_parallel_tasks)
        ]

    def submit(self, update_id: int, user_id: int, chat_id: int, text: str) -> bool:
        self.settings.authorize(user_id)
        if not self.accepting:
            raise BotError("Бот завершує роботу. Надішліть запит після перезапуску.")
        if len(text) > self.settings.max_input_chars:
            raise BotError("Повідомлення завелике.")
        if self.queue.full():
            raise BotError("Черга заповнена. Спробуйте пізніше.")
        if not self.store.enqueue(update_id, user_id, chat_id, text):
            return False
        self.queue.put_nowait(update_id)
        self.store.notify(chat_id, f"Задачу {update_id} додано до черги. /cancel {update_id}")
        return True

    def cancel(self, job_id: int, user_id: int, chat_id: int) -> str:
        self.settings.authorize(user_id)
        try:
            job = self.store.job(job_id)
        except KeyError as exc:
            raise BotError("Задачу не знайдено.") from exc
        if job["user_id"] != user_id or job["chat_id"] != chat_id:
            raise BotError("Доступ до задачі заборонено.")
        if job["status"] not in {"queued", "running"}:
            return "Задача вже завершена."
        task = self.running.get(job_id)
        if task:
            task.cancel()
        self.store.finish(job_id, "cancelled", f"Задачу {job_id} скасовано.")
        return "Скасування запитано."

    async def worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                job = self.store.job(job_id)
                async with self.locks[(job["user_id"], job["chat_id"])]:
                    if self.store.job(job_id)["status"] != "queued":
                        continue
                    with self.store.transaction() as db:
                        db.execute("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
                    self.store.notify(job["chat_id"], f"Виконую задачу {job_id}…")
                    task = asyncio.create_task(self.execute(job))
                    self.running[job_id] = task
                    try:
                        await task
                    except asyncio.CancelledError:
                        worker = asyncio.current_task()
                        if worker is not None and worker.cancelling():
                            raise
                    finally:
                        self.running.pop(job_id, None)
            finally:
                self.queue.task_done()

    async def execute(self, job: dict) -> None:
        job_id = job["id"]
        extra = {"job_id": job_id, "user_id": job["user_id"], "chat_id": job["chat_id"]}
        logger.info("job.started", extra=extra)
        try:
            result = await self.runtime.run(job["user_id"], job["chat_id"], job_id, job["text"])
            self.store.finish(job_id, "done", result)
            logger.info("job.done", extra=extra)
        except asyncio.CancelledError:
            if self.store.job(job_id)["status"] != "cancelled":
                self.store.finish(job_id, "interrupted", f"Задачу {job_id} перервано.")
            raise
        except (BotError, TimeoutError) as exc:
            message = str(exc) if isinstance(exc, BotError) else "Перевищено час виконання задачі."
            self.store.finish(job_id, "failed", message)
            logger.warning("job.failed", extra={**extra, "error_type": type(exc).__name__})
        except Exception as exc:
            # External providers may include credentials/prompts in exception strings.
            logger.error("job.failed", extra={**extra, "error_type": type(exc).__name__})
            self.store.finish(
                job_id,
                "failed",
                "Помилка моделі або інструмента. Перевірте Ollama, пам’ять та /model; "
                "дії з побічними ефектами автоматично не повторюються.",
            )

    async def shutdown(self) -> None:
        self.accepting = False
        try:
            async with asyncio.timeout(self.settings.shutdown_timeout_seconds):
                await self.queue.join()
        except TimeoutError:
            pass  # Outstanding jobs below are explicitly interrupted, never silently retried.
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.store.recover()

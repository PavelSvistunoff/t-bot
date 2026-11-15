import asyncio

import pytest

from t_bot.errors import BotError
from t_bot.jobs import Jobs


class RecordingRuntime:
    def __init__(self):
        self.started = []
        self.release = asyncio.Event()

    async def run(self, user_id, chat_id, job_id, text):
        self.started.append(job_id)
        await self.release.wait()
        return text.upper()


async def test_order_dedup_and_cancel(settings, store):
    runtime = RecordingRuntime()
    jobs = Jobs(settings, store, runtime)
    jobs.start()
    try:
        assert jobs.submit(100, 1, 5, "one")
        assert not jobs.submit(100, 1, 5, "duplicate")
        jobs.submit(101, 1, 5, "two")
        for _ in range(100):
            if runtime.started:
                break
            await asyncio.sleep(0.001)
        assert runtime.started == [100]
        with pytest.raises(BotError):
            jobs.cancel(100, 99, 5)
        jobs.cancel(100, 1, 5)
        runtime.release.set()
        await asyncio.wait_for(jobs.queue.join(), 2)
        assert store.job(100)["status"] == "cancelled"
        assert store.job(101)["result"] == "TWO"
    finally:
        runtime.release.set()
        await jobs.shutdown()


async def test_shutdown_recovers_queued_jobs(settings, store):
    settings.shutdown_timeout_seconds = 1
    runtime = RecordingRuntime()
    jobs = Jobs(settings, store, runtime)
    jobs.start()
    jobs.submit(100, 1, 5, "one")
    jobs.submit(101, 1, 5, "two")
    await jobs.shutdown()
    assert store.job(100)["status"] == "interrupted"
    assert store.job(101)["status"] == "interrupted"

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from t_bot.development_progress import DevelopmentEvent, ProgressReporter
from t_bot.manifest import Manifest
from t_bot.runtime import Invocation, current_invocation


@pytest.mark.sandbox
async def test_parallel_builds_report_each_stage_only_to_origin_chat(
    components, store, monkeypatch
):
    runtime, _, _, _ = components

    async def generate(model, prompt, schema, **kwargs):
        fields = schema["properties"]["manifest"]["properties"]
        manifest = Manifest(
            id=fields["id"]["const"],
            kind="tool",
            version=1,
            description="Example",
            example_usage="Use example",
        )
        return json.dumps(
            {
                "manifest": manifest.model_dump(),
                "source": '''def invoke(payload: dict) -> dict:
    """Return a value; PRIVATE_SOURCE must not appear in progress."""
    return {"ok": True}
''',
            }
        )

    monkeypatch.setattr(runtime.ollama, "generate", AsyncMock(side_effect=generate))

    async def create(extension_id, chat_id, job_id):
        token = current_invocation.set(Invocation(1, chat_id, job_id))
        try:
            return await runtime.create_extension("PRIVATE_REQUEST", "tool", extension_id)
        finally:
            current_invocation.reset(token)

    results = await asyncio.gather(create("first_tool", 555, 101), create("second_tool", 777, 102))
    assert all(result["status"] == "active" for result in results)
    rows = list(store.connection.execute("SELECT chat_id,text FROM outbox ORDER BY id"))
    for chat_id, extension_id, job_id in [(555, "first_tool", 101), (777, "second_tool", 102)]:
        messages = [row["text"] for row in rows if row["chat_id"] == chat_id]
        assert messages and all(
            text.startswith(f"{extension_id} · задача {job_id}\n") for text in messages
        )
        joined = "\n".join(messages)
        for label in [
            "PM складає",
            "Аналітик планує",
            "шукає документацію",
            "Аналітик уточнює",
            "PM затверджує",
            "Developer",
            "QA пише",
            "QA-тестів",
            "QA аналізує",
            "Code reviewer",
            "Активація",
        ]:
            assert label in joined
        assert "/build " in joined and "Активовано версію 1" in messages[-1]
        assert "PRIVATE_SOURCE" not in joined and "PRIVATE_REQUEST" not in joined


async def test_heartbeat_keeps_user_informed_until_role_finishes(monkeypatch):
    monkeypatch.setattr("t_bot.development_progress.HEARTBEAT_SECONDS", 0.01)
    release = asyncio.Event()
    events = []

    def notify(event):
        events.append(event)
        release.set()

    async def operation():
        await release.wait()
        return {"done": True}

    result = await asyncio.wait_for(
        ProgressReporter(notify).wait(
            operation(), DevelopmentEvent("developer", "started", model="gemma4:12b")
        ),
        1,
    )
    assert result == {"done": True}
    assert events and events[0].status == "running" and events[0].phase == "developer"


async def test_cancellation_stops_role_behind_heartbeat():
    started, stopped = asyncio.Event(), asyncio.Event()

    async def operation():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            stopped.set()

    task = asyncio.create_task(
        ProgressReporter(lambda event: None).wait(
            operation(), DevelopmentEvent("developer", "started")
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


async def test_notification_failure_does_not_interrupt_development():
    def unavailable(event):
        raise RuntimeError("PRIVATE_TRANSPORT_ERROR")

    reporter = ProgressReporter(unavailable)
    reporter.emit(DevelopmentEvent("developer", "started"))

    async def operation():
        return {"done": True}

    assert await reporter.wait(operation(), DevelopmentEvent("developer", "started")) == {
        "done": True
    }

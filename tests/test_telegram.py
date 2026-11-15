from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from t_bot.errors import BotError
from t_bot.jobs import Jobs
from t_bot.telegram import TelegramAdapter, chunks


def test_telegram_unicode_limits():
    text = "аб😀" * 4000
    parts = chunks(text)
    assert "".join(parts) == text
    assert all(len(part.encode("utf-16-le")) // 2 <= 3500 for part in parts)


async def test_commands_and_access(settings, store, components, monkeypatch):
    runtime, extensions, _, control = components
    jobs = Jobs(settings, store, runtime)
    adapter = TelegramAdapter(settings, store, runtime.ollama, extensions, jobs, control)

    async def models():
        return ["gemma4:12b", "other:latest"]

    async def require_model(model):
        if model not in await models():
            raise BotError("Missing model")

    monkeypatch.setattr(runtime.ollama, "models", models)
    monkeypatch.setattr(runtime.ollama, "require_model", require_model)
    assert "gemma4:12b" in await adapter.command("/models", 1, 5)
    await adapter.command("/model other:latest", 1, 5)
    assert store.get("model:root") == "other:latest"
    with pytest.raises(BotError):
        await adapter.command("/restart", 99, 5)
    with pytest.raises(BotError):
        await adapter.command("/model absent", 1, 5)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        effective_chat=SimpleNamespace(id=5),
        effective_message=SimpleNamespace(text="hello"),
        update_id=100,
    )
    context = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
    await adapter.handle(update, context)
    await adapter.handle(update, context)
    assert jobs.queue.qsize() == 1


async def test_open_access_accepts_messages_and_reports_actual_errors(settings, store, components):
    settings.allowed_user_ids = set()
    runtime, extensions, _, control = components
    jobs = Jobs(settings, store, runtime)
    adapter = TelegramAdapter(settings, store, runtime.ollama, extensions, jobs, control)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=99),
        effective_chat=SimpleNamespace(id=5),
        effective_message=SimpleNamespace(text="hello"),
        update_id=200,
    )
    context = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
    await adapter.handle(update, context)
    assert store.job(200)["user_id"] == 99
    update.update_id = 201
    update.effective_message.text = "/cancel"
    await adapter.handle(update, context)
    assert store.pending_messages()[-1]["text"] == "/cancel TASK_ID"
    context.bot.send_message.assert_not_awaited()

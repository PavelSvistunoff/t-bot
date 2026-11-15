from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from t_bot.errors import BotError
from t_bot.jobs import Jobs
from t_bot.runtime import Invocation, current_invocation
from t_bot.telegram import TelegramAdapter


@pytest.mark.parametrize(
    "text", ["допомога", "HELP", " Допомога?! ", "/help", "/help@my_bot", "/start"]
)
async def test_help_shortcuts_are_deduplicated_without_model(components, settings, store, text):
    runtime, extensions, _, control = components
    jobs = Jobs(settings, store, runtime)
    adapter = TelegramAdapter(settings, store, runtime.ollama, extensions, jobs, control)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        effective_chat=SimpleNamespace(id=555),
        effective_message=SimpleNamespace(text=text),
        update_id=100,
    )
    context = SimpleNamespace(bot=SimpleNamespace(username="my_bot", send_message=AsyncMock()))
    await adapter.handle(update, context)
    await adapter.handle(update, context)
    messages = store.pending_messages()
    assert len(messages) == 1 and messages[0]["chat_id"] == 555
    guide = messages[0]["text"]
    for tool in runtime.basic_tools.functions:
        assert tool.__name__ in guide
    assert "/cancel TASK_ID" in guide and "Приклад:" in guide
    assert jobs.queue.empty()
    update.effective_message.text = "/help@other_bot"
    update.update_id = 101
    await adapter.handle(update, context)
    assert len(store.pending_messages()) == 1


async def test_help_tool_authorizes_and_matches_dynamic_telegram_guide(
    components, settings, store, monkeypatch
):
    from t_bot.manifest import Manifest

    runtime, extensions, _, control = components
    manifests = []
    monkeypatch.setattr(
        extensions, "records", lambda: [SimpleNamespace(manifest=m) for m in manifests]
    )
    token = current_invocation.set(Invocation(99, 555, 1))
    try:
        with pytest.raises(BotError):
            await runtime.basic_tools.get_help()
        settings.allowed_user_ids = set()
        before = await runtime.basic_tools.get_help()
        manifests.append(
            Manifest(
                id="translator",
                kind="agent",
                version=1,
                description="Перекладач",
                example_usage="Попроси translator перекласти текст українською.",
            )
        )
        after = await runtime.basic_tools.get_help()
        assert "translator" not in before["text"]
        assert manifests[0].example_usage in after["text"]
        jobs = Jobs(settings, store, runtime)
        adapter = TelegramAdapter(settings, store, runtime.ollama, extensions, jobs, control)
        assert await adapter.command("/help", 99, 555) == after["text"]
        manifests.clear()
        assert "translator" not in (await runtime.basic_tools.get_help())["text"]
    finally:
        current_invocation.reset(token)

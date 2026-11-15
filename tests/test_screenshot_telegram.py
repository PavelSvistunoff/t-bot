import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import BadRequest, NetworkError

from t_bot.errors import BotError
from t_bot.jobs import Jobs
from t_bot.runtime import Invocation, current_invocation
from t_bot.screenshot import ScreenImage
from t_bot.telegram import TelegramAdapter, is_screenshot_request

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII="
)


@pytest.fixture
async def screen_adapter(settings, store, components, monkeypatch):
    runtime, extensions, _, control = components
    jobs = Jobs(settings, store, runtime)
    adapter = TelegramAdapter(settings, store, runtime.ollama, extensions, jobs, control)
    capture = AsyncMock(return_value=ScreenImage(PNG, 1, 1))
    photo = AsyncMock(return_value=SimpleNamespace(message_id=42))
    document = AsyncMock(return_value=SimpleNamespace(message_id=43))
    monkeypatch.setattr(adapter.screen_capture, "capture", capture)
    monkeypatch.setattr(type(adapter.application.bot), "send_photo", photo)
    monkeypatch.setattr(type(adapter.application.bot), "send_document", document)
    runtime.screenshot_sender = adapter.send_screenshot
    return adapter, runtime, capture, photo, document


@pytest.mark.parametrize(
    "text",
    [
        "Що зараз на екрані",
        "Що на екрані?",
        "  ЩО   НА ЕКРАНІ?! ",
        "/screenshot",
        "/screenshot@my_bot",
    ],
)
async def test_phrases_send_actual_image_to_origin_chat_once(screen_adapter, text):
    adapter, _, capture, photo, document = screen_adapter
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        effective_chat=SimpleNamespace(id=555),
        effective_message=SimpleNamespace(text=text),
        update_id=100,
    )
    context = SimpleNamespace(bot=SimpleNamespace(username="my_bot"))
    await adapter.handle(update, context)
    await adapter.handle(update, context)
    capture.assert_awaited_once()
    photo.assert_awaited_once()
    assert photo.call_args.kwargs["chat_id"] == 555
    assert photo.call_args.kwargs["photo"].input_file_content == PNG
    document.assert_not_awaited()
    assert adapter.jobs.queue.empty()


async def test_agent_tool_targets_context_chat_and_returns_metadata_only(screen_adapter):
    _, runtime, _, photo, _ = screen_adapter
    token = current_invocation.set(Invocation(1, 888, 101))
    try:
        result = await runtime.basic_tools.take_screenshot()
    finally:
        current_invocation.reset(token)
    assert result["status"] == "sent"
    assert result["message_id"] == 42
    assert photo.call_args.kwargs["chat_id"] == 888
    assert not any(isinstance(value, bytes) for value in result.values())


async def test_unauthorized_request_never_captures(screen_adapter):
    adapter, runtime, capture, photo, _ = screen_adapter
    with pytest.raises(BotError):
        await adapter.send_screenshot(99, 555)
    token = current_invocation.set(Invocation(99, 555, 102))
    try:
        with pytest.raises(BotError):
            await runtime.basic_tools.take_screenshot()
    finally:
        current_invocation.reset(token)
    capture.assert_not_awaited()
    photo.assert_not_awaited()


async def test_open_access_can_request_screenshot(screen_adapter, settings):
    adapter, _, _, photo, _ = screen_adapter
    settings.allowed_user_ids = set()
    await adapter.send_screenshot(99, 555)
    assert photo.call_args.kwargs["chat_id"] == 555


async def test_photo_rejection_or_oversized_dimensions_use_document(screen_adapter):
    adapter, _, capture, photo, document = screen_adapter
    photo.side_effect = BadRequest("Photo_invalid_dimensions")
    assert (await adapter.send_screenshot(1, 555))["delivery"] == "document"
    assert document.call_args.kwargs["document"].input_file_content == PNG
    photo.reset_mock()
    document.reset_mock()
    capture.return_value = ScreenImage(PNG, 10000, 10000)
    assert (await adapter.send_screenshot(1, 555))["delivery"] == "document"
    photo.assert_not_awaited()


async def test_unknown_delivery_failure_is_not_retried(screen_adapter):
    adapter, _, capture, photo, document = screen_adapter
    photo.side_effect = NetworkError("offline")
    with pytest.raises(BotError, match="не повторюється"):
        await adapter.send_screenshot(1, 555)
    capture.assert_awaited_once()
    document.assert_not_awaited()


async def test_capture_failure_is_reported_to_requesting_chat(screen_adapter):
    adapter, _, capture, photo, _ = screen_adapter
    capture.side_effect = BotError("Немає дозволу на скріншоти")
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        effective_chat=SimpleNamespace(id=555),
        effective_message=SimpleNamespace(text="Що на екрані"),
        update_id=100,
    )
    await adapter.handle(update, SimpleNamespace(bot=adapter.application.bot))
    assert adapter.store.pending_messages()[0]["text"] == "Немає дозволу на скріншоти"
    photo.assert_not_awaited()


def test_unrelated_text_does_not_trigger_automatic_capture():
    assert not is_screenshot_request("Не роби скріншот")
    assert not is_screenshot_request("Що на екрані означає ця помилка?")


async def test_command_addressed_to_another_bot_does_not_capture(screen_adapter):
    adapter, _, capture, photo, _ = screen_adapter
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        effective_chat=SimpleNamespace(id=555),
        effective_message=SimpleNamespace(text="/screenshot@other_bot"),
        update_id=100,
    )
    await adapter.handle(update, SimpleNamespace(bot=SimpleNamespace(username="my_bot")))
    capture.assert_not_awaited()
    photo.assert_not_awaited()

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import NetworkError

from t_bot.errors import BotError
from t_bot.jobs import Jobs
from t_bot.runtime import Invocation, current_invocation
from t_bot.telegram import TelegramAdapter


@pytest.fixture(autouse=True)
def workspace(settings):
    settings.filesystem_root.mkdir()


@pytest.fixture
async def attachment_adapter(components, settings, store, monkeypatch):
    runtime, extensions, _, control = components
    adapter = TelegramAdapter(
        settings, store, runtime.ollama, extensions, Jobs(settings, store, runtime), control
    )
    document = AsyncMock(return_value=SimpleNamespace(message_id=42))
    monkeypatch.setattr(type(adapter.application.bot), "send_document", document)
    runtime.file_sender = adapter.send_file
    return runtime, document


async def test_binary_attachment_targets_invocation_chat_and_returns_metadata(
    attachment_adapter, settings
):
    runtime, document = attachment_adapter
    data = b"\x00\xffPRIVATE BINARY CONTENT"
    (settings.filesystem_root / "звіт.pdf").write_bytes(data)
    token = current_invocation.set(Invocation(1, 888, 101))
    try:
        result = await runtime.basic_tools.send_file("звіт.pdf")
    finally:
        current_invocation.reset(token)
    document.assert_awaited_once()
    sent = document.call_args.kwargs
    assert sent["chat_id"] == 888
    assert sent["document"].input_file_content == data
    assert sent["document"].filename == "звіт.pdf"
    assert result == {
        "status": "sent",
        "chat_id": 888,
        "message_id": 42,
        "filename": "звіт.pdf",
        "size_bytes": len(data),
        "delivery": "document",
    }
    assert "PRIVATE BINARY CONTENT" not in str(result)


@pytest.mark.parametrize(
    "path", ["../outside.txt", ".env", "link", "folder", "missing", "empty", "large", "hardlink"]
)
async def test_invalid_files_are_never_uploaded(attachment_adapter, settings, tmp_path, path):
    runtime, document = attachment_adapter
    root = settings.filesystem_root
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"SECRET")
    (root / ".env").write_text("TOKEN")
    (root / "link").symlink_to(outside)
    os.link(outside, root / "hardlink")
    (root / "folder").mkdir()
    (root / "empty").touch()
    (root / "large").write_bytes(b"12345")
    runtime.basic_tools.attachment_max_bytes = 4
    token = current_invocation.set(Invocation(1, 888, 101))
    try:
        assert "error" in await runtime.basic_tools.send_file(path)
    finally:
        current_invocation.reset(token)
    document.assert_not_awaited()


async def test_attachment_access_open_mode_and_no_network_retry(attachment_adapter, settings):
    runtime, document = attachment_adapter
    (settings.filesystem_root / "notes.txt").write_text("text")
    token = current_invocation.set(Invocation(99, 555, 101))
    try:
        with pytest.raises(BotError):
            await runtime.basic_tools.send_file("notes.txt")
        document.assert_not_awaited()
        settings.allowed_user_ids = set()
        document.side_effect = NetworkError("lost response")
        result = await runtime.basic_tools.send_file("notes.txt")
        assert "error" in result and "автоматично не повторюється" in result["error"]
        document.assert_awaited_once()
        document.reset_mock()
        document.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await runtime.basic_tools.send_file("notes.txt")
    finally:
        current_invocation.reset(token)


async def test_disconnected_adapter_reports_error(components, settings):
    runtime, _, _, _ = components
    (settings.filesystem_root / "notes.txt").write_text("text")
    token = current_invocation.set(Invocation(1, 555, 101))
    try:
        assert "error" in await runtime.basic_tools.send_file("notes.txt")
    finally:
        current_invocation.reset(token)

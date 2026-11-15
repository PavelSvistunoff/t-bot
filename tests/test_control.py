import asyncio

import pytest

from t_bot.control import RestartControl
from t_bot.errors import BotError
from t_bot.supervisor import process_lock


def test_restart_persists_and_is_rate_limited(settings, store, monkeypatch):
    stop = asyncio.Event()
    control = RestartControl(settings, store, stop)
    with pytest.raises(BotError):
        control.request(1, 5, "restart")
    monkeypatch.setenv("T_BOT_SUPERVISED", "1")
    control.request(1, 5, "restart")
    assert stop.is_set()
    assert store.get("restart_pending")["user_id"] == 1
    assert store.get("restart_pending")["chat_id"] == 5
    another = RestartControl(settings, store, asyncio.Event())
    with pytest.raises(BotError):
        another.request(1, 5, "again")


def test_single_instance_lock(tmp_path):
    with process_lock(tmp_path / "process.lock"):
        with pytest.raises(BotError):
            with process_lock(tmp_path / "process.lock"):
                pytest.fail("Second instance was accepted")


async def test_administrative_agent_tools_recheck_authorization(components):
    from t_bot.runtime import Invocation, current_invocation

    runtime, _, _, _ = components
    token = current_invocation.set(Invocation(99, 5, 1))
    try:
        with pytest.raises(BotError):
            await runtime.restart_bot("unauthorized")
        with pytest.raises(BotError):
            await runtime.create_extension("test", "tool", "test_tool")
        with pytest.raises(BotError):
            await runtime.set_model("gemma4:12b")
    finally:
        current_invocation.reset(token)


async def test_open_access_allows_administrative_agent_tools(settings, components, monkeypatch):
    from t_bot.runtime import Invocation, current_invocation

    settings.allowed_user_ids = set()
    runtime, _, _, control = components
    monkeypatch.setenv("T_BOT_SUPERVISED", "1")
    token = current_invocation.set(Invocation(99, 5, 200))
    try:
        result = await runtime.restart_bot("open access")
        assert "error" not in result
        assert control.stop.is_set()
    finally:
        current_invocation.reset(token)

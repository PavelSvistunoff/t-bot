from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from t_bot.config import Settings
from t_bot.current_time import current_datetime, format_datetime
from t_bot.errors import BotError
from t_bot.jobs import Jobs
from t_bot.runtime import Invocation, current_invocation
from t_bot.telegram import TelegramAdapter, is_time_request


@pytest.mark.parametrize(
    "instant,date,time,offset",
    [
        (datetime(2026, 1, 15, 22, 30, tzinfo=UTC), "2026-01-16", "00:30:00", "+02:00"),
        (datetime(2026, 3, 29, 0, 30, tzinfo=UTC), "2026-03-29", "02:30:00", "+02:00"),
        (datetime(2026, 3, 29, 1, 30, tzinfo=UTC), "2026-03-29", "04:30:00", "+03:00"),
    ],
)
def test_clock_converts_date_and_daylight_saving(monkeypatch, instant, date, time, offset):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz)

    monkeypatch.setattr("t_bot.current_time.datetime", FixedDateTime)
    value = current_datetime("Europe/Kyiv")
    assert value["date"] == date and value["time"] == time
    assert value["utc_offset"] == offset
    assert value["timezone"] == "Europe/Kyiv"
    assert datetime.fromisoformat(value["iso8601"]).astimezone(UTC) == instant
    assert time in format_datetime(value)


def test_timezone_configuration_rejects_unknown_zone(monkeypatch):
    monkeypatch.setenv("BOT_TIMEZONE", "UTC")
    assert Settings(_env_file=None).bot_timezone == "UTC"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, bot_timezone="Invalid/Zone")


@pytest.mark.parametrize(
    "text",
    ["Котра година", "Скока время", " КОТРА   ГОДИНА?! ", "Скока время?", "/time", "/time@my_bot"],
)
async def test_time_shortcuts_bypass_model_and_deduplicate(
    components,
    settings,
    store,
    monkeypatch,
    text,
):
    runtime, extensions, _, control = components
    jobs = Jobs(settings, store, runtime)
    adapter = TelegramAdapter(settings, store, runtime.ollama, extensions, jobs, control)
    value = {
        "date": "2026-01-16",
        "time": "00:30:00",
        "timezone": "Europe/Kyiv",
        "utc_offset": "+02:00",
        "iso8601": "2026-01-16T00:30:00+02:00",
    }
    monkeypatch.setattr("t_bot.telegram.current_datetime", lambda timezone: value)
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
    assert len(messages) == 1
    assert messages[0]["chat_id"] == 555
    assert "16.01.2026" in messages[0]["text"]
    assert "00:30:00" in messages[0]["text"]
    assert "Europe/Kyiv" in messages[0]["text"]
    assert jobs.queue.empty()


async def test_clock_tool_checks_access_and_supports_open_mode(components, settings):
    runtime, _, _, _ = components
    token = current_invocation.set(Invocation(99, 5, 1))
    try:
        with pytest.raises(BotError):
            await runtime.basic_tools.get_current_datetime()
        settings.allowed_user_ids = set()
        value = await runtime.basic_tools.get_current_datetime()
        assert value["timezone"] == settings.bot_timezone
        assert datetime.fromisoformat(value["iso8601"]).tzinfo is not None
    finally:
        current_invocation.reset(token)


def test_unrelated_question_is_not_a_clock_shortcut():
    assert not is_time_request("Котра година була вчора на цьому фото?")

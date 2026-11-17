"""Opt-in checks against the actual Ollama instance; no Telegram token is required."""

import os

import pytest
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from t_bot.runtime import APP_NAME

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE") != "1",
        reason="Set RUN_LIVE=1 to use local Ollama",
    ),
]


async def test_real_model_dialogue_and_session(components):
    runtime, _, _, _ = components
    answer = await runtime.run(1, 5, 100, "Відповідай лише словом: Працює")
    assert answer.strip()
    session = await runtime.sessions.get_session(
        app_name=APP_NAME,
        user_id="1",
        session_id="u1_c5_g0",
    )
    assert session and len(session.events) >= 2
    assert (
        await runtime.sessions.get_session(
            app_name=APP_NAME,
            user_id="2",
            session_id="u1_c5_g0",
        )
        is None
    )


async def test_real_model_structured_tool_and_delegation(components, monkeypatch):
    runtime, _, _, _ = components
    calls = []

    async def verify_code(code: str) -> dict:
        """Verify the exact code supplied by the user and return a proof."""
        calls.append(code)
        return {"proof": "verified-78123", "code": code}

    child = LlmAgent(
        name="verifier",
        model=runtime.model(runtime.settings.default_model),
        description="Verify user codes with verify_code tool",
        instruction="Call verify_code with the code from the user. Return its proof.",
        tools=[verify_code],
        before_model_callback=runtime.before_model,
    )
    root = LlmAgent(
        name="root",
        model=runtime.model(runtime.settings.default_model),
        instruction="You must delegate verification to verifier tool. Return its result.",
        tools=[AgentTool(agent=child)],
        before_model_callback=runtime.before_model,
    )
    monkeypatch.setattr(runtime, "build_agent", lambda: root)
    answer = await runtime.run(
        1, 5, 101, "Verify code AB123 using verifier, then give me the proof."
    )
    assert calls == ["AB123"]
    assert "78123" in answer


async def test_real_generation_and_execution(components):
    _, extensions, _, _ = components
    generated = await extensions.create(
        "Count words in payload text, returning count. No dependencies. "
        "JSON input has required text string, example text='hello world'.",
        "tool",
        "count_words",
        "root",
        "gemma4:12b",
        1,
    )
    assert generated["status"] == "active"
    record = extensions.get("count_words")
    result = await extensions.invoke(record, {"text": "one two three"}, 1)
    assert 3 in result.values()


async def test_real_model_requests_restart_and_preserves_session(components, monkeypatch):
    runtime, _, _, control = components
    monkeypatch.setenv("T_BOT_SUPERVISED", "1")
    await runtime.run(
        1,
        5,
        102,
        "Негайно перезапусти цього Telegram-бота. Виклич restart_bot з причиною integration_test.",
    )
    assert control.stop.is_set()
    assert runtime.store.get("restart_pending")["chat_id"] == 5
    from google.adk.sessions import DatabaseSessionService

    reopened = DatabaseSessionService(
        db_url=f"sqlite+aiosqlite:///{runtime.settings.data_dir / 'sessions.sqlite3'}",
    )
    try:
        session = await reopened.get_session(app_name=APP_NAME, user_id="1", session_id="u1_c5_g0")
        assert session and session.events
    finally:
        await reopened.close()


async def test_real_generated_agent_and_subagent(components):
    runtime, extensions, _, _ = components
    await extensions.create(
        "Agent that delegates text proofreading to its child editor tool and returns the result.",
        "agent",
        "writer",
        "root",
        "gemma4:12b",
        1,
    )
    await extensions.create(
        "Proofreading agent. Return the corrected text, fixing teh to the. No other output.",
        "subagent",
        "editor",
        "writer",
        "gemma4:12b",
        1,
    )
    response = await runtime.run(
        1,
        5,
        103,
        "Delegate to writer, which must use editor to fix this text: 'teh cat'.",
    )
    assert "the cat" in response.lower()


async def test_real_model_through_telegram_queue_and_outbox(
    components, settings, store, monkeypatch
):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from t_bot.jobs import Jobs
    from t_bot.telegram import TelegramAdapter

    runtime, extensions, _, control = components
    jobs = Jobs(settings, store, runtime)
    adapter = TelegramAdapter(settings, store, runtime.ollama, extensions, jobs, control)
    sent = AsyncMock()
    monkeypatch.setattr(type(adapter.application.bot), "send_message", sent)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        effective_chat=SimpleNamespace(id=5),
        effective_message=SimpleNamespace(text="Відповідай: тест успішний"),
        update_id=104,
    )
    jobs.start()
    try:
        await adapter.handle(update, SimpleNamespace(bot=adapter.application.bot))
        await asyncio.wait_for(jobs.queue.join(), 120)
        await adapter.flush()
        assert store.job(104)["status"] == "done"
        assert sent.await_count >= 3
        assert not store.pending_messages()
    finally:
        await jobs.shutdown()


@pytest.mark.skipif(not os.environ.get("LIVE_ALTERNATE_MODEL"), reason="Set LIVE_ALTERNATE_MODEL")
async def test_real_alternative_model_is_persistent(components):
    runtime, _, _, _ = components
    from t_bot.runtime import Invocation, current_invocation
    from t_bot.store import Store

    model = os.environ["LIVE_ALTERNATE_MODEL"]
    token = current_invocation.set(Invocation(1, 5, 105))
    try:
        assert (await runtime.set_model(model))["model"] == model
    finally:
        current_invocation.reset(token)
    reopened = Store(runtime.store.path)
    try:
        assert reopened.get("model:root") == model
    finally:
        reopened.close()
    assert (await runtime.run(1, 5, 105, "Reply with OK")).strip()


async def test_natural_language_creates_and_runs_new_tool(components):
    runtime, extensions, _, _ = components
    await runtime.run(
        1,
        5,
        106,
        "Створи й активуй Python-інструмент letters (kind=tool, parent=root): "
        "вхід payload має text: str; результат JSON count: int — довжина text. "
        "Без залежностей. Виклич create_extension зараз, а не просто надай код.",
    )
    assert extensions.get("letters").manifest.kind == "tool"
    answer = await runtime.run(
        1,
        5,
        107,
        'Виклич інструмент letters для text="hello" і повідом кількість символів.',
    )
    invocation = runtime.store.connection.execute(
        "SELECT 1 FROM audit WHERE action='extension.invoke' AND details='letters'",
    ).fetchone()
    assert invocation
    assert "5" in answer


@pytest.mark.skipif(os.environ.get("RUN_DEPENDENCY_TESTS") != "1", reason="Requires PyPI access")
async def test_approved_dependency_installs_into_isolated_environment(
    components, settings, tmp_path
):
    _, _, sandbox, _ = components
    from t_bot.manifest import Manifest

    settings.allowed_dependencies = {"six==1.17.0"}
    manifest = Manifest(
        id="dependency_probe",
        kind="tool",
        version=1,
        description="Check approved dependency",
        dependencies=["six==1.17.0"],
        example_usage="Check that six is installed",
    )
    (tmp_path / "extension.py").write_text(
        "import six\n"
        "def invoke(payload: dict) -> dict:\n"
        '    """Check a dependency loaded from the dedicated directory."""\n'
        '    return {"is_string": isinstance("hello", six.string_types), "path": six.__file__}\n'
    )
    result = await sandbox.execute(tmp_path, manifest.model_dump(), "invoke", {})
    assert result["is_string"]
    assert str(sandbox.dependency_path(manifest.dependencies)) in result["path"]


async def test_real_model_uses_filesystem_tools(components, settings):
    runtime, _, _, _ = components
    response = await runtime.run(
        1,
        5,
        108,
        "Створи файл hello.txt з вмістом HELLO_WORKSPACE через write_file, "
        "потім прочитай його через read_file і поверни прочитаний вміст.",
    )
    assert (settings.filesystem_root / "hello.txt").read_text() == "HELLO_WORKSPACE"
    assert "HELLO_WORKSPACE" in response
    operations = [
        row[0]
        for row in runtime.store.connection.execute(
            "SELECT action FROM audit WHERE action LIKE 'filesystem.%'",
        )
    ]
    assert "filesystem.write" in operations and "filesystem.read" in operations


@pytest.mark.skipif(os.environ.get("RUN_WEB_TESTS") != "1", reason="Requires public search engines")
async def test_real_web_search_from_agent(components):
    runtime, _, _, _ = components
    response = await runtime.run(
        1,
        5,
        109,
        "Знайди Python documentation в інтернеті через search_web (max_results=3). "
        "Відповідай посиланнями зі справжніх результатів пошуку.",
    )
    assert "https://" in response
    assert runtime.store.connection.execute(
        "SELECT 1 FROM audit WHERE action='web.search'",
    ).fetchone()


async def test_real_model_uses_sqlite_tools(components):
    runtime, _, _, _ = components
    response = await runtime.run(
        1,
        5,
        110,
        "Використай SQLite-інструменти: створи базу diary.db, створи таблицю notes "
        "з колонками id INTEGER PRIMARY KEY і text TEXT, додай рядок text='hello SQLite', "
        "покажи структуру notes й виконай SELECT text FROM notes. "
        "Поверни фактично прочитаний текст. Не використовуй write_file для бази.",
    )
    result = runtime.basic_tools.sqlite.execute_sql("diary.db", "SELECT text FROM notes")
    assert result["rows"] == [["hello SQLite"]]
    assert "hello SQLite" in response
    actions = {
        row[0]
        for row in runtime.store.connection.execute(
            "SELECT action FROM audit WHERE action LIKE 'sqlite.%'",
        )
    }
    assert "sqlite.create" in actions and "sqlite.table" in actions


async def test_real_model_uses_console_tool(components):
    runtime, _, _, _ = components
    response = await runtime.run(
        1,
        5,
        111,
        "Виконай у консолі команду printf HELLO_CONSOLE через run_command. "
        "Поверни stdout і код завершення фактично виконаної команди.",
    )
    assert "HELLO_CONSOLE" in response
    assert runtime.store.connection.execute(
        "SELECT 1 FROM audit WHERE action='console.run'",
    ).fetchone()


async def test_real_model_requests_screenshot_and_sends_it_to_origin_chat(
    components,
    settings,
    store,
    monkeypatch,
):
    import base64
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from t_bot.jobs import Jobs
    from t_bot.screenshot import ScreenImage
    from t_bot.telegram import TelegramAdapter

    runtime, extensions, _, control = components
    adapter = TelegramAdapter(
        settings,
        store,
        runtime.ollama,
        extensions,
        Jobs(settings, store, runtime),
        control,
    )
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII="
    )
    monkeypatch.setattr(
        adapter.screen_capture, "capture", AsyncMock(return_value=ScreenImage(png, 1, 1))
    )
    photo = AsyncMock(return_value=SimpleNamespace(message_id=42))
    monkeypatch.setattr(type(adapter.application.bot), "send_photo", photo)
    runtime.screenshot_sender = adapter.send_screenshot
    response = await runtime.run(
        1,
        777,
        112,
        "Зроби скріншот екрана комп’ютера бота через take_screenshot і надішли у поточний чат.",
    )
    assert response.strip()
    photo.assert_awaited_once()
    assert photo.call_args.kwargs["chat_id"] == 777
    assert photo.call_args.kwargs["photo"].input_file_content == png


@pytest.mark.skipif(os.environ.get("RUN_TEAM_TESTS") != "1", reason="Set RUN_TEAM_TESTS=1")
async def test_real_extension_team_creates_tested_tool(components, settings):
    _, extensions, _, _ = components
    settings.extension_team_timeout_seconds = 600
    result = await extensions.create(
        "Create a tool doubling integer number, returning doubled. "
        "Reject bool and strings with ValueError; cover zero and negatives. No dependencies.",
        "tool",
        "live_team_doubler",
        "root",
        settings.default_model,
        1,
        dependencies=[],
    )
    assert result["status"] == "active" and result["qa"]["passed"]
    assert result["qa"]["tests_run"] >= 3 and result["review"]["decision"] == "approve"
    record = extensions.get("live_team_doubler")
    assert await extensions.invoke(record, {"number": -3}, 1) == {"doubled": -6}

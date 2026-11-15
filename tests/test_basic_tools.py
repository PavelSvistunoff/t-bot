import pytest
from google.adk.tools import FunctionTool

from t_bot.errors import BotError
from t_bot.runtime import Invocation, current_invocation
from t_bot.tool_names import BASIC_TOOL_DESCRIPTIONS


async def test_builtins_are_discoverable_and_have_valid_adk_schemas(components):
    runtime, _, _, _ = components
    root = runtime.build_agent()
    names = {getattr(tool, "__name__", getattr(tool, "name", "")) for tool in root.tools}
    assert set(BASIC_TOOL_DESCRIPTIONS) <= names
    for function in runtime.basic_tools.functions:
        declaration = FunctionTool(function)._get_declaration()
        assert declaration and declaration.name == function.__name__
    token = current_invocation.set(Invocation(1, 5, 1))
    try:
        resources = await runtime.list_resources("extensions")
        assert resources["builtin_tools"] == BASIC_TOOL_DESCRIPTIONS
    finally:
        current_invocation.reset(token)


async def test_authorized_file_operations_and_expected_errors(components):
    runtime, _, _, _ = components
    token = current_invocation.set(Invocation(1, 5, 1))
    try:
        assert "error" not in await runtime.basic_tools.create_directory("notes")
        assert "error" not in await runtime.basic_tools.write_file("notes/hello.txt", "hello")
        assert (await runtime.basic_tools.read_file("notes/hello.txt"))["content"] == "hello"
        assert "error" in await runtime.basic_tools.read_file("missing.txt")
        assert "error" in await runtime.basic_tools.read_file("../.env")
        assert "error" not in await runtime.basic_tools.delete_file("notes/hello.txt")
    finally:
        current_invocation.reset(token)


async def test_both_tool_groups_recheck_access(components):
    runtime, _, _, _ = components
    token = current_invocation.set(Invocation(99, 5, 1))
    try:
        with pytest.raises(BotError, match="Доступ"):
            await runtime.basic_tools.list_files()
        with pytest.raises(BotError, match="Доступ"):
            await runtime.basic_tools.search_web("Python")
    finally:
        current_invocation.reset(token)


async def test_open_access_search_and_provider_error_are_returned(
    components, settings, monkeypatch
):
    runtime, _, _, _ = components
    settings.allowed_user_ids = set()

    def search(query, count):
        return {"results": [{"title": "Python", "url": "https://python.org", "snippet": "Docs"}]}

    monkeypatch.setattr(runtime.basic_tools.web, "search", search)
    token = current_invocation.set(Invocation(99, 5, 1))
    try:
        assert (await runtime.basic_tools.search_web("Python"))["results"]

        def fail(*args):
            raise BotError("Пошук недоступний")

        monkeypatch.setattr(runtime.basic_tools.web, "search", fail)
        assert (await runtime.basic_tools.search_web("Python"))["error"] == "Пошук недоступний"
    finally:
        current_invocation.reset(token)


async def test_sqlite_tools_check_access_and_support_open_mode(components, settings):
    runtime, _, _, _ = components
    token = current_invocation.set(Invocation(99, 5, 1))
    try:
        with pytest.raises(BotError, match="Доступ"):
            await runtime.basic_tools.list_databases()
        settings.allowed_user_ids = set()
        assert "error" not in await runtime.basic_tools.create_database("test.db")
        assert "error" not in await runtime.basic_tools.execute_sql("test.db", "CREATE TABLE t(x)")
        assert "error" not in await runtime.basic_tools.execute_sql(
            "test.db", "INSERT INTO t VALUES (1)"
        )
        assert (await runtime.basic_tools.execute_sql("test.db", "SELECT x FROM t"))["rows"] == [
            [1]
        ]
        assert "error" in await runtime.basic_tools.execute_sql("state", "DELETE FROM jobs")
    finally:
        current_invocation.reset(token)


async def test_console_tool_access_and_open_mode(components, settings, monkeypatch):
    runtime, _, _, _ = components
    calls = []

    async def run(command, cwd, timeout_seconds):
        calls.append(command)
        return {"exit_code": 0, "stdout": "OK", "stderr": "", "timed_out": False}

    monkeypatch.setattr(runtime.basic_tools.console, "run", run)
    token = current_invocation.set(Invocation(99, 5, 1))
    try:
        with pytest.raises(BotError, match="Доступ"):
            await runtime.basic_tools.run_command("echo OK")
        assert not calls
        settings.allowed_user_ids = set()
        assert (await runtime.basic_tools.run_command("echo OK"))["stdout"] == "OK"
        assert calls == ["echo OK"]
    finally:
        current_invocation.reset(token)

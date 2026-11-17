import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from t_bot.errors import BotError
from t_bot.extension_team import SPEC_SCHEMA, TeamFailure
from t_bot.manifest import Manifest
from t_bot.runtime import Invocation, current_invocation
from t_bot.team_agents import RoleAgents

pytestmark = pytest.mark.team
SOURCE = '''def invoke(payload: dict) -> dict:
    """Double an integer; reject other values."""
    number = payload.get("number")
    if not isinstance(number, int) or isinstance(number, bool):
        raise ValueError("Expected an integer")
    return {"doubled": number * 2}
'''
TESTS = """import unittest
import extension

class DoublingTests(unittest.TestCase):
    def test_happy_double(self):
        self.assertEqual(extension.invoke({"number": 3}), {"doubled": 6})
    def test_failure_not_integer(self):
        with self.assertRaises(ValueError):
            extension.invoke({"number": "three"})
    def test_edge_zero(self):
        self.assertEqual(extension.invoke({"number": 0}), {"doubled": 0})
"""


def artifact():
    return {
        "source": SOURCE,
        "manifest": Manifest(
            id="doubler",
            kind="tool",
            version=1,
            description="Double",
            example_usage="Double 3",
            input_schema={
                "type": "object",
                "properties": {"number": {"type": "integer"}},
                "required": ["number"],
            },
            example={"number": 3},
        ).model_dump(),
    }


def spec():
    return {
        name: "Double integer" if name == "objective" else ["Check behavior"]
        for name in SPEC_SCHEMA["required"]
    }


def research(team, monkeypatch):
    monkeypatch.setattr(
        team.web,
        "search",
        lambda *args: {
            "results": [
                {
                    "url": "https://docs.python.org/3/library/unittest.html",
                    "title": "Official docs",
                }
            ]
        },
    )


@pytest.mark.sandbox
async def test_team_rework_routes_and_real_qa(components, monkeypatch):
    _, extensions, _, _ = components
    research(extensions.team, monkeypatch)
    calls = []
    counts = {"developer": 0, "reviewer": 0}

    async def ask(role, model, context, schema):
        calls.append(role)
        if role == "Developer":
            counts["developer"] += 1
            result = artifact()
            if counts["developer"] == 1:
                result["source"] = SOURCE.replace("number * 2}", "number * 2 + 1}")
            else:
                assert "AssertionError" in context
            return result
        if role == "QA":
            if "decision" in schema["properties"]:
                passed = json.loads(context)["test_results"]["passed"]
                return {"decision": "approve" if passed else "developer", "findings": []}
            return {"tests": TESTS, "notes": "Positive, invalid type and zero boundary."}
        if role == "Reviewer":
            counts["reviewer"] += 1
            return {
                "decision": ["developer", "qa", "approve"][counts["reviewer"] - 1],
                "findings": [],
            }
        if "queries" in schema["properties"]:
            return {"queries": ["Python unittest official docs"]}
        return spec()

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    result = await extensions.create("Double integers", "tool", "doubler", "root", "", 1)
    assert counts == {"developer": 2, "reviewer": 3}
    assert calls[:4] == ["PM", "Analyst", "Analyst", "PM"]
    assert result["qa"]["passed"] and result["qa"]["tests_run"] == 3
    report = extensions.team.report(result["team_run_id"], 1)
    assert report["status"] == "active" and report["final_spec"] == spec()
    routes = [e["data"]["target"] for e in report["events"] if e["stage"] == "Rework"]
    assert routes == ["developer", "qa"]
    record = extensions.get("doubler")
    assert (extensions.directory(record.manifest) / "tests.py").read_text() == TESTS
    assert await extensions.invoke(record, {"number": -2}, 1) == {"doubled": -4}


async def test_team_exhaustion_keeps_artifacts_without_activation(
    components, settings, monkeypatch
):
    _, extensions, _, _ = components
    settings.extension_team_max_rounds = 2
    research(extensions.team, monkeypatch)

    async def ask(role, model, context, schema):
        if role == "Developer":
            return artifact()
        if role == "QA" and "tests" in schema["properties"]:
            return {"tests": TESTS, "notes": "Tests"}
        if role in {"QA", "Reviewer"}:
            return {"decision": "approve", "findings": []}
        if "queries" in schema["properties"]:
            return {"queries": ["Python docs"]}
        return spec()

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    validate = AsyncMock(side_effect=lambda candidate: candidate)
    execute = AsyncMock(return_value={"passed": False, "failures": ["Wrong answer"]})
    with pytest.raises(TeamFailure) as failed:
        await extensions.team.run(
            request="Double",
            extension_id="doubler",
            user_id=1,
            model="fake",
            developer_context="Implement",
            developer_schema={},
            validate=validate,
            execute_tests=execute,
        )
    report = extensions.team.report(failed.value.report["id"], 1)
    assert report["status"] == "failed" and report["candidate"] == artifact()
    assert report["tests"] == TESTS and not extensions.records()
    assert execute.await_count == 2
    with pytest.raises(BotError):
        extensions.team.report(report["id"], 99)


async def test_team_fallback_and_restart_recovery(components, settings, monkeypatch):
    runtime, extensions, _, _ = components
    settings.extension_team_fallback_models = ["fallback"]
    ask = AsyncMock(side_effect=BotError("Ollama offline"))
    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    token = current_invocation.set(Invocation(1, 5, 123))
    try:
        result = await runtime.create_extension("Double integer", "tool", "doubler")
    finally:
        current_invocation.reset(token)
    assert result["status"] == "not_activated" and result["team_run_id"]
    assert [c.args[1] for c in ask.call_args_list] == ["gemma4:12b", "fallback"]
    report = extensions.team.report(result["team_run_id"], 1)
    assert report["status"] == "failed" and "PM" in report["error"]
    report["status"] = "running"
    await extensions.team.save(report)
    await extensions.team.recover()
    assert extensions.team.report(report["id"], 1)["status"] == "interrupted"


async def test_team_cancellation_persists_report(components, monkeypatch):
    _, extensions, _, _ = components
    started = asyncio.Event()

    async def ask(*args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    task = asyncio.create_task(extensions.create("Double", "tool", "doubler", "root", "", 1))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert extensions.team.reports(1)[0]["status"] == "cancelled"


async def test_roles_run_through_adk():
    client = type("Client", (), {})()
    client.generate = AsyncMock(return_value=json.dumps(spec()))
    result = await RoleAgents(client).ask("PM", "fake", "User task", SPEC_SCHEMA)
    assert result == spec()
    assert "project manager" in client.generate.call_args.args[1]


@pytest.mark.sandbox
async def test_qa_requires_cases_and_uses_separate_workspace(components, settings, tmp_path):
    _, _, sandbox, _ = components
    settings.filesystem_root.mkdir()
    directory = tmp_path / "qa_source"
    directory.mkdir()
    (directory / "extension.py").write_text(SOURCE)
    isolated_test = """
    def test_edge_isolated_workspace(self):
        from pathlib import Path
        Path("qa_created.txt").write_text("test artifact")
        self.assertTrue(Path("qa_created.txt").is_file())
"""
    (directory / "tests.py").write_text(TESTS + isolated_test)
    result = await sandbox.execute(directory, artifact()["manifest"], "test", {})
    assert result["passed"] and result["tests_run"] == 4
    assert not (settings.filesystem_root / "qa_created.txt").exists()
    (directory / "tests.py").write_text("import unittest\nclass Empty(unittest.TestCase): pass")
    result = await sandbox.execute(directory, artifact()["manifest"], "test", {})
    assert not result["passed"] and not any(result["coverage"].values())


async def test_team_timeout_is_reported_and_bad_checkpoints_do_not_break_recovery(
    components, settings, monkeypatch
):
    _, extensions, _, _ = components
    settings.extension_team_timeout_seconds = 1

    async def stall(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(extensions.team.roles, "ask", stall)
    with pytest.raises(TeamFailure) as failed:
        await extensions.create("Double", "tool", "doubler", "root", "", 1)
    report = extensions.team.report(failed.value.report["id"], 1)
    assert report["status"] == "failed" and "час" in report["error"]
    directory = extensions.team.directory / ("a" * 32)
    directory.mkdir()
    (directory / "report.json").write_text("not JSON")
    await extensions.team.recover()
    assert len(extensions.team.reports(1)) == 1


async def test_restart_recovers_activation_commit_before_final_checkpoint(components):
    import hashlib
    from uuid import uuid4

    _, extensions, _, _ = components
    candidate = artifact()
    report = {
        "id": uuid4().hex,
        "extension_id": "doubler",
        "user_id": 1,
        "status": "approved",
        "stage": "completed",
        "candidate": candidate,
    }
    await extensions.team.save(report)
    digest = hashlib.sha256(candidate["source"].encode()).hexdigest()
    await extensions.team.recover({("doubler", 1): digest})
    assert extensions.team.report(report["id"], 1)["status"] == "active"


async def test_qa_rejects_syntactically_invalid_python_before_execution():
    from t_bot.extension_team import QA_SCHEMA

    client = type("Client", (), {})()
    client.generate = AsyncMock(return_value=json.dumps({"tests": ["**def bad(:**"], "notes": ""}))
    with pytest.raises(BotError, match="невалідний Python"):
        await RoleAgents(client).ask("QA", "fake", "Write tests", QA_SCHEMA)


async def test_build_commands_work_without_ollama_and_keep_user_boundary(
    components, settings, store
):
    from uuid import uuid4

    from t_bot.jobs import Jobs
    from t_bot.telegram import TelegramAdapter

    runtime, extensions, _, control = components
    report = {
        "id": uuid4().hex,
        "user_id": 1,
        "extension_id": "doubler",
        "status": "failed",
        "stage": "PM",
        "error": "Offline",
        "events": [],
    }
    await extensions.team.save(report)
    adapter = TelegramAdapter(
        settings, store, runtime.ollama, extensions, Jobs(settings, store, runtime), control
    )
    assert report["id"] in await adapter.command("/builds", 1, 5)
    assert "Offline" in await adapter.command("/build " + report["id"], 1, 5)
    settings.allowed_user_ids = set()
    with pytest.raises(BotError):
        await adapter.command("/build " + report["id"], 99, 5)


async def test_team_discovers_installed_coder_fallback(components, monkeypatch):
    _, extensions, _, _ = components
    research(extensions.team, monkeypatch)
    calls = []
    monkeypatch.setattr(
        extensions.team.roles.ollama,
        "models",
        AsyncMock(
            return_value=[
                "embeddinggemma:latest",
                "other-chat",
                "qwen3-coder:30b",
            ]
        ),
    )

    async def ask(role, model, context, schema):
        calls.append((role, model))
        if len(calls) == 1:
            raise BotError("Invalid response")
        if role == "Developer":
            return artifact()
        if "tests" in schema["properties"]:
            return {"tests": TESTS, "notes": "Tests"}
        if "decision" in schema["properties"]:
            return {"decision": "approve", "findings": []}
        if "queries" in schema["properties"]:
            return {"queries": ["Python docs"]}
        return spec()

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    result = await extensions.team.run(
        request="Double",
        extension_id="doubler",
        user_id=1,
        model="primary",
        developer_context="Implement",
        developer_schema={},
        validate=AsyncMock(side_effect=lambda candidate: candidate),
        execute_tests=AsyncMock(return_value={"passed": True, "tests_run": 3}),
    )
    assert calls[:2] == [("PM", "primary"), ("PM", "qwen3-coder:30b")]
    assert result["qa"]["passed"]


async def test_root_returns_team_result_without_second_model_call(components, monkeypatch):
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    runtime, _, _, _ = components
    monkeypatch.setattr(runtime.ollama, "require_model", AsyncMock())

    class CallingModel(BaseLlm):
        calls: int = 0

        async def generate_content_async(self, llm_request, stream=False):
            self.calls += 1
            assert self.calls == 1, "Team completion must not request a model summary"
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="create_extension",
                                args={
                                    "request": "Double",
                                    "kind": "tool",
                                    "extension_id": "doubler",
                                },
                            ),
                        )
                    ],
                )
            )

    model = CallingModel(model="fake")
    monkeypatch.setattr(runtime, "model", lambda name: model)

    async def create_extension(request: str, kind: str, extension_id: str) -> dict:
        return {
            "status": "active",
            "manifest": {"id": "doubler", "version": 1},
            "team_run_id": "a" * 32,
            "qa": {"tests_run": 3},
            "usage": "Double an integer",
        }

    monkeypatch.setattr(runtime, "create_extension", create_extension)
    result = await runtime.run(1, 5, 7, "Create doubler")
    assert "Створено doubler v1" in result and "QA: 3" in result
    assert model.calls == 1

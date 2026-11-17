import asyncio
from pathlib import Path

import pytest

from t_bot.config import Settings
from t_bot.control import RestartControl
from t_bot.extensions import Extensions
from t_bot.ollama import Ollama
from t_bot.runtime import AgentRuntime
from t_bot.sandbox import Sandbox
from t_bot.store import Store


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    config = Settings(
        _env_file=None,
        allowed_user_ids={1},
        telegram_bot_token="123:test",
        data_dir=tmp_path / "data",
        extensions_dir=tmp_path / "extensions",
        filesystem_root=tmp_path / "workspace",
        sandbox_timeout_seconds=15,
        restart_cooldown_seconds=1,
    )
    config.prepare_directories()
    return config


@pytest.fixture
def store(settings):
    state = Store(settings.data_dir / "state.sqlite3")
    yield state
    state.close()


@pytest.fixture
async def components(settings, store, monkeypatch, request):
    ollama = Ollama(settings.ollama_base_url)
    sandbox = Sandbox(settings)
    extensions = Extensions(settings, store, sandbox, ollama)
    control = RestartControl(settings, store, asyncio.Event())
    runtime = AgentRuntime(settings, store, ollama, extensions, control)
    if not request.node.get_closest_marker("live"):
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            extensions.team.webpages,
            "read",
            AsyncMock(
                return_value={
                    "url": "https://docs.python.org/3/",
                    "title": "Python",
                    "text": "Official Python documentation",
                    "truncated": False,
                }
            ),
        )
    if not request.node.get_closest_marker("live") and not request.node.get_closest_marker("team"):
        # Existing contract tests mock only Developer's generator. Keep other role outputs
        # deterministic; dedicated team tests exercise planning, feedback and QA behavior.
        import json

        from t_bot.extension_team import SPEC_SCHEMA

        original = extensions.team.roles.ask

        async def other_roles(role, model, context, schema):
            if role == "Developer":
                return await original(role, model, context, schema)
            if role == "Reviewer" or (role == "QA" and "decision" in schema.get("properties", {})):
                return {"decision": "approve", "findings": []}
            if role == "QA":
                candidate = json.loads(context)["implementation"]
                manifest = candidate["manifest"]
                source = """import unittest
import jsonschema
import extension

class ContractTests(unittest.TestCase):
    def test_happy_example(self):
        self.assertTrue(callable(getattr(extension, ENTRYPOINT)))
    def test_failure_schema(self):
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate([], INPUT_SCHEMA)
    def test_edge_contract(self):
        self.assertIsInstance(EXAMPLE, dict)
"""
                source = source.replace("ENTRYPOINT", repr(manifest["entrypoint"]))
                source = source.replace("INPUT_SCHEMA", repr(manifest["input_schema"]))
                source = source.replace("EXAMPLE", repr(manifest["example"]))
                return {
                    "tests": source,
                    "notes": "Deterministic role fixture for host contract tests.",
                }
            if "queries" in schema.get("properties", {}):
                return {"queries": ["Python official documentation"]}
            return {
                "source_checks": [
                    {
                        "conclusion": "Use documentation",
                        "url": "https://docs.python.org/3/",
                        "quote": "Official Python documentation",
                    }
                ],
                **{
                    name: "Objective" if name == "objective" else ["Acceptance"]
                    for name in SPEC_SCHEMA["required"]
                },
            }

        monkeypatch.setattr(extensions.team.roles, "ask", other_roles)
        monkeypatch.setattr(
            extensions.team.web,
            "search",
            lambda *args: {
                "results": [
                    {"title": "Python", "url": "https://docs.python.org/3/", "snippet": "Docs"}
                ]
            },
        )
    yield runtime, extensions, sandbox, control
    await runtime.close()
    await ollama.close()


@pytest.fixture
def pip_report():
    """Write a deterministic pip dry-run report in tests that replace package downloads."""
    import json

    def write(command, packages=None):
        if "--dry-run" not in command:
            return False
        index = command.index("--report")
        if packages is None:
            requirements = command[index + 2 :]
            if "--constraint" in requirements:
                requirements = requirements[2:]
            packages = [p if "==" in p else p + "==1.0" for p in requirements]
        Path(command[index + 1]).write_text(
            json.dumps(
                {
                    "version": "1",
                    "install": [
                        {
                            "metadata": {"name": p.split("==")[0], "version": p.split("==")[1]},
                            "download_info": {"url": "https://example.org/package.whl"},
                            "is_direct": False,
                        }
                        for p in packages
                    ],
                }
            )
        )
        return True

    return write

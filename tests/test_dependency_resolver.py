from unittest.mock import AsyncMock

import pytest

from t_bot.errors import BotError


async def test_resolver_adds_transitive_pins_and_checks_allowlist_before_install(
    components, settings, monkeypatch, pip_report
):
    _, _, sandbox, _ = components
    commands = []
    pins = ["matplotlib==3.8.0", "numpy==1.26.4", "kiwisolver==1.4.5"]

    async def run(command, **kwargs):
        commands.append(command)
        pip_report(command, pins)
        return b""

    monkeypatch.setattr("t_bot.sandbox.bounded_process", run)
    settings.allowed_dependencies = {"matplotlib==3.8.0", "numpy==1.26.4"}
    with pytest.raises(BotError, match="kiwisolver"):
        await sandbox.prepare_dependencies(["matplotlib"])
    assert len(commands) == 1 and "--dry-run" in commands[0]
    assert not sandbox.dependency_path(pins).exists()
    settings.allowed_dependencies = set(pins)
    resolved = await sandbox.prepare_dependencies(["matplotlib"])
    assert resolved == sorted(pins)
    assert "--no-deps" in commands[-1] and set(pins) <= set(commands[-1])
    before = len(commands)
    assert await sandbox.prepare_dependencies(resolved) == resolved
    assert len(commands) == before
    settings.allowed_dependencies = {"matplotlib==3.8.0"}
    with pytest.raises(BotError, match="ALLOWED_DEPENDENCIES"):
        await sandbox.prepare_dependencies(["matplotlib"])


async def test_resolver_failure_is_actionable_and_does_not_install(components, monkeypatch):
    _, _, sandbox, _ = components
    process = AsyncMock(
        side_effect=BotError("ERROR: No matching distribution found for kiwisolver==1.4.1")
    )
    monkeypatch.setattr("t_bot.sandbox.bounded_process", process)
    with pytest.raises(BotError, match="основні назви"):
        await sandbox.prepare_dependencies(["kiwisolver==1.4.1"])
    assert process.await_count == 1
    assert not list(sandbox.dependencies_dir.iterdir())


async def test_resolver_report_must_include_roots(components, monkeypatch, pip_report):
    _, _, sandbox, _ = components

    async def run(command, **kwargs):
        pip_report(command, ["other==1"])
        return b""

    monkeypatch.setattr("t_bot.sandbox.bounded_process", run)
    with pytest.raises(BotError, match="звіт"):
        await sandbox.prepare_dependencies(["matplotlib"])


async def test_resolver_cannot_replace_explicitly_pinned_version(
    components, monkeypatch, pip_report
):
    _, _, sandbox, _ = components

    async def run(command, **kwargs):
        pip_report(command, ["matplotlib==9.0"])
        return b""

    monkeypatch.setattr("t_bot.sandbox.bounded_process", run)
    with pytest.raises(BotError, match="звіт"):
        await sandbox.prepare_dependencies(["matplotlib==3.8.0"])

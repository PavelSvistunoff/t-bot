import socket
from pathlib import Path

import pytest

from t_bot.errors import BotError
from t_bot.manifest import Manifest
from t_bot.sandbox import Sandbox

pytestmark = pytest.mark.sandbox


def tool_manifest(**kwargs):
    return Manifest(
        id="probe_tool",
        kind="tool",
        version=1,
        description="Test tool",
        example_usage="Run the probe",
        **kwargs,
    )


async def run_code(settings, tmp_path, code, **kwargs):
    directory = tmp_path / "code"
    directory.mkdir(exist_ok=True)
    (directory / "extension.py").write_text(code)
    return await Sandbox(settings).execute(
        directory,
        tool_manifest(**kwargs).model_dump(),
        "invoke",
        {},
    )


async def test_probe_and_real_tool(settings, tmp_path):
    assert await Sandbox(settings).ready()
    result = await run_code(
        settings,
        tmp_path,
        '''def invoke(payload: dict) -> dict:
    """Return a result."""
    return {"answer": 42}
''',
    )
    assert result == {"answer": 42}


@pytest.mark.parametrize("permissions", [[], ["workspace"]])
async def test_secret_files_environment_network_and_fork_denied(
    settings, tmp_path, monkeypatch, permissions
):
    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "DO-NOT-LEAK")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        code = f'''import os, socket

def invoke(payload: dict) -> dict:
    """Probe denied capabilities."""
    result = {{"token": os.environ.get("TELEGRAM_BOT_TOKEN")}}
    try:
        open({str(secret)!r}).read()
        result["file"] = "allowed"
    except OSError:
        result["file"] = "denied"
    try:
        socket.create_connection(("127.0.0.1", {port}), timeout=1)
        result["network"] = "allowed"
    except OSError:
        result["network"] = "denied"
    try:
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        result["fork"] = "allowed"
    except OSError:
        result["fork"] = "denied"
    return result
'''
        result = await run_code(settings, tmp_path, code, permissions=permissions)
    assert result == {"token": None, "file": "denied", "network": "denied", "fork": "denied"}


async def test_timeout_terminates_process(settings, tmp_path):
    settings.sandbox_timeout_seconds = 1
    with pytest.raises(BotError):
        await run_code(
            settings,
            tmp_path,
            '''import time

def invoke(payload: dict) -> dict:
    """Exceed wall time without consuming CPU."""
    time.sleep(30)
    return {}
''',
        )


async def test_output_limit(settings, tmp_path):
    with pytest.raises(BotError):
        await run_code(
            settings,
            tmp_path,
            '''def invoke(payload: dict) -> dict:
    """Flood stdout."""
    print("x" * 200000)
    return {}
''',
        )


async def test_workspace_available_without_explicit_permission(settings, tmp_path):
    code = '''from pathlib import Path

def invoke(payload: dict) -> dict:
    """Write only to explicitly allowed persistent workspace."""
    try:
        Path("workspace/result.txt").write_text("temporary")
        return {"written": True}
    except OSError:
        return {"written": False}
'''
    assert await run_code(settings, tmp_path, code, permissions=[]) == {"written": True}
    assert await run_code(settings, tmp_path, code, permissions=["workspace"]) == {"written": True}
    assert (settings.filesystem_root / "result.txt").read_text() == "temporary"


async def test_memory_limit_is_enforced(settings, tmp_path):
    settings.sandbox_memory_mb = 8
    with pytest.raises(BotError):
        await run_code(
            settings,
            tmp_path,
            '''import time

def invoke(payload: dict) -> dict:
    """Keep the interpreter alive for RSS monitoring."""
    time.sleep(2)
    return {}
''',
        )


async def test_cancellation_terminates_the_child(settings, tmp_path, monkeypatch):
    import asyncio

    launch = asyncio.create_subprocess_exec
    children = []
    started = asyncio.Event()

    async def record_child(*args, **kwargs):
        process = await launch(*args, **kwargs)
        if "sandbox-exec" in args[0] or "bwrap" in args[0]:
            children.append(process)
            started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record_child)
    task = asyncio.create_task(
        run_code(
            settings,
            tmp_path,
            '''import time

def invoke(payload: dict) -> dict:
    """Wait until cancelled."""
    time.sleep(30)
    return {}
''',
        )
    )
    await asyncio.wait_for(started.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert children and all(child.returncode is not None for child in children)


@pytest.mark.parametrize("framework_exists", [True, False])
def test_macos_framework_launcher_has_exact_execution_permissions(
    settings, tmp_path, monkeypatch, framework_exists
):
    import json
    import sys

    from t_bot import sandbox

    framework_root = tmp_path / "Python.framework/Versions/3.12"
    interpreter = framework_root / "Resources/Python.app/Contents/MacOS/Python"
    if framework_exists:
        interpreter.parent.mkdir(parents=True)
        interpreter.touch()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "base_prefix", str(framework_root))
    monkeypatch.setattr(sandbox.sysconfig, "get_config_var", lambda name: "Python")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    command = Sandbox(settings).command(tmp_path, tmp_path / "dependencies")
    profile = command[2]
    execution_rule = next(
        rule for rule in profile.split("(allow ") if rule.startswith("process-exec ")
    )
    if framework_exists:
        assert f"(literal {json.dumps(str(interpreter))})" in execution_rule
    else:
        assert str(interpreter) not in execution_rule
    assert f"(literal {json.dumps(str(Path(sys.executable).resolve()))})" in execution_rule
    assert "(allow process-fork)" not in profile
    assert '(subpath "/")' not in profile


async def test_workspace_environment_and_files_persist_between_invocations(settings, tmp_path):
    settings.filesystem_root.mkdir()
    (settings.filesystem_root / "input.txt").write_text("user input")
    code = '''import os
from pathlib import Path

def invoke(payload: dict) -> dict:
    """Read host workspace and persist output across isolated invocations."""
    root = Path(os.environ["FILESYSTEM_ROOT"])
    output = root / "output.txt"
    previous = output.read_text() if output.exists() else ""
    output.write_text(previous + root.joinpath("input.txt").read_text())
    return {"text": Path("workspace/output.txt").read_text(), "root": str(root)}
'''
    first = await run_code(settings, tmp_path, code, permissions=[])
    second = await run_code(settings, tmp_path, code, permissions=[])
    assert first == {"text": "user input", "root": str(settings.filesystem_root)}
    assert second["text"] == "user inputuser input"
    assert (settings.filesystem_root / "output.txt").read_text() == second["text"]


async def test_workspace_hides_secrets_and_external_links(settings, tmp_path):
    settings.filesystem_root.mkdir()
    nested = settings.filesystem_root / "nested"
    nested.mkdir()
    (nested / ".env.production").write_text("SECRET")
    external = tmp_path / "external.txt"
    external.write_text("PRIVATE")
    (nested / "external-link").symlink_to(external)
    code = '''from pathlib import Path

def invoke(payload: dict) -> dict:
    """Check private files cannot be read through the workspace."""
    result = {}
    for name in [".env.production", "external-link"]:
        try:
            result[name] = Path("workspace/nested", name).read_text()
        except OSError:
            result[name] = "denied"
    return result
'''
    assert await run_code(settings, tmp_path, code, permissions=["workspace"]) == {
        ".env.production": "denied",
        "external-link": "denied",
    }


async def test_workspace_cannot_include_bot_state(settings, tmp_path):
    settings.filesystem_root = tmp_path
    with pytest.raises(BotError, match="FILESYSTEM_ROOT"):
        await run_code(settings, tmp_path, "", permissions=["workspace"])


@pytest.mark.parametrize("permissions", [[], ["workspace"]])
async def test_absolute_workspace_path_is_available_by_default(settings, tmp_path, permissions):
    settings.filesystem_root.mkdir()
    target = settings.filesystem_root / "existing.txt"
    target.write_text("visible")
    code = f'''import os
from pathlib import Path

def invoke(payload: dict) -> dict:
    """Probe absolute workspace access and environment capability."""
    try:
        text = Path({str(target)!r}).read_text()
    except OSError:
        text = "denied"
    return {{"text": text, "environment": os.environ.get("FILESYSTEM_ROOT")}}
'''
    result = await run_code(settings, tmp_path, code, permissions=permissions)
    assert result == {
        "text": "visible",
        "environment": str(settings.filesystem_root),
    }

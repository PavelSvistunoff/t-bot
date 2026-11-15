import socket

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


async def test_secret_files_environment_network_and_fork_denied(settings, tmp_path, monkeypatch):
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
        result = await run_code(settings, tmp_path, code)
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


async def test_workspace_requires_permission(settings, tmp_path):
    code = '''from pathlib import Path

def invoke(payload: dict) -> dict:
    """Write only to explicitly allowed temporary workspace."""
    try:
        Path("workspace/result.txt").write_text("temporary")
        return {"written": True}
    except OSError:
        return {"written": False}
'''
    assert await run_code(settings, tmp_path, code) == {"written": False}
    assert await run_code(settings, tmp_path, code, permissions=["workspace"]) == {"written": True}


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

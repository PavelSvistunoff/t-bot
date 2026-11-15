import asyncio
import os
import shlex

import pytest

from t_bot.console import Console
from t_bot.errors import BotError
from t_bot.filesystem import FileSystem

pytestmark = pytest.mark.sandbox


async def test_command_pipes_redirection_stderr_and_exit_code(components, settings):
    runtime, _, _, _ = components
    result = await runtime.basic_tools.console.run(
        "printf 'beta\\nalpha\\n' | sort > words.txt; cat words.txt; printf issue >&2; exit 7",
    )
    assert result["exit_code"] == 7
    assert result["stdout"] == "alpha\nbeta\n"
    assert result["stderr"] == "issue"
    assert not result["timed_out"]
    assert (settings.filesystem_root / "words.txt").read_text() == "alpha\nbeta\n"


async def test_python_path_and_subdirectory(components, settings):
    runtime, _, _, _ = components
    (settings.filesystem_root / "nested").mkdir(parents=True)
    result = await runtime.basic_tools.console.run(
        "python -c 'import os; print(os.getcwd())'", "nested"
    )
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == str(settings.filesystem_root / "nested")


async def test_output_is_bounded_and_does_not_deadlock(components, settings):
    runtime, _, _, _ = components
    settings.console_max_output_bytes = 256
    result = await runtime.basic_tools.console.run(
        'python -c \'import sys; print("x"*100000); print("e"*100000,file=sys.stderr)\'',
    )
    assert result["exit_code"] == 0
    assert len(result["stdout"].encode()) == 256
    assert len(result["stderr"].encode()) == 256
    assert result["stdout_truncated"] and result["stderr_truncated"]


async def test_timeout_kills_command_and_child_process(components):
    runtime, _, _, _ = components
    result = await asyncio.wait_for(
        runtime.basic_tools.console.run("sleep 30 & wait", timeout_seconds=1),
        5,
    )
    assert result["timed_out"]
    assert result["exit_code"] != 0


async def test_cancellation_cleans_up_process(components, monkeypatch):
    runtime, _, _, _ = components
    launch = asyncio.create_subprocess_exec
    started = asyncio.Event()
    processes = []

    async def record_process(*args, **kwargs):
        process = await launch(*args, **kwargs)
        if "sandbox-exec" in args[0] or "bwrap" in args[0]:
            processes.append(process)
            started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record_process)
    task = asyncio.create_task(runtime.basic_tools.console.run("sleep 30 & wait"))
    await asyncio.wait_for(started.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert processes and processes[0].returncode is not None


async def test_secrets_are_not_inherited_or_read(components, settings, tmp_path, monkeypatch):
    runtime, _, _, _ = components
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "private-token")
    monkeypatch.setenv("T_BOT_SUPERVISED", "1")
    secret = tmp_path / ".env"
    secret.write_text("private-token")
    filesystem = FileSystem(tmp_path, 262144, ())
    console = Console(settings, filesystem)
    console.project = tmp_path
    code = (
        "import os; print(os.environ.get('TELEGRAM_BOT_TOKEN')); "
        "print(os.environ.get('T_BOT_SUPERVISED'))"
    )
    result = await console.run("python -c " + shlex.quote(code))
    assert result["stdout"].strip() == "None\nNone"
    result = await console.run("cat .env")
    assert result["exit_code"] != 0
    assert "private-token" not in result["stdout"]


@pytest.mark.parametrize("attack", ["direct", "hardlink", "symlink"])
async def test_console_cannot_modify_system_state(components, settings, store, attack):
    runtime, _, _, _ = components
    state = settings.data_dir / "state.sqlite3"
    escaped = shlex.quote(str(state))
    if attack == "direct":
        command = f"printf corrupt > {escaped}"
    elif attack == "hardlink":
        command = f"ln {escaped} alias.db && printf corrupt > alias.db"
    else:
        command = f"ln -s {escaped} alias.db; printf corrupt > alias.db"
    result = await runtime.basic_tools.console.run(command)
    assert result["exit_code"] != 0
    assert state.read_bytes().startswith(b"SQLite format 3")
    assert store.get("anything") is None


async def test_command_cannot_signal_bot_process(components):
    runtime, _, _, _ = components
    result = await runtime.basic_tools.console.run(f"kill -0 {os.getpid()}")
    assert result["exit_code"] != 0


@pytest.mark.parametrize(
    "command,cwd,timeout_seconds",
    [("", ".", 0), ("pwd", "..", 0), ("pwd", ".", -1), ("pwd", ".", 999)],
)
async def test_invalid_command_arguments_are_rejected(components, command, cwd, timeout_seconds):
    runtime, _, _, _ = components
    with pytest.raises(BotError):
        await runtime.basic_tools.console.run(command, cwd, timeout_seconds)


async def test_timeout_with_unbounded_output_does_not_block_cleanup(components):
    runtime, _, _, _ = components
    result = await asyncio.wait_for(
        runtime.basic_tools.console.run("yes output", timeout_seconds=1),
        5,
    )
    assert result["timed_out"]
    assert result["stdout_truncated"]

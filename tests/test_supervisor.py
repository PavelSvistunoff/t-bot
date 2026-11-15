import asyncio
import sys

from t_bot.cli import backup
from t_bot.store import Store
from t_bot.supervisor import supervise


async def test_supervisor_restarts_a_real_child_process(settings, monkeypatch, tmp_path):
    counter = tmp_path / "starts.txt"
    script = tmp_path / "worker.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        f"p = Path({str(counter)!r})\n"
        "count = int(p.read_text()) + 1 if p.exists() else 1\n"
        "p.write_text(str(count))\n"
        "sys.exit(75 if count == 1 else 0)\n"
    )
    launch = asyncio.create_subprocess_exec

    async def child_process(*args, **kwargs):
        return await launch(sys.executable, str(script), **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", child_process)
    assert await asyncio.wait_for(supervise(settings), 8) == 0
    assert counter.read_text() == "2"


async def test_backup_contains_state_sessions_and_extensions(settings, store, components, tmp_path):
    runtime, _, _, _ = components
    store.set("model:root", "other-model")
    await runtime.sessions.create_session(app_name="t_bot", user_id="1", session_id="saved")
    backup_path = tmp_path / "snapshot"
    backup(settings, backup_path)
    copied = Store(backup_path / "state.sqlite3")
    try:
        assert copied.get("model:root") == "other-model"
        assert (backup_path / "sessions.sqlite3").is_file()
        assert (backup_path / "extensions").is_dir()
    finally:
        copied.close()

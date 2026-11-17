from pathlib import Path

import pytest

from t_bot.config import Settings
from t_bot.errors import BotError
from t_bot.factory_reset import factory_reset
from t_bot.supervisor import process_lock


def configuration(project: Path, **changes) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=project / "data",
        extensions_dir=project / "data/extensions",
        filesystem_root=project / "workspace",
        sqlite_database_dir=project / "data/databases",
        **changes,
    )


def write(path: Path, content="keep"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_reset_removes_state_extensions_files_and_caches_preserving_installation(tmp_path):
    settings = configuration(tmp_path)
    removed = [
        write(tmp_path / name)
        for name in (
            "data/state.sqlite3",
            "data/sessions.sqlite3-wal",
            "data/team_runs/run/report.json",
            "data/extensions/tool/1/extension.py",
            "data/dependencies/pillow/PIL.py",
            "data/databases/user.sqlite3",
            "workspace/chart.png",
            "backups/old/state.sqlite3",
            ".cache/uv/wheel",
            ".pytest_cache/state",
            "src/t_bot/__pycache__/old.pyc",
            "tests/.ruff_cache/state",
        )
    ]
    kept = [
        write(tmp_path / name)
        for name in (
            ".env",
            "src/t_bot/main.py",
            "tests/test_app.py",
            ".git/config",
            ".venv/bin/python",
            "README.md",
            "factory-reset",
        )
    ]
    factory_reset(tmp_path, settings)
    assert all(not path.exists() for path in removed)
    assert all(path.read_text() == "keep" for path in kept)
    assert (settings.data_dir / "worker.lock").exists()
    # Repeated reset remains safe after state has already been removed.
    factory_reset(tmp_path, settings)
    assert all(path.exists() for path in kept)


def test_dry_run_does_not_remove_files_or_create_lock_files(tmp_path):
    settings = configuration(tmp_path)
    state = write(settings.data_dir / "state.sqlite3")
    assert settings.data_dir in factory_reset(tmp_path, settings, dry_run=True)
    assert state.exists()
    assert not (settings.data_dir / "worker.lock").exists()


@pytest.mark.parametrize("lock_name", ["supervisor.lock", "worker.lock"])
def test_running_bot_prevents_all_deletion(tmp_path, lock_name):
    settings = configuration(tmp_path)
    state = write(settings.data_dir / "state.sqlite3")
    cache = write(tmp_path / ".cache/uv/data")
    with process_lock(settings.data_dir / lock_name), pytest.raises(BotError):
        factory_reset(tmp_path, settings)
    assert state.exists() and cache.exists()


@pytest.mark.parametrize("destination", [".", "src", ".venv", "src/generated", "../outside"])
def test_unsafe_configuration_fails_before_any_deletion(tmp_path, destination):
    settings = configuration(tmp_path)
    state = write(settings.data_dir / "state.sqlite3")
    settings.filesystem_root = (tmp_path / destination).resolve()
    with pytest.raises(BotError):
        factory_reset(tmp_path, settings)
    assert state.exists()


def test_symlink_target_is_rejected_and_nested_symlink_never_followed(tmp_path):
    settings = configuration(tmp_path)
    outside = write(tmp_path.parent / f"{tmp_path.name}-outside" / "secret.txt")
    (tmp_path / "workspace").symlink_to(outside.parent, target_is_directory=True)
    with pytest.raises(BotError):
        factory_reset(tmp_path, settings)
    assert outside.read_text() == "keep"
    (tmp_path / "workspace").unlink()
    write(tmp_path / "workspace/local.txt")
    (tmp_path / "workspace/link").symlink_to(outside.parent, target_is_directory=True)
    factory_reset(tmp_path, settings)
    assert outside.read_text() == "keep"


def test_custom_storage_paths_are_removed(tmp_path):
    settings = configuration(tmp_path)
    settings.data_dir = tmp_path / "local/state"
    settings.extensions_dir = tmp_path / "local/plugins"
    settings.filesystem_root = tmp_path / "local/files"
    settings.sqlite_database_dir = tmp_path / "local/sqlite"
    files = [
        write(path / "generated")
        for path in (
            settings.data_dir,
            settings.extensions_dir,
            settings.filesystem_root,
            settings.sqlite_database_dir,
        )
    ]
    factory_reset(tmp_path, settings)
    assert not any(path.exists() for path in files)


def test_uv_managed_interpreter_survives_cache_cleanup(tmp_path, monkeypatch):
    settings = configuration(tmp_path)
    interpreter = write(tmp_path / ".cache/python/runtime/bin/python")
    wheel = write(tmp_path / ".cache/uv/wheel")
    monkeypatch.setattr("t_bot.factory_reset.sys.base_prefix", str(interpreter.parent.parent))
    factory_reset(tmp_path, settings)
    assert interpreter.read_text() == "keep"
    assert not wheel.exists()

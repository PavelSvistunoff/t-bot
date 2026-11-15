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
async def components(settings, store):
    ollama = Ollama(settings.ollama_base_url)
    sandbox = Sandbox(settings)
    extensions = Extensions(settings, store, sandbox, ollama)
    control = RestartControl(settings, store, asyncio.Event())
    runtime = AgentRuntime(settings, store, ollama, extensions, control)
    yield runtime, extensions, sandbox, control
    await runtime.close()
    await ollama.close()

from t_bot.application import run_worker
from t_bot.ollama import Ollama
from t_bot.store import Store
from t_bot.supervisor import RESTART_EXIT_CODE
from t_bot.telegram import TelegramAdapter


async def test_worker_restart_recovers_pending_notification(settings, monkeypatch):
    monkeypatch.setenv("T_BOT_SUPERVISED", "1")
    starts = 0

    async def start(adapter):
        nonlocal starts
        starts += 1
        if starts == 1:
            adapter.restart.request(1, 5, "integration")
        else:
            adapter.restart.stop.set()

    async def flush(adapter):
        return None

    async def require_model(client, model):
        return None

    monkeypatch.setattr(TelegramAdapter, "start", start)
    monkeypatch.setattr(TelegramAdapter, "flush", flush)
    monkeypatch.setattr(Ollama, "require_model", require_model)
    assert await run_worker(settings) == RESTART_EXIT_CODE
    store = Store(settings.data_dir / "state.sqlite3")
    assert store.get("restart_pending")["chat_id"] == 5
    store.set("model:root", "persisted-model")
    store.close()
    assert await run_worker(settings) == 0
    store = Store(settings.data_dir / "state.sqlite3")
    try:
        assert store.get("restart_pending") is None
        assert store.get("model:root") == "persisted-model"
        assert any("Перезапуск завершено" in row["text"] for row in store.pending_messages())
    finally:
        store.close()

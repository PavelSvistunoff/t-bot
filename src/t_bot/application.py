import asyncio
import os
import signal
import time

from t_bot.config import Settings
from t_bot.control import RestartControl
from t_bot.errors import BotError
from t_bot.extensions import Extensions
from t_bot.jobs import Jobs
from t_bot.ollama import Ollama
from t_bot.runtime import AgentRuntime
from t_bot.sandbox import Sandbox
from t_bot.store import Store
from t_bot.supervisor import RESTART_EXIT_CODE, process_lock
from t_bot.telegram import TelegramAdapter


async def run_worker(settings: Settings) -> int:
    settings.require_telegram()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    with process_lock(settings.data_dir / "worker.lock"):
        store = Store(settings.data_dir / "state.sqlite3")
        ollama = Ollama(settings.ollama_base_url, settings.model_timeout_seconds)
        sandbox = Sandbox(settings)
        extensions = Extensions(settings, store, sandbox, ollama)
        pending_restart = store.get("restart_pending")
        restart = RestartControl(settings, store, stop)
        runtime = AgentRuntime(settings, store, ollama, extensions, restart)
        jobs = Jobs(settings, store, runtime)
        adapter = TelegramAdapter(settings, store, ollama, extensions, jobs, restart)
        runtime.screenshot_sender = adapter.send_screenshot
        try:
            jobs.start()
            await adapter.start()
            # A missing model leaves commands available so the operator can select another one.
            try:
                await ollama.require_model(store.get("model:root", settings.default_model))
                runtime.build_agent()
                await runtime.sessions.get_session(
                    app_name="t_bot",
                    user_id="system",
                    session_id="readiness",
                )
                health = "готовий"
            except (BotError, ValueError):
                health = "потребує налаштування моделі: /models, /model"
            store.set(
                "worker_state",
                {
                    "ready": health == "готовий",
                    "pid": os.getpid(),
                    "started": time.time(),
                },
            )
            if pending_restart:
                store.notify(pending_restart["chat_id"], f"Перезапуск завершено. Бот {health}.")
                if store.get("restart_pending") == pending_restart:
                    store.set("restart_pending", None)
            await stop.wait()
        finally:
            jobs.accepting = False
            await adapter.stop_polling()
            await jobs.shutdown()
            await adapter.close()
            await runtime.close()
            await ollama.close()
            store.set("worker_state", {"ready": False, "stopped": time.time()})
            store.close()
        return RESTART_EXIT_CODE if restart.requested else 0

import asyncio
import os
import time

from t_bot.config import Settings
from t_bot.errors import BotError
from t_bot.store import Store


class RestartControl:
    def __init__(self, settings: Settings, store: Store, stop: asyncio.Event):
        self.settings = settings
        self.store = store
        self.stop = stop
        self.requested = False

    def request(self, user_id: int, chat_id: int, reason: str) -> str:
        self.settings.authorize(user_id)
        if not os.environ.get("T_BOT_SUPERVISED"):
            raise BotError("Перезапуск потребує запуску через t-bot run (supervisor).")
        if self.requested:
            return "Перезапуск уже заплановано."
        now = time.time()
        if now - self.store.get("last_restart", 0) < self.settings.restart_cooldown_seconds:
            raise BotError("Перезапуск тимчасово обмежено; зачекайте хвилину.")
        with self.store.transaction():
            self.store.set("last_restart", now)
            self.store.set(
                "restart_pending", {"user_id": user_id, "chat_id": chat_id, "requested_at": now}
            )
            self.store.audit(user_id, "restart", reason[:500])
            self.store.notify(chat_id, "Перезапускаю бота зі збереженням сесії.")
        self.requested = True
        self.stop.set()
        return "Перезапуск заплановано. Після запуску надійде повідомлення про готовність."

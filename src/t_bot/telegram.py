import asyncio
import json
import logging

from telegram import Bot, InputFile, Message, Update
from telegram.error import BadRequest, NetworkError, RetryAfter, TelegramError
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from t_bot.config import Settings
from t_bot.control import RestartControl
from t_bot.current_time import current_datetime, format_datetime
from t_bot.errors import BotError
from t_bot.extensions import Extensions
from t_bot.help_guide import render_help
from t_bot.jobs import Jobs
from t_bot.ollama import Ollama
from t_bot.screenshot import ScreenCapture, ScreenImage
from t_bot.store import Store
from t_bot.tool_names import BASIC_TOOL_DESCRIPTIONS

logger = logging.getLogger(__name__)


def chunks(text: str, limit: int = 3500) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    size = 0
    for character in text:
        width = len(character.encode("utf-16-le")) // 2
        if size + width > limit:
            result.append("".join(current))
            current, size = [], 0
        current.append(character)
        size += width
    if current:
        result.append("".join(current))
    return result or ["(порожня відповідь)"]


async def send_text(bot: Bot, chat_id: int, text: str) -> None:
    for part in chunks(text):
        await bot.send_message(chat_id=chat_id, text=part)


def normalize_request(text: str) -> str:
    return " ".join(text.casefold().split()).strip(' ?!.,…«»"“”')


def is_screenshot_request(text: str) -> bool:
    normalized = normalize_request(text)
    if normalized in {"що зараз на екрані", "що на екрані"}:
        return True
    return bool(normalized) and normalized.split()[0].split("@")[0] == "/screenshot"


def is_time_request(text: str) -> bool:
    normalized = normalize_request(text)
    if normalized in {"котра година", "скока время"}:
        return True
    return bool(normalized) and normalized.split()[0].split("@")[0] == "/time"


class TelegramAdapter:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        ollama: Ollama,
        extensions: Extensions,
        jobs: Jobs,
        restart: RestartControl,
    ):
        self.settings = settings
        self.store = store
        self.ollama = ollama
        self.extensions = extensions
        self.jobs = jobs
        self.restart = restart
        self.application = (
            Application.builder()
            .token(
                settings.telegram_bot_token.get_secret_value(),
            )
            .build()
        )
        self.application.add_handler(MessageHandler(filters.TEXT, self.handle))
        self.application.add_error_handler(self.on_error)
        self.sender: asyncio.Task | None = None
        self.sending = True
        self.screen_capture = ScreenCapture(
            settings.screenshot_timeout_seconds,
            settings.screenshot_max_bytes,
        )
        self.screenshot_lock = asyncio.Lock()

    async def _send_screen_image(self, image: ScreenImage, chat_id: int) -> tuple[Message, str]:
        bot = self.application.bot
        if image.fits_telegram_photo:
            try:
                message = await bot.send_photo(
                    chat_id=chat_id,
                    photo=InputFile(image.data, filename="screen.png"),
                    caption="Що зараз на екрані",
                    write_timeout=30,
                    read_timeout=30,
                )
                return message, "photo"
            except BadRequest as exc:
                # Retry as a file only for definite image rejection, not unknown delivery failures.
                if not any(word in str(exc).casefold() for word in ("photo", "image", "dimension")):
                    raise
        message = await bot.send_document(
            chat_id=chat_id,
            document=InputFile(image.data, filename="screen.png"),
            caption="Скріншот екрана",
            write_timeout=30,
            read_timeout=30,
        )
        return message, "document"

    async def send_screenshot(self, user_id: int, chat_id: int) -> dict:
        self.settings.authorize(user_id)
        async with self.screenshot_lock:
            self.store.audit(user_id, "screenshot.capture", str(chat_id))
            image = await self.screen_capture.capture()
            try:
                message, delivery = await self._send_screen_image(image, chat_id)
            except TelegramError as exc:
                raise BotError(
                    "Не вдалося підтвердити надсилання скріншота. Перевірте чат і з’єднання; "
                    "запит автоматично не повторюється."
                ) from exc
            self.store.audit(user_id, "screenshot.sent", str(chat_id))
            return {
                "status": "sent",
                "chat_id": chat_id,
                "message_id": message.message_id,
                "delivery": delivery,
                "width": image.width,
                "height": image.height,
            }

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.effective_chat or not update.effective_message:
            return
        user_id, chat_id = update.effective_user.id, update.effective_chat.id
        try:
            self.settings.authorize(user_id)
            text = update.effective_message.text or ""
            screen_request = is_screenshot_request(text)
            time_request = is_time_request(text)
            help_request = normalize_request(text) in {"допомога", "help"} or (
                bool(text.split())
                and text.split()[0].split("@")[0].casefold() in {"/help", "/start"}
            )
            if screen_request or time_request or help_request:
                if text.lstrip().startswith("/"):
                    target = text.split()[0].partition("@")[2]
                    if target and target.casefold() != context.bot.username.casefold():
                        return
                if self.store.claim_update(update.update_id):
                    if help_request:
                        self.store.notify(chat_id, await self.command("/help", user_id, chat_id))
                    elif time_request:
                        self.store.notify(chat_id, await self.command("/time", user_id, chat_id))
                    else:
                        await self.send_screenshot(user_id, chat_id)
                return
            if not text.startswith("/"):
                self.jobs.submit(update.update_id, user_id, chat_id, text)
                return
            if not self.store.claim_update(update.update_id):
                return
            response = await self.command(text, user_id, chat_id)
            self.store.notify(chat_id, response)
        except (BotError, ValueError) as exc:
            message = str(exc) if isinstance(exc, BotError) else "Некоректні аргументи. /help"
            if self.settings.allows(user_id):
                self.store.notify(chat_id, message)
            else:
                await send_text(context.bot, chat_id, "Доступ заборонено.")

    async def command(self, text: str, user_id: int, chat_id: int) -> str:
        self.settings.authorize(user_id)
        parts = text.split()
        command = parts[0].split("@")[0].lower()
        args = parts[1:]
        if command in {"/start", "/help"}:
            self.store.audit(user_id, "help.read", "")
            return render_help(r.manifest for r in self.extensions.records())
        if command == "/time":
            self.store.audit(user_id, "clock.read", self.settings.bot_timezone)
            return format_datetime(current_datetime(self.settings.bot_timezone))
        if command == "/models":
            return "\n".join(await self.ollama.models()) or "Моделей немає."
        if command == "/model":
            if not args:
                return "Поточна модель: " + self.store.get(
                    "model:root", self.settings.default_model
                )
            model, agent_id = args[0], args[1] if len(args) > 1 else "root"
            await self.ollama.require_model(model)
            if agent_id != "root" and self.extensions.get(agent_id).manifest.kind == "tool":
                raise BotError("Інструмент не має моделі.")
            self.store.set("model:" + agent_id, model)
            self.store.audit(user_id, "model", f"{agent_id}:{model}")
            return f"{agent_id}: {model}; застосовується з наступного повідомлення."
        if command in {"/agents", "/tools", "/extensions"}:
            records = self.extensions.records(active_only=command != "/extensions")
            if command == "/tools":
                records = [r for r in records if r.manifest.kind == "tool"]
            elif command == "/agents":
                records = [r for r in records if r.manifest.kind != "tool"]
            listing = (
                "\n".join(
                    f"{r.manifest.id} v{r.manifest.version} "
                    f"[{'active' if r.active else 'inactive'}] ({r.manifest.kind}, "
                    f"parent={r.manifest.parent}): {r.manifest.description}"
                    for r in records
                )
                or "Розширень немає. Попросіть агента створити потрібне."
            )
            if command == "/tools":
                builtins = "\n".join(
                    f"{name}: {description}"
                    for name, description in BASIC_TOOL_DESCRIPTIONS.items()
                )
                return "Базові інструменти:\n" + builtins + "\n\nРозширення:\n" + listing
            return listing
        if command == "/source":
            if len(args) not in {1, 2}:
                raise BotError("/source ID [VERSION]")
            record = self.extensions.get(args[0], int(args[1]) if len(args) == 2 else None)
            source = (self.extensions.directory(record.manifest) / "extension.py").read_text()
            return record.manifest.model_dump_json(indent=2) + "\n\n" + source
        if command == "/extension":
            if len(args) not in {2, 3}:
                raise BotError("/extension enable|disable|delete|rollback ID [VERSION]")
            return await self.extensions.manage(
                args[1],
                args[0],  # type: ignore[arg-type]
                user_id,
                int(args[2]) if len(args) > 2 else None,
            )
        if command == "/cancel":
            if len(args) != 1:
                raise BotError("/cancel TASK_ID")
            return self.jobs.cancel(int(args[0]), user_id, chat_id)
        if command == "/new":
            if self.store.has_pending_jobs(user_id, chat_id):
                raise BotError("Спершу завершіть або скасуйте поточні задачі.")
            key = f"session_generation:{user_id}:{chat_id}"
            self.store.set(key, self.store.get(key, 0) + 1)
            return "Розпочато нову сесію."
        if command == "/restart":
            return self.restart.request(user_id, chat_id, "Telegram /restart")
        if command == "/status":
            try:
                models = await self.ollama.models()
                health = "Ollama: доступна"
            except BotError:
                models, health = [], "Ollama: недоступна"
            model = self.store.get("model:root", self.settings.default_model)
            return (
                f"{health}\nМодель: {model} ({'встановлена' if model in models else 'відсутня'})\n"
                f"Черга: {self.jobs.queue.qsize()}\n"
                + json.dumps(self.store.jobs(user_id, chat_id), ensure_ascii=False, indent=2)
            )
        return "Невідома команда. /help"

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("telegram.handler_failed", extra={"error_type": type(context.error).__name__})

    async def flush(self) -> None:
        for message in self.store.pending_messages():
            try:
                await send_text(self.application.bot, message["chat_id"], message["text"])
                self.store.mark_sent(message["id"])
            except RetryAfter as exc:
                seconds = exc.retry_after
                delay = seconds.total_seconds() if hasattr(seconds, "total_seconds") else seconds
                await asyncio.sleep(min(float(delay), 30))
                break
            except NetworkError:
                break  # Keep messages in the durable outbox for a later delivery attempt.
            except TelegramError:
                self.store.mark_sent(message["id"])
                logger.warning("telegram.message_undeliverable")

    async def deliver(self) -> None:
        while self.sending:
            await self.flush()
            await asyncio.sleep(0.5)

    async def start(self) -> None:
        await self.application.initialize()  # getMe verifies the configured token.
        await self.application.start()
        assert self.application.updater is not None
        await self.application.updater.start_polling(drop_pending_updates=False)
        self.sender = asyncio.create_task(self.deliver(), name="telegram-outbox")

    async def stop_polling(self) -> None:
        if self.application.updater and self.application.updater.running:
            await self.application.updater.stop()

    async def close(self) -> None:
        self.sending = False
        if self.sender:
            self.sender.cancel()
            await asyncio.gather(self.sender, return_exceptions=True)
        try:
            async with asyncio.timeout(5):
                await self.flush()
        except TimeoutError:
            pass  # Undelivered messages remain durable for next startup.
        if self.application.running:
            await self.application.stop()
        await self.application.shutdown()

"""Scoped, safe user-facing progress for extension development."""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

logger = logging.getLogger(__name__)
HEARTBEAT_SECONDS = 30
T = TypeVar("T")
PHASES = {
    "supervisor": "Supervisor розбирає конфлікт між QA та Reviewer",
    "reviewer_rework": "Reviewer переглядає зауваження за доказами Supervisor",
    "preparation": "Підготовка запиту та залежностей",
    "pipeline": "Команда розпочала розробку",
    "pm_draft": "PM складає технічне завдання",
    "analyst_plan": "Аналітик планує дослідження",
    "research": "Аналітик шукає документацію в інтернеті",
    "analyst_revision": "Аналітик уточнює технічне завдання",
    "pm_revision": "PM виправляє суперечності технічного завдання",
    "pm_final": "PM затверджує фінальне технічне завдання",
    "developer": "Developer пише або доопрацьовує код",
    "validation": "Перевірка коду й успішного прикладу в sandbox",
    "qa_write": "QA пише тести happy / failure",
    "qa_execute": "Виконання QA-тестів у sandbox",
    "qa_triage": "QA аналізує результати тестів",
    "reviewer": "Code reviewer перевіряє код і тести",
    "rework": "Повернення на доопрацювання",
    "activation": "Активація перевіреного розширення",
}


@dataclass(frozen=True)
class DevelopmentEvent:
    phase: str
    status: Literal["started", "completed", "failed", "running", "rework", "cancelled"]
    build_id: str = ""
    model: str = ""
    attempt: int = 0
    attempts: int = 0
    detail: str = ""


ProgressCallback = Callable[[DevelopmentEvent], None]


def progress_text(event: DevelopmentEvent, extension_id: str, job_id: int) -> str:
    symbols = {
        "started": "▶",
        "completed": "✓",
        "failed": "✗",
        "running": "⏳",
        "rework": "↩",
        "cancelled": "■",
    }
    status = {
        "started": "початок",
        "completed": "завершено",
        "failed": "не вдалося",
        "running": "триває",
        "rework": "потрібні правки",
        "cancelled": "скасовано",
    }
    label = extension_id if re.fullmatch(r"[a-z][a-z0-9_]{0,47}", extension_id) else "Розширення"
    lines = [
        f"{label} · задача {job_id}",
        f"{symbols[event.status]} {PHASES.get(event.phase, event.phase)} — {status[event.status]}.",
    ]
    if event.attempt:
        lines.append(f"Спроба {event.attempt}/{event.attempts}.")
    if event.model:
        lines.append("Модель: " + " ".join(event.model.split())[:100])
    if event.detail:
        lines.append(event.detail)
    if event.build_id and event.phase == "pipeline":
        lines.append("Звіт: /build " + event.build_id)
    return "\n".join(lines)


class ProgressReporter:
    def __init__(self, callback: ProgressCallback | None = None):
        self.callback = callback

    def emit(self, event: DevelopmentEvent) -> None:
        if self.callback is not None:
            try:
                self.callback(event)
            except Exception as exc:
                logger.warning(
                    "development_progress.delivery_failed",
                    extra={
                        "error_type": type(exc).__name__,
                    },
                )

    async def wait(self, operation: Awaitable[T], event: DevelopmentEvent) -> T:
        """Send heartbeats while a role runs; cancellation also stops the role task."""
        if self.callback is None:
            return await operation
        task: asyncio.Future[T] = asyncio.ensure_future(operation)
        elapsed = 0
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=HEARTBEAT_SECONDS)
                if done:
                    return await task
                elapsed += HEARTBEAT_SECONDS
                self.emit(
                    DevelopmentEvent(
                        phase=event.phase,
                        status="running",
                        build_id=event.build_id,
                        model=event.model,
                        attempt=event.attempt,
                        attempts=event.attempts,
                        detail=f"Етап виконується вже {elapsed} с.",
                    )
                )
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

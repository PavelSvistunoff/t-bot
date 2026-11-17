"""Report observed tool outcomes when the model stops without a user-facing answer."""

from dataclasses import dataclass, field

from google.adk.events import Event

from t_bot.errors import BotError


@dataclass
class TaskProgress:
    outcomes: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    team_result: dict | None = None

    def observe(self, event: Event) -> None:
        for part in (event.content.parts or []) if event.content else []:
            response = part.function_response
            if response is None:
                continue
            data = response.response or {}
            name = response.name or "інструмент"
            if name == "create_extension" and data.get("team_run_id"):
                self.team_result = data
            error = data.get("error")
            if error:
                # Show an actionable error, never model thoughts or an entire subprocess traceback.
                lines = str(error).strip().splitlines()
                detail = next(
                    (
                        line
                        for line in reversed(lines)
                        if line.startswith(("ERROR:", "ModuleNotFoundError:"))
                    ),
                    lines[-1] if lines else "Невідома помилка інструмента.",
                )
                self.failures[name] = detail[:600]
                self.outcomes.append(f"{name}: помилка — {detail[:600]}")
                continue
            self.failures.pop(name, None)
            if name == "create_extension" and data.get("status") == "active":
                self.failures.pop("install_dependencies", None)
                manifest = data.get("manifest", {})
                self.outcomes.append(
                    f"Створено {manifest.get('id')} v{manifest.get('version')}; перевірка пройдена."
                )
            elif name == "install_dependencies" and data.get("status") == "installed":
                self.outcomes.append(
                    "Встановлено бібліотеки: " + ", ".join(data.get("dependencies", []))
                )
            elif name == "send_file" and data.get("status") == "sent":
                self.outcomes.append("Файл надіслано: " + str(data.get("filename", "")))

    def team_summary(self) -> str:
        """Produce a factual final answer without another model call after the team finishes."""
        if self.team_result is None:
            raise BotError("Результату команди ще немає.")
        result = self.team_result
        command = result.get("report_command", "/build " + result["team_run_id"])
        if result.get("status") != "active":
            raise BotError(
                "Розширення не активовано. "
                + str(result.get("error", ""))
                + "\nЗбережений звіт: "
                + command
            )
        manifest = result["manifest"]
        return (
            f"Створено {manifest['id']} v{manifest['version']}. "
            f"QA: {result.get('qa', {}).get('tests_run', 0)} тестів пройдено; "
            "Code reviewer схвалив реалізацію.\n"
            + str(result.get("usage", ""))
            + "\nЗвіт команди: "
            + command
            + "\nРозширення доступне з наступного повідомлення."
        )

    def missing_answer(self) -> str:
        text = "Модель завершила роботу без текстової відповіді (MODEL_RETURNED_NO_CONTENT)."
        if self.outcomes:
            text += "\nФактичні результати:\n" + "\n".join(self.outcomes[-6:])
        else:
            text += "\nПідтверджених результатів інструментів немає."
        text += "\nДії автоматично не повторювалися."
        if self.failures:
            text += " Деякі операції не завершилися; перевірте помилки вище."
            raise BotError(text)
        if not self.outcomes:
            raise BotError(text)
        return text

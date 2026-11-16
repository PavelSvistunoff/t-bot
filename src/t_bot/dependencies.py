"""Resolve wheel-only extension requirements before installing an exact package set."""

import asyncio
import json
import re
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlparse

from t_bot.config import Settings
from t_bot.errors import BotError
from t_bot.manifest import validate_dependency_list


def package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.split("==")[0]).lower()


def validate_requests(values: list[str]) -> list[str]:
    if len(values) > 64 or any(
        not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*(==[a-zA-Z0-9_.+!-]+)?", value)
        for value in values
    ):
        raise BotError("Укажіть до 64 назв пакетів або package==version, без URL і pip-параметрів.")
    return sorted(set(values))


class DependencyResolver:
    def __init__(self, settings: Settings, run: Callable[..., Awaitable[bytes]]):
        self.settings = settings
        self.run = run
        self.cache: dict[tuple[str, ...], list[str]] = {}
        self.lock = asyncio.Lock()

    def check_allowed(self, packages: list[str]) -> None:
        allowed = self.settings.allowed_dependencies
        if allowed:
            normalized = {(package_name(p), p.split("==", 1)[-1]) for p in allowed}
            if any((package_name(p), p.split("==", 1)[-1]) not in normalized for p in packages):
                raise BotError(
                    "Набір містить залежність, відсутню в ALLOWED_DEPENDENCIES: "
                    + ", ".join(
                        p
                        for p in packages
                        if (package_name(p), p.split("==", 1)[-1]) not in normalized
                    )
                )

    async def resolve(self, requirements: list[str]) -> list[str]:
        requests = validate_requests(requirements)
        if not requests:
            return []
        allowed = self.settings.allowed_dependencies
        if allowed:
            for request in requests:
                if "==" in request:
                    self.check_allowed([request])
                elif not any(package_name(p) == package_name(request) for p in allowed):
                    raise BotError("Пакет відсутній в ALLOWED_DEPENDENCIES: " + request)
        async with self.lock:
            key = tuple(requests)
            if key in self.cache:
                self.check_allowed(self.cache[key])
                return list(self.cache[key])
            try:
                import pip

                with tempfile.TemporaryDirectory(prefix="t-bot-resolve-") as temporary:
                    report = Path(temporary) / "report.json"
                    command = [
                        sys.executable,
                        "-I",
                        str(Path(pip.__file__).parent / "__main__.py"),
                        "install",
                        "--isolated",
                        "--no-cache-dir",
                        "--only-binary=:all:",
                        "--ignore-installed",
                        "--dry-run",
                        "--quiet",
                        "--report",
                        str(report),
                    ]
                    if allowed:
                        constraints = Path(temporary) / "constraints.txt"
                        await asyncio.to_thread(constraints.write_text, "\n".join(sorted(allowed)))
                        command += ["--constraint", str(constraints)]
                    try:
                        await self.run(command + requests, timeout_seconds=180)
                    except BotError as exc:
                        raise BotError(
                            "pip resolver не знайшов сумісного набору wheel-пакетів. "
                            "Вилучіть вгадані транзитивні залежності й передайте основні назви "
                            "(наприклад matplotlib), або виберіть доступну версію. "
                            + str(exc)[-1200:]
                        ) from exc
                    data = json.loads(await asyncio.to_thread(report.read_text))
                    if data.get("version") != "1" or not data.get("install"):
                        raise ValueError("Unsupported or empty pip report")
                    pins = []
                    for item in data["install"]:
                        url = item["download_info"]["url"]
                        if item.get("is_direct") or not urlparse(url).path.lower().endswith(".whl"):
                            raise ValueError("Only index wheels are supported")
                        metadata = item["metadata"]
                        pins.append(f"{metadata['name']}=={metadata['version']}")
                    pins = validate_dependency_list(pins)
                    self.check_allowed(pins)
                    versions = {package_name(p): p.split("==", 1)[1] for p in pins}
                    if len(versions) != len(pins) or any(
                        package_name(p) not in versions
                        or ("==" in p and versions[package_name(p)] != p.split("==", 1)[1])
                        for p in requests
                    ):
                        raise ValueError(
                            "Report differs from requested packages or pinned versions"
                        )
                    self.cache[key] = pins
                    self.cache[tuple(pins)] = pins
                    return list(pins)
            except ImportError as exc:
                raise BotError("Установіть pip у середовище бота.") from exc
            except (ValueError, KeyError, TypeError, OSError) as exc:
                raise BotError("Не вдалося прочитати коректний звіт pip resolver.") from exc

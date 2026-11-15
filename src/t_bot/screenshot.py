"""Capture a display on request, keeping image files only in temporary storage."""

import asyncio
import ctypes
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from t_bot.errors import BotError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ScreenImage:
    data: bytes
    width: int
    height: int

    @property
    def fits_telegram_photo(self) -> bool:
        return (
            len(self.data) <= 10_000_000
            and self.width + self.height <= 10000
            and max(self.width, self.height) / min(self.width, self.height) <= 20
        )


class ScreenCapture:
    def __init__(self, timeout_seconds: int = 10, max_bytes: int = 20_000_000):
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes

    @staticmethod
    def _system_tool(name: str) -> str | None:
        # Workspace executables must not become trusted, unrestricted capture backends via PATH.
        for directory in ("/usr/bin", "/bin"):
            path = Path(directory) / name
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        return None

    @staticmethod
    def _mac_permission() -> bool:
        library = ctypes.CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics",
        )
        check = library.CGPreflightScreenCaptureAccess
        check.argtypes = []
        check.restype = ctypes.c_bool
        return bool(check())

    def _command(self, path: Path) -> list[str]:
        if sys.platform == "darwin":
            if not self._mac_permission():
                raise BotError(
                    "Немає дозволу на скріншоти. У налаштуваннях конфіденційності macOS "
                    "дозвольте запис екрана терміналу або IDE, що запускає бота, "
                    "і перезапустіть бота."
                )
            return ["/usr/sbin/screencapture", "-x", "-m", "-t", "png", str(path)]
        if sys.platform.startswith("linux"):
            if os.environ.get("WAYLAND_DISPLAY") and (grim := self._system_tool("grim")):
                return [grim, str(path)]
            if os.environ.get("DISPLAY") and (scrot := self._system_tool("scrot")):
                return [scrot, str(path)]
            if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and (
                gnome := self._system_tool("gnome-screenshot")
            ):
                return [gnome, "-f", str(path)]
            raise BotError(
                "Скріншот потребує графічного сеансу DISPLAY/WAYLAND_DISPLAY і "
                "встановленого grim, scrot або gnome-screenshot."
            )
        raise BotError("Скріншоти підтримуються на macOS і Linux із графічним сеансом.")

    def _read_image(self, path: Path) -> ScreenImage:
        try:
            with path.open("rb") as file:
                data = file.read(self.max_bytes + 1)
        except OSError as exc:
            raise BotError(
                "Система не створила файл скріншота. Перевірте доступ до екрана."
            ) from exc
        if len(data) > self.max_bytes:
            raise BotError("Скріншот перевищує SCREENSHOT_MAX_BYTES.")
        if len(data) < 24 or not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
            raise BotError("Система повернула некоректний PNG-скріншот.")
        width, height = struct.unpack(">II", data[16:24])
        if not width or not height:
            raise BotError("Некоректні розміри скріншота.")
        return ScreenImage(data, width, height)

    async def capture(self) -> ScreenImage:
        with tempfile.TemporaryDirectory(prefix="t-bot-screen-") as directory:
            path = Path(directory) / "screen.png"
            arguments = await asyncio.to_thread(self._command, path)
            environment = {
                key: value
                for key, value in os.environ.items()
                if key
                in {
                    "PATH",
                    "HOME",
                    "LANG",
                    "DISPLAY",
                    "WAYLAND_DISPLAY",
                    "XDG_RUNTIME_DIR",
                    "DBUS_SESSION_BUS_ADDRESS",
                    "XAUTHORITY",
                    "XDG_SESSION_TYPE",
                }
            }
            environment["TMPDIR"] = directory
            try:
                process = await asyncio.create_subprocess_exec(
                    *arguments,
                    env=environment,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except OSError as exc:
                raise BotError("Не вдалося запустити системну утиліту скріншота.") from exc
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    await process.wait()
                if process.returncode:
                    raise BotError(
                        "Не вдалося зробити скріншот. Перевірте доступ до запису екрана "
                        "і наявність активного графічного сеансу."
                    )
                return await asyncio.to_thread(self._read_image, path)
            except TimeoutError as exc:
                raise BotError("Перевищено час створення скріншота.") from exc
            finally:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()

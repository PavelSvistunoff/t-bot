import asyncio
import base64

import pytest

from t_bot.errors import BotError
from t_bot.screenshot import ScreenCapture, ScreenImage

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII="
)


def test_image_validates_format_size_and_photo_dimensions(tmp_path):
    path = tmp_path / "screen.png"
    path.write_bytes(PNG)
    image = ScreenCapture()._read_image(path)
    assert (image.width, image.height) == (1, 1)
    assert image.fits_telegram_photo
    assert not ScreenImage(PNG, 10000, 10000).fits_telegram_photo
    assert not ScreenImage(PNG, 1000, 1).fits_telegram_photo
    path.write_bytes(b"not a png")
    with pytest.raises(BotError, match="PNG"):
        ScreenCapture()._read_image(path)
    path.write_bytes(PNG * 100)
    with pytest.raises(BotError, match="SCREENSHOT_MAX_BYTES"):
        ScreenCapture(max_bytes=1024)._read_image(path)


async def test_capture_returns_bytes_and_removes_temporary_file(monkeypatch):
    capture = ScreenCapture()
    seen = []

    def command(path):
        seen.append(path)
        return ["fake-capture", str(path)]

    class Process:
        returncode = None

        async def wait(self):
            seen[0].write_bytes(PNG)
            self.returncode = 0
            return 0

    async def launch(*args, **kwargs):
        assert "TELEGRAM_BOT_TOKEN" not in kwargs["env"]
        return Process()

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "private-token")
    monkeypatch.setattr(capture, "_command", command)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    assert (await capture.capture()).data == PNG
    assert not seen[0].parent.exists()


@pytest.mark.parametrize("mode", ["timeout", "failure"])
async def test_capture_errors_are_reported_and_process_is_cleaned(monkeypatch, mode):
    capture = ScreenCapture(timeout_seconds=0.01)

    class Process:
        returncode = None
        killed = False

        async def wait(self):
            if self.killed:
                self.returncode = -9
                return -9
            if mode == "timeout":
                await asyncio.sleep(1)
            self.returncode = 1
            return 1

        def kill(self):
            self.killed = True

    process = Process()

    async def launch(*args, **kwargs):
        return process

    monkeypatch.setattr(capture, "_command", lambda path: ["fake"])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    with pytest.raises(BotError):
        await capture.capture()
    if mode == "timeout":
        assert process.killed


def test_missing_mac_permission_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setattr("t_bot.screenshot.sys.platform", "darwin")
    monkeypatch.setattr(ScreenCapture, "_mac_permission", staticmethod(lambda: False))
    with pytest.raises(BotError, match="дозвольте запис екрана"):
        ScreenCapture()._command(tmp_path / "screen.png")


def test_linux_backend_selection_and_headless_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("t_bot.screenshot.sys.platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(
        ScreenCapture,
        "_system_tool",
        staticmethod(lambda name: "/usr/bin/grim" if name == "grim" else None),
    )
    assert ScreenCapture()._command(tmp_path / "screen.png")[0] == "/usr/bin/grim"
    monkeypatch.delenv("WAYLAND_DISPLAY")
    with pytest.raises(BotError, match="графічного сеансу"):
        ScreenCapture()._command(tmp_path / "screen.png")


async def test_capture_cancellation_terminates_process(monkeypatch):
    capture = ScreenCapture()
    started = asyncio.Event()

    class Process:
        returncode = None
        killed = False

        async def wait(self):
            if not self.killed:
                started.set()
                await asyncio.sleep(30)
            self.returncode = -9
            return -9

        def kill(self):
            self.killed = True

    process = Process()

    async def launch(*args, **kwargs):
        return process

    monkeypatch.setattr(capture, "_command", lambda path: ["fake"])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    task = asyncio.create_task(capture.capture())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.killed

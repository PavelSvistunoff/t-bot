"""Finite, non-interactive shell commands with native OS isolation and bounded output."""

import asyncio
import json
import os
import shutil
import signal
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from t_bot.config import Settings
from t_bot.errors import BotError
from t_bot.filesystem import FileSystem


@dataclass
class Output:
    limit: int
    data: bytearray = field(default_factory=bytearray)
    truncated: bool = False

    async def read(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(8192):
            remaining = self.limit - len(self.data)
            self.data.extend(chunk[:remaining])
            self.truncated |= len(chunk) > remaining

    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


class Console:
    def __init__(self, settings: Settings, filesystem: FileSystem):
        self.settings = settings
        self.filesystem = filesystem
        self.project = Path.cwd().resolve()

    def _paths(self, temporary: Path) -> tuple[list[Path], list[Path], list[Path]]:
        readable = [
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            self.filesystem.root,
            temporary,
        ]
        protected = [
            self.settings.data_dir,
            self.settings.extensions_dir,
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            Path(__file__).parent,
            self.project / "AGENTS.md",
        ]
        private = [
            self.project / ".env",
            Path.home() / ".ssh",
            Path.home() / ".aws",
            Path.home() / ".codex",
        ]
        return readable, protected, private

    def _command(self, command: str, temporary: Path) -> list[str]:
        readable, protected, private = self._paths(temporary)
        shell = ["/bin/sh", "-c", command]
        if sys.platform == "darwin" and shutil.which("sandbox-exec"):

            def paths(values: list[Path]) -> str:
                return " ".join(
                    f"(subpath {json.dumps(str(p), ensure_ascii=False)})" for p in values
                )

            profile = (
                "(version 1)(deny default)(allow process-fork)(allow process-exec)"
                "(allow signal (target same-sandbox))(allow sysctl-read)"
                '(allow file-read-metadata)(allow file-read* (literal "/"))'
                f"(allow file-read* {paths(readable)} "
                '(subpath "/System/Library") (subpath "/usr") (subpath "/bin") '
                '(subpath "/sbin") (subpath "/opt/homebrew") (subpath "/Library/Apple") '
                '(subpath "/Library/Developer") '
                '(literal "/dev/null") (literal "/dev/random") (literal "/dev/urandom") '
                '(literal "/private/etc/hosts") (literal "/private/etc/resolv.conf") '
                '(literal "/private/etc/localtime") (literal "/private/etc/services") '
                '(literal "/private/etc/protocols") (literal "/private/etc/ssl/openssl.cnf") '
                '(literal "/private/etc/ssl/cert.pem") (subpath "/private/etc/ssl/certs"))'
                f"(allow file-write* {paths([self.filesystem.root, temporary])} "
                '(literal "/dev/null"))'
                f"(deny file-write* {paths(protected)})"
                f"(deny file-read* file-write* {paths(private)})"
                "(deny file-link)(allow network*)"
            )
            return ["/usr/bin/sandbox-exec", "-p", profile, *shell]
        if sys.platform.startswith("linux") and shutil.which("bwrap"):
            result = [
                "bwrap",
                "--unshare-all",
                "--share-net",
                "--die-with-parent",
                "--new-session",
                "--cap-drop",
                "ALL",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
            ]
            for path in [
                Path("/usr"),
                Path("/bin"),
                Path("/sbin"),
                Path("/lib"),
                Path("/lib64"),
                Path("/etc/ssl/certs"),
                Path("/etc/hosts"),
                Path("/etc/resolv.conf"),
                *readable,
            ]:
                if path.exists():
                    result += ["--ro-bind", str(path), str(path)]
            for path in (self.filesystem.root, temporary):
                result += ["--bind", str(path), str(path)]
            for path in protected:
                if path.exists():
                    result += ["--ro-bind", str(path), str(path)]
            for path in private:
                if path.is_file():
                    result += ["--ro-bind", "/dev/null", str(path)]
                elif path.is_dir():
                    result += ["--tmpfs", str(path)]
            return [*result, *shell]
        raise BotError("Консоль потребує macOS sandbox-exec або Linux bubblewrap.")

    async def _terminate_group(self, process: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except PermissionError:
            # Some host policies permit signalling individual children but reject killpg.
            listing = await asyncio.create_subprocess_exec(
                "/bin/ps",
                "-axo",
                "pid=,pgid=",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            output, _ = await listing.communicate()
            members = [
                int(parts[0])
                for line in output.splitlines()
                if len(parts := line.split()) == 2 and int(parts[1]) == process.pid
            ]
            for pid in reversed(members):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.returncode is None:
                process.kill()

    async def run(self, command: str, cwd: str = ".", timeout_seconds: int = 0) -> dict:
        if not command.strip() or len(command) > 12000 or "\x00" in command:
            raise BotError("Команда має містити від 1 до 12000 символів без NUL.")
        timeout_seconds = timeout_seconds or self.settings.console_timeout_seconds
        if not 1 <= timeout_seconds <= self.settings.console_timeout_seconds:
            raise BotError(
                f"Таймаут має бути від 1 до {self.settings.console_timeout_seconds} секунд."
            )
        directory = await asyncio.to_thread(self.filesystem.directory_path, cwd)
        with tempfile.TemporaryDirectory(prefix="t-bot-console-") as name:
            temporary = await asyncio.to_thread(Path(name).resolve)
            arguments = self._command(command, temporary)
            environment = {
                "PATH": (
                    f"{Path(sys.executable).parent}:/usr/bin:/bin:/usr/sbin:/sbin:"
                    "/opt/homebrew/bin:/usr/local/bin"
                ),
                "HOME": str(temporary),
                "TMPDIR": str(temporary),
                "LANG": "en_US.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "DO_NOT_TRACK": "1",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            }
            try:
                process = await asyncio.create_subprocess_exec(
                    *arguments,
                    cwd=directory,
                    env=environment,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except OSError as exc:
                raise BotError("Не вдалося запустити консольну команду.") from exc
            assert process.stdout and process.stderr
            stdout = Output(self.settings.console_max_output_bytes)
            stderr = Output(self.settings.console_max_output_bytes)
            readers = [
                asyncio.create_task(stdout.read(process.stdout)),
                asyncio.create_task(stderr.read(process.stderr)),
            ]
            capture = asyncio.gather(*readers)
            timed_out = False
            try:
                async with asyncio.timeout(timeout_seconds):
                    # Keep draining pipes while timeout/cancellation stops the process group.
                    await asyncio.shield(capture)
                    await process.wait()
            except TimeoutError:
                timed_out = True
            finally:
                cleanup = asyncio.create_task(self._terminate_group(process))
                await asyncio.shield(cleanup)
                await process.wait()
                for reader in readers:
                    reader.cancel()
                await asyncio.gather(capture, return_exceptions=True)
            return {
                "stdout": stdout.text(),
                "stderr": stderr.text(),
                "exit_code": process.returncode,
                "timed_out": timed_out,
                "stdout_truncated": stdout.truncated,
                "stderr_truncated": stderr.truncated,
                "cwd": str(directory),
            }

"""OS-enforced subprocess isolation: Seatbelt on macOS, bubblewrap on Linux."""

import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path

from t_bot.config import Settings
from t_bot.errors import BotError
from t_bot.filesystem import BLOCKED_COMPONENTS
from t_bot.manifest import validate_dependency_list

MAX_OUTPUT_BYTES = 128_000


async def bounded_process(
    command: list[str],
    *,
    input_data: bytes = b"",
    timeout_seconds: int,
    cwd: Path | None = None,
    memory_mb: int = 512,
) -> bytes:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            start_new_session=True,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(cwd or "/tmp"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "DO_NOT_TRACK": "1",
                "LANG": "en_US.UTF-8",
            },
        )
    except FileNotFoundError as exc:
        raise BotError("Відсутній виконуваний файл для ізоляції процесу.") from exc

    async def read_limited(stream: asyncio.StreamReader) -> bytes:
        chunks = bytearray()
        while chunk := await stream.read(8192):
            chunks.extend(chunk)
            if len(chunks) > MAX_OUTPUT_BYTES:
                raise BotError("Розширення перевищило ліміт виводу.")
        return bytes(chunks)

    async def monitor_memory() -> None:
        # Seatbelt denies fork; RSS of this single process is the total extension RSS.
        while process.returncode is None:
            probe = await asyncio.create_subprocess_exec(
                "/bin/ps",
                "-o",
                "rss=",
                "-p",
                str(process.pid),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            output, _ = await probe.communicate()
            if output.strip() and int(output.strip()) > memory_mb * 1024:
                raise BotError("Розширення перевищило ліміт пам’яті.")
            await asyncio.sleep(0.1)

    async def communicate() -> tuple[bytes, bytes]:
        assert process.stdin and process.stdout and process.stderr
        process.stdin.write(input_data)
        await process.stdin.drain()
        process.stdin.close()
        output, error = await asyncio.gather(
            read_limited(process.stdout),
            read_limited(process.stderr),
        )
        await process.wait()
        return output, error

    monitor = asyncio.create_task(monitor_memory())
    communication = asyncio.create_task(communicate())
    try:
        async with asyncio.timeout(timeout_seconds):
            done, _ = await asyncio.wait(
                [monitor, communication],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if monitor in done:
                await monitor
            output, error = await communication
        if process.returncode:
            raise BotError(
                f"Ізольована операція не вдалася (код {process.returncode}): "
                + error.decode(errors="replace")[-1500:]
            )
        return output
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass  # The process may have exited between polling and termination.
            await process.wait()
        for task in (monitor, communication):
            task.cancel()
        await asyncio.gather(monitor, communication, return_exceptions=True)


class Sandbox:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dependencies_dir = settings.data_dir / "dependencies"
        self.dependencies_dir.mkdir(exist_ok=True, mode=0o700)
        self.provision_lock = asyncio.Lock()

    def dependency_path(self, dependencies: list[str]) -> Path:
        digest = hashlib.sha256("\n".join(sorted(dependencies)).encode()).hexdigest()[:20]
        return self.dependencies_dir / digest

    def _read_paths(self, extension_dir: Path, dependencies: Path) -> list[Path]:
        return list(
            dict.fromkeys(
                path.resolve()
                for path in (
                    Path(sys.prefix),
                    Path(sys.base_prefix),
                    extension_dir,
                    Path(__file__).with_name("sandbox_worker.py"),
                    dependencies,
                )
            )
        )

    def workspace_root(self) -> Path:
        """Validate a persistent workspace that cannot expose the host's private runtime state."""
        root = self.settings.filesystem_root
        if root.resolve() != root or any(
            part.casefold() in BLOCKED_COMPONENTS or part.casefold().startswith(".env")
            for part in root.parts
        ):
            raise BotError("FILESYSTEM_ROOT не може бути секретним каталогом або посиланням.")
        protected = (
            self.settings.data_dir,
            self.settings.extensions_dir,
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            Path(__file__).parent.resolve(),
            Path.cwd() / "AGENTS.md",
        )
        if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
            raise BotError(
                "FILESYSTEM_ROOT для extensions має бути окремим від стану, коду й Python "
                "середовища бота; виберіть окремий робочий каталог."
            )
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return root

    def private_workspace_paths(self, root: Path) -> list[Path]:
        """Find existing private entries to hide in a Linux workspace bind mount."""
        hidden = []
        for directory, folders, files in os.walk(root, followlinks=False):
            for name in [*folders, *files]:
                path = Path(directory) / name
                private = name.casefold() in BLOCKED_COMPONENTS or name.casefold().startswith(
                    ".env"
                )
                if path.is_symlink():
                    continue  # Targets outside the mounted workspace remain inaccessible.
                if private or (path.is_file() and path.stat().st_nlink > 1):
                    hidden.append(path)
                    if name in folders:
                        folders.remove(name)
        return hidden

    def command(self, directory: Path, dependencies: Path) -> list[str]:
        worker = str(Path(__file__).with_name("sandbox_worker.py").resolve())
        python = str(Path(sys.executable).absolute())
        reads = self._read_paths(directory, dependencies)
        writable_names = ("tmp",)
        root = self.workspace_root()
        if sys.platform == "darwin" and shutil.which("sandbox-exec"):
            literals = " ".join(f"(subpath {json.dumps(str(path))})" for path in reads)
            writable = " ".join(
                f"(subpath {json.dumps(str(directory / name))})" for name in writable_names
            )
            writable += f" (subpath {json.dumps(str(root))})"
            executables = [Path(python).resolve()]
            framework = sysconfig.get_config_var("PYTHONFRAMEWORK")
            if framework:
                # Framework Python's launcher replaces itself with the bundled interpreter.
                # Permit that exact executable, while process-fork remains denied.
                interpreter = (
                    Path(sys.base_prefix)
                    / "Resources"
                    / f"{framework}.app"
                    / "Contents/MacOS"
                    / framework
                )
                if interpreter.is_file():
                    executables.append(interpreter.resolve())
            executable_literals = " ".join(
                f"(literal {json.dumps(str(path))})" for path in executables
            )
            profile = (
                "(version 1)(deny default)(allow signal (target self))"
                "(allow sysctl-read)"
                '(allow file-read-metadata)(allow file-read* (literal "/"))'
                f"(allow file-read* {literals} "
                '(subpath "/System/Library") (subpath "/usr/lib") '
                '(subpath "/Library/Apple") (subpath "/usr/share/zoneinfo") '
                '(literal "/dev/urandom") (literal "/dev/random") (literal "/dev/null"))'
                f'(allow file-write* {writable} (literal "/dev/null"))'
                f"(allow file-read* {writable})"
                f"(allow process-exec {executable_literals})"
            )
            components = "|".join(re.escape(name) for name in sorted(BLOCKED_COMPONENTS))
            private_pattern = (
                "^" + re.escape(str(root)) + "/(.*/)?(" + components + "|\\.env[^/]*)(/|$)"
            )
            profile += (
                f"(deny file-read* file-write* (regex {json.dumps(private_pattern)}))"
                "(deny file-link)"
            )
            for path in self.private_workspace_paths(root):
                profile += f"(deny file-read* file-write* (subpath {json.dumps(str(path))}))"
            return ["/usr/bin/sandbox-exec", "-p", profile, python, "-I", worker]
        if sys.platform.startswith("linux") and shutil.which("bwrap"):
            command = [
                "bwrap",
                "--unshare-all",
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
            for path in [Path("/usr"), Path("/lib"), Path("/lib64"), *reads]:
                if path.exists():
                    command += ["--ro-bind", str(path), str(path)]
            for name in writable_names:
                path = directory / name
                command += ["--bind", str(path), str(path)]
            command += ["--bind", str(root), str(root)]
            for path in self.private_workspace_paths(root):
                if path.is_dir():
                    command += ["--tmpfs", str(path)]
                else:
                    command += ["--ro-bind", "/dev/null", str(path)]
            return command + ["--chdir", str(directory), python, "-I", worker]
        raise BotError("Ізоляція недоступна: потрібні macOS sandbox-exec або Linux bubblewrap.")

    async def ready(self) -> bool:
        try:
            await self.execute(None, {}, "probe", {})
            return True
        except (BotError, TimeoutError):
            return False

    async def provision(self, dependencies: list[str]) -> None:
        try:
            dependencies = validate_dependency_list(dependencies)
        except ValueError as exc:
            raise BotError(str(exc)) from exc
        if (
            self.settings.allowed_dependencies
            and not set(dependencies) <= self.settings.allowed_dependencies
        ):
            raise BotError("Залежність відсутня в операторському ALLOWED_DEPENDENCIES.")
        destination = self.dependency_path(dependencies)
        async with self.provision_lock:
            if destination.exists():
                return
            if not dependencies:
                destination.mkdir(mode=0o755)
                return
            # Wheels only; no setup.py, source builds or package-provided executables run.
            # Transitive dependencies must also be pinned and included in the manifest.
            staging = Path(tempfile.mkdtemp(dir=self.dependencies_dir))
            try:
                import pip  # type: ignore[import-not-found]

                pip_main = Path(pip.__file__).parent / "__main__.py"
                await bounded_process(
                    [
                        sys.executable,
                        "-I",
                        str(pip_main),
                        "install",
                        "--isolated",
                        "--no-cache-dir",
                        "--only-binary=:all:",
                        "--no-deps",
                        "--target",
                        str(staging),
                        *dependencies,
                    ],
                    timeout_seconds=180,
                )
                await asyncio.to_thread(staging.rename, destination)
            except ImportError as exc:
                raise BotError("Установіть pip у середовище: python -m ensurepip.") from exc
            finally:
                if await asyncio.to_thread(staging.exists):
                    shutil.rmtree(staging)

    async def execute(
        self,
        directory: Path | None,
        manifest: dict,
        action: str,
        payload: dict,
    ) -> dict:
        dependencies = manifest.get("dependencies", [])
        await self.provision(dependencies)
        with tempfile.TemporaryDirectory(prefix="t-bot-isolated-") as temporary:
            isolated = await asyncio.to_thread(Path(temporary).resolve)
            (isolated / "tmp").mkdir()
            root = await asyncio.to_thread(self.workspace_root)
            if directory:
                shutil.copyfile(directory / "extension.py", isolated / "extension.py")
            request = {
                "action": action,
                "manifest": manifest,
                "payload": payload,
                "directory": str(isolated),
                "filesystem_root": str(root),
                "dependencies": str(self.dependency_path(dependencies)),
                "timeout": self.settings.sandbox_timeout_seconds,
                "memory_mb": self.settings.sandbox_memory_mb,
            }
            data = json.dumps(request).encode()
            if len(data) > MAX_OUTPUT_BYTES:
                raise BotError("Завеликий запит до розширення.")
            try:
                output = await bounded_process(
                    await asyncio.to_thread(
                        self.command,
                        isolated,
                        self.dependency_path(dependencies),
                    ),
                    input_data=data,
                    timeout_seconds=self.settings.sandbox_timeout_seconds,
                    cwd=isolated,
                    memory_mb=self.settings.sandbox_memory_mb,
                )
                result = json.loads(output)
                if not isinstance(result, dict):
                    raise BotError("Розширення має повертати JSON-об’єкт.")
                return result
            except (ValueError, TimeoutError) as exc:
                raise BotError(
                    "Некоректна відповідь розширення або перевищено час виконання."
                ) from exc

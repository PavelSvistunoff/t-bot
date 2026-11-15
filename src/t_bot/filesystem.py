"""Bounded file operations anchored to directory descriptors, without following symlinks."""

import json
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from t_bot.errors import BotError

BLOCKED_COMPONENTS = {".git", ".ssh", ".aws", ".codex", ".agents", ".venv", ".tools", ".cache"}
MAX_DIRECTORY_ENTRIES = 100
MAX_LISTING_CHARS = 10000
MAX_READ_CHARS = 12000


class FileSystem:
    def __init__(self, root: Path, max_file_bytes: int, protected_paths: tuple[Path, ...]):
        self.root = root.resolve()
        self.max_file_bytes = max_file_bytes
        self.protected_paths = tuple(path.resolve() for path in protected_paths)

    def _parts(self, path: str) -> tuple[str, ...]:
        if "\x00" in path or len(path) > 4096:
            raise BotError("Некоректний шлях файлу.")
        candidate = Path(path)
        if ".." in candidate.parts:
            raise BotError("Вихід за межі FILESYSTEM_ROOT заборонено.")
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as exc:
            raise BotError("Шлях має бути всередині FILESYSTEM_ROOT.") from exc
        for component in relative.parts:
            name = component.casefold()
            if name in BLOCKED_COMPONENTS or name.startswith(".env"):
                raise BotError("Доступ до секретів і службових каталогів заборонено.")
        normalized = Path(str(candidate).casefold())
        if any(
            normalized.is_relative_to(Path(str(protected).casefold()))
            for protected in self.protected_paths
        ):
            raise BotError("Доступ до внутрішніх файлів бота заборонено.")
        return relative.parts

    @contextmanager
    def _directory(self, parts: tuple[str, ...], *, create: bool = False) -> Iterator[int]:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts:
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass  # Opening below still rejects non-directories and symlinks.
                following = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = following
            yield descriptor
        finally:
            os.close(descriptor)

    def list_files(self, path: str = ".") -> dict:
        parts = self._parts(path)
        entries: list[dict] = []
        truncated, length = False, 0
        with self._directory(parts) as directory, os.scandir(directory) as contents:
            for entry in contents:
                try:
                    self._parts(str(Path(*parts) / entry.name))
                    info = entry.stat(follow_symlinks=False)
                except (BotError, FileNotFoundError):
                    continue
                kind = "directory" if stat.S_ISDIR(info.st_mode) else "file"
                if stat.S_ISLNK(info.st_mode):
                    kind = "symlink"
                item = {"name": entry.name, "kind": kind, "size_bytes": info.st_size}
                length += len(json.dumps(item, ensure_ascii=False))
                if len(entries) == MAX_DIRECTORY_ENTRIES or length > MAX_LISTING_CHARS:
                    truncated = True
                    break
                entries.append(item)
        return {
            "path": str(Path(*parts)),
            "entries": sorted(entries, key=lambda e: e["name"]),
            "truncated": truncated,
        }

    def directory_path(self, path: str = ".") -> Path:
        """Validate an existing directory for a command's initial working directory."""
        parts = self._parts(path)
        with self._directory(parts):
            return self.root.joinpath(*parts)

    def read_file(self, path: str, offset: int = 0, limit: int = MAX_READ_CHARS) -> dict:
        if offset < 0 or not 1 <= limit <= MAX_READ_CHARS:
            raise BotError("offset має бути >= 0, limit — від 1 до 12000 символів.")
        parts = self._parts(path)
        if not parts:
            raise BotError("Укажіть файл, а не кореневий каталог.")
        with self._directory(parts[:-1]) as directory:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
            with os.fdopen(descriptor, "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise BotError("Читання підтримується лише для звичайних файлів.")
                data = handle.read(self.max_file_bytes + 1)
        if len(data) > self.max_file_bytes:
            raise BotError("Файл перевищує FILESYSTEM_MAX_FILE_BYTES.")
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BotError("Файл має бути текстовим у кодуванні UTF-8.") from exc
        if "\x00" in content:
            raise BotError("Бінарні файли не підтримуються.")
        return {
            "path": str(Path(*parts)),
            "content": content[offset : offset + limit],
            "offset": offset,
            "total_chars": len(content),
            "truncated": offset + limit < len(content),
        }

    def read_attachment(self, path: str, max_bytes: int) -> tuple[str, bytes]:
        """Read a bounded regular binary file without following links or exposing internals."""
        parts = self._parts(path)
        if not parts:
            raise BotError("Укажіть файл, а не каталог.")
        with self._directory(parts[:-1]) as directory:
            descriptor = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
            with os.fdopen(descriptor, "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise BotError("Дозволені лише звичайні файли без жорстких посилань.")
                if info.st_size > max_bytes:
                    raise BotError("Файл перевищує ATTACHMENT_MAX_BYTES.")
                data = handle.read(max_bytes + 1)
        if not data:
            raise BotError("Не можна надіслати порожній файл.")
        if len(data) > max_bytes:
            raise BotError("Файл перевищує ATTACHMENT_MAX_BYTES.")
        return parts[-1], data

    def write_file(self, path: str, content: str) -> dict:
        parts = self._parts(path)
        if not parts:
            raise BotError("Укажіть файл, а не кореневий каталог.")
        data = content.encode("utf-8")
        if len(data) > self.max_file_bytes:
            raise BotError("Вміст перевищує FILESYSTEM_MAX_FILE_BYTES.")
        temporary = ".write-" + secrets.token_hex(12)
        with self._directory(parts[:-1]) as directory:
            try:
                existing = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
                if not stat.S_ISREG(existing.st_mode):
                    raise BotError("Можна замінювати лише звичайні файли.")
            except FileNotFoundError:
                pass  # Creating a new regular file is supported.
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, parts[-1], src_dir_fd=directory, dst_dir_fd=directory)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass  # The atomic rename consumes the temporary file on success.
        return {"path": str(Path(*parts)), "written_bytes": len(data)}

    def create_directory(self, path: str) -> dict:
        parts = self._parts(path)
        with self._directory(parts, create=True):
            return {"path": str(Path(*parts)), "status": "directory_ready"}

    def delete_file(self, path: str) -> dict:
        parts = self._parts(path)
        if not parts:
            raise BotError("Не можна видаляти кореневий каталог.")
        with self._directory(parts[:-1]) as directory:
            info = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise BotError("Видалення підтримується лише для звичайних файлів.")
            os.unlink(parts[-1], dir_fd=directory)
        return {"path": str(Path(*parts)), "status": "deleted"}

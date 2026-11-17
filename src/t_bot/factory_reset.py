"""Remove local application state while preserving source, configuration and runtime."""

import argparse
import os
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

from pydantic import ValidationError

from t_bot.config import Settings
from t_bot.errors import BotError
from t_bot.supervisor import process_lock

CACHE_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis"}
PRESERVED = {
    ".git",
    ".env",
    ".env.example",
    ".venv",
    ".tools",
    ".idea",
    ".agents",
    ".codex",
    "src",
    "tests",
    "examples",
    ".github",
    "docs",
    "start",
    "factory-reset",
    "pyproject.toml",
    "uv.lock",
    "AGENTS.md",
    "TASK.md",
    "README.md",
    "LICENSE",
}
LOCK_NAMES = {"supervisor.lock", "worker.lock"}


def reset_targets(project: Path, settings: Settings) -> list[Path]:
    """Preflight all configured paths; reject external, symlinked and protected targets."""
    configured = [
        settings.data_dir,
        settings.extensions_dir,
        settings.filesystem_root,
        settings.sqlite_database_dir,
    ]
    targets = [project / name for name in ("data", "workspace", "backups")]
    cache = project / ".cache"
    runtimes = [Path(value).resolve() for value in (sys.prefix, sys.base_prefix)]
    if any(runtime.is_relative_to(cache) for runtime in runtimes) and cache.is_dir():
        # uv may install the base interpreter here: it is required by the preserved .venv.
        targets.extend(
            child
            for child in cache.iterdir()
            if not any(runtime.is_relative_to(child) for runtime in runtimes)
        )
    else:
        targets.append(cache)
    targets.extend(path for path in configured if path is not None)
    for target in targets:
        if target == project or not target.is_relative_to(project):
            raise BotError(f"Відмова очищати каталог поза проєктом або його корінь: {target}")
        if any(
            Path(runtime).resolve().is_relative_to(target)
            for runtime in (sys.prefix, sys.base_prefix)
        ):
            raise BotError(f"Каталог містить активний Python-інтерпретатор: {target}")
        if any((project / name).is_relative_to(target) for name in PRESERVED):
            raise BotError(f"Каталог містить код або конфігурацію проєкту: {target}")
        if any(part in PRESERVED for part in target.relative_to(project).parts):
            raise BotError(f"Захищений шлях: {target}")
        current = project
        for part in target.relative_to(project).parts:
            current /= part
            if current.is_symlink():
                raise BotError(f"Каталог очищення не може бути symlink: {current}")
    unique = sorted(set(targets), key=lambda path: (len(path.parts), str(path)))
    return [path for path in unique if not any(path.is_relative_to(p) for p in unique if p != path)]


def cache_targets(project: Path, excluded: list[Path]) -> list[Path]:
    """Find project caches without traversing symlinks, environments or application data."""
    targets = []
    for directory, names, files in os.walk(project, followlinks=False):
        root = Path(directory)
        for name in list(names):
            path = root / name
            if name in CACHE_NAMES:
                targets.append(path)
                names.remove(name)
            elif (
                name in PRESERVED - {"src", "tests", "examples", "docs"}
                or name == ".cache"
                or path in excluded
            ):
                names.remove(name)
        for name in files:
            if name.endswith((".pyc", ".pyo")) or name in {".coverage", "coverage.xml"}:
                targets.append(root / name)
    return targets


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def factory_reset(project: Path, settings: Settings, *, dry_run: bool = False) -> list[Path]:
    """Delete application-owned directories and local caches; refuse a running bot.

    DATA_DIR remains with its process lock files so another process cannot bypass held locks.
    External storage is deliberately refused: it may contain unrelated personal files.
    """
    project = project.resolve()
    targets = reset_targets(project, settings)
    targets += cache_targets(project, targets)
    if dry_run:
        return targets
    with ExitStack() as locks:
        if settings.data_dir.exists():
            for name in sorted(LOCK_NAMES):
                locks.enter_context(process_lock(settings.data_dir / name))
        # Validate everything before removal; no partial cleanup on configuration errors.
        for path in targets:
            if settings.data_dir.is_relative_to(path) and settings.data_dir.exists():
                _remove_except_locks(path, settings.data_dir)
            else:
                remove_path(path)
    return targets


def _remove_except_locks(path: Path, data_dir: Path) -> None:
    """Preserve the data directory and locked inodes, including when it is nested."""
    for child in path.iterdir():
        if path == data_dir and child.name in LOCK_NAMES:
            continue
        if not child.is_symlink() and data_dir.is_relative_to(child):
            _remove_except_locks(child, data_dir)
        else:
            remove_path(child)


def main() -> int:
    parser = argparse.ArgumentParser(description="Очистити дані бота та локальні кеші проєкту.")
    parser.add_argument("--dry-run", action="store_true", help="Показати шляхи без видалення")
    args = parser.parse_args()
    try:
        paths = factory_reset(Path.cwd(), Settings(), dry_run=args.dry_run)
    except ValidationError:
        print("Некоректна конфігурація .env; скидання не виконано.", file=sys.stderr)
        return 1
    except (BotError, OSError) as exc:
        print(f"Скидання не завершено: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(f"{'Буде очищено' if args.dry_run else 'Очищено'}: {path}")
    if not args.dry_run:
        print("Заводське скидання завершено. Код, .env та Python-середовище збережено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

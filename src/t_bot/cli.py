import argparse
import asyncio
import json
import os
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from t_bot.config import Settings
from t_bot.errors import BotError
from t_bot.logging_setup import configure_logging


async def doctor(settings: Settings) -> int:
    from t_bot.ollama import Ollama
    from t_bot.sandbox import Sandbox

    ollama = Ollama(settings.ollama_base_url, settings.model_timeout_seconds)
    try:
        models = await ollama.models()
        print("Ollama: OK; models:", ", ".join(models))
        await ollama.require_model(settings.default_model)
        print("Default model: OK")
        isolated = await Sandbox(settings).ready()
        print("OS sandbox:", "OK" if isolated else "UNAVAILABLE (extensions disabled)")
        print(
            "Telegram config:",
            "set" if settings.telegram_bot_token.get_secret_value() else "missing",
        )
        try:
            settings.require_telegram()
            configured = True
        except BotError:
            configured = False
        return 0 if isolated and configured else 1
    finally:
        await ollama.close()


def backup(settings: Settings, destination: Path) -> None:
    from t_bot.store import Store
    from t_bot.supervisor import process_lock

    destination = destination.resolve()
    if destination.is_relative_to(settings.data_dir) or destination.is_relative_to(
        settings.extensions_dir
    ):
        raise BotError("Резервну копію слід зберігати поза робочими каталогами.")
    # Taking the worker lock guarantees registry, files and ADK DB form one stopped snapshot.
    with process_lock(settings.data_dir / "worker.lock"):
        destination.mkdir(parents=True, exist_ok=False, mode=0o700)
        store = Store(settings.data_dir / "state.sqlite3")
        try:
            store.backup(destination / "state.sqlite3")
        finally:
            store.close()
        source = settings.data_dir / "sessions.sqlite3"
        if source.exists():
            with (
                sqlite3.connect(source) as db,
                sqlite3.connect(destination / source.name) as target,
            ):
                db.backup(target)
        shutil.copytree(settings.extensions_dir, destination / "extensions")
        team_runs = settings.data_dir / "team_runs"
        if team_runs.exists():
            shutil.copytree(team_runs, destination / "team_runs")
        (destination / "metadata.json").write_text(
            json.dumps(
                {
                    "created": datetime.now(UTC).isoformat(),
                    "schema": 1,
                }
            )
        )
    print(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram + Google ADK + Ollama (без Docker)")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "worker", "doctor", "stop"):
        subcommands.add_parser(name)
    backup_parser = subcommands.add_parser("backup")
    backup_parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    configure_logging()
    try:
        settings = Settings()
        settings.prepare_directories()
        if args.command == "run":
            from t_bot.supervisor import supervise

            settings.require_telegram()
            code = asyncio.run(supervise(settings))
        elif args.command == "worker":
            from t_bot.application import run_worker

            code = asyncio.run(run_worker(settings))
        elif args.command == "doctor":
            code = asyncio.run(doctor(settings))
        elif args.command == "stop":
            from t_bot.supervisor import stop_supervisor

            stop_supervisor(settings)
            code = 0
        else:
            backup(settings, args.destination)
            code = 0
    except (BotError, ValueError, OSError) as exc:
        # Pydantic errors can include secret input values, so do not print validation reprs.
        print(
            str(exc)
            if isinstance(exc, BotError)
            else f"Помилка конфігурації/запуску: {type(exc).__name__}"
        )
        code = 1
    except KeyboardInterrupt:
        code = 0
    except Exception as exc:
        # Provider exceptions can embed tokens and URLs; report only their class at this boundary.
        print(f"Не вдалося запустити застосунок: {type(exc).__name__}")
        code = 1
    raise SystemExit(code)

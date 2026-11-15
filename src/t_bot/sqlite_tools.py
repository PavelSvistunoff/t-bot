"""SQLite workbench: read-only application databases and writable user databases."""

import base64
import json
import os
import re
import sqlite3
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from t_bot.errors import BotError

MAX_RESULT_CHARS = 12000
MAX_SQL_CHARS = 20000
MAX_VALUE_BYTES = 1048576
DATABASE_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}\.(db|sqlite|sqlite3)$", re.I)
READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}
READ_PRAGMAS = {
    "database_list",
    "schema_version",
    "user_version",
    "application_id",
    "encoding",
    "page_count",
    "freelist_count",
    "page_size",
    "journal_mode",
    "foreign_keys",
    "compile_options",
    "collation_list",
    "function_list",
    "module_list",
    "pragma_list",
    "table_list",
    "table_info",
    "table_xinfo",
    "index_list",
    "index_info",
    "index_xinfo",
    "foreign_key_list",
    "integrity_check",
    "quick_check",
}
ARGUMENT_READ_PRAGMAS = {
    "table_info",
    "table_xinfo",
    "index_list",
    "index_info",
    "index_xinfo",
    "foreign_key_list",
    "integrity_check",
    "quick_check",
    "table_list",
}
GLOBAL_PRAGMAS = {
    "temp_store_directory",
    "data_store_directory",
    "soft_heap_limit",
    "hard_heap_limit",
}


def json_value(value: object) -> object:
    if isinstance(value, bytes):
        return {
            "base64": base64.b64encode(value[:3072]).decode(),
            "bytes": len(value),
            "truncated": len(value) > 3072,
        }
    if isinstance(value, str) and len(value) > 4096:
        return {"text": value[:4096], "chars": len(value), "truncated": True}
    return value


def collect_rows(
    cursor: sqlite3.Cursor,
    max_rows: int,
    max_chars: int = MAX_RESULT_CHARS,
) -> tuple[list[list], bool]:
    rows: list[list] = []
    size = 0
    for row in cursor:
        values = [json_value(value) for value in row]
        size += len(json.dumps(values, ensure_ascii=False))
        if len(rows) == max_rows or size > max_chars:
            return rows, True
        rows.append(values)
    return rows, False


class SQLiteTools:
    def __init__(self, data_dir: Path, database_dir: Path, timeout_seconds: int, max_rows: int):
        self.database_dir = database_dir.resolve()
        self.system = {
            "state": data_dir / "state.sqlite3",
            "session": data_dir / "sessions.sqlite3",
        }
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows

    def _name(self, database: str) -> str:
        if not DATABASE_PATTERN.fullmatch(database):
            raise BotError(
                "Назва бази: літери, цифри, _ або - та розширення .db, .sqlite чи .sqlite3."
            )
        if Path(database).stem.casefold() in {"state", "session", "sessions"}:
            raise BotError("Імена state, session і sessions зарезервовані для службових баз.")
        return database

    def _path(self, database: str) -> tuple[Path, bool]:
        alias = database.casefold()
        if alias in {
            "state",
            "state.sqlite3",
            "session",
            "sessions",
            "session.sqlite3",
            "sessions.sqlite3",
        }:
            key = "state" if alias.startswith("state") else "session"
            path = self.system[key]
            read_only = True
        else:
            path = self.database_dir / self._name(database)
            read_only = False
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise BotError(
                "Базу не знайдено. Використайте list_databases або create_database."
            ) from exc
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            raise BotError("База має бути звичайним файлом без символьного посилання.")
        if not read_only:
            if path.parent.resolve() != self.database_dir:
                raise BotError("База виходить за межі SQLITE_DATABASE_DIR.")
            for protected in self.system.values():
                if protected.exists() and os.path.samefile(path, protected):
                    raise BotError("Службову базу можна відкривати лише через read-only ім’я.")
        return path, read_only

    @staticmethod
    def _authorizer(read_only: bool):
        def authorize(
            action: int,
            first: str | None,
            second: str | None,
            database: str | None,
            source: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_ATTACH:
                # VACUUM uses an unnamed internal database; VACUUM INTO/ATTACH name a host file.
                return sqlite3.SQLITE_OK if not read_only and first == "" else sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_DETACH:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION:
                if (second or "").casefold() in {"load_extension", "readfile", "writefile"}:
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_PRAGMA:
                pragma = (first or "").casefold()
                if pragma in GLOBAL_PRAGMAS:
                    return sqlite3.SQLITE_DENY
                if read_only and (
                    pragma not in READ_PRAGMAS
                    or (second is not None and pragma not in ARGUMENT_READ_PRAGMAS)
                ):
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            if read_only and action not in READ_ACTIONS:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        return authorize

    @contextmanager
    def _connect(self, database: str) -> Iterator[tuple[sqlite3.Connection, bool]]:
        path, read_only = self._path(database)
        uri = path.as_uri() + ("?mode=ro" if read_only else "?mode=rw")
        connection = sqlite3.connect(uri, uri=True, timeout=min(self.timeout_seconds, 5))
        try:
            if read_only:
                connection.execute("PRAGMA query_only=ON")
            else:
                connection.execute("PRAGMA foreign_keys=ON")
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_SQL_CHARS)
            deadline = time.monotonic() + self.timeout_seconds
            connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            connection.set_authorizer(self._authorizer(read_only))
            yield connection, read_only
        except sqlite3.Error as exc:
            connection.rollback()
            raise BotError(f"SQLite: {exc}") from exc
        finally:
            connection.close()

    def list_databases(self) -> dict:
        databases = [
            {"database": name, "read_only": True, "exists": path.is_file()}
            for name, path in self.system.items()
        ]
        if self.database_dir.exists():
            for path in sorted(self.database_dir.iterdir()):
                if len(databases) >= 102:
                    return {"databases": databases, "truncated": True}
                try:
                    _, read_only = self._path(path.name)
                    if read_only:
                        continue
                except BotError:
                    continue
                databases.append({"database": path.name, "read_only": False, "exists": True})
        return {"databases": databases, "truncated": False}

    def create_database(self, name: str) -> dict:
        name = self._name(name)
        self.database_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.database_dir / name
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError as exc:
            raise BotError("База вже існує; її не перезаписано.") from exc
        os.close(descriptor)
        try:
            with self._connect(name) as (connection, _):
                # Materialize a valid SQLite header without introducing a user table.
                connection.execute("PRAGMA user_version=0")
                connection.commit()
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return {"database": name, "read_only": False, "status": "created"}

    def database_schema(self, database: str) -> dict:
        with self._connect(database) as (connection, read_only):
            cursor = connection.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name",
            )
            rows, truncated = collect_rows(cursor, self.max_rows)
        return {
            "database": database,
            "read_only": read_only,
            "columns": ["type", "name", "table_name", "sql"],
            "objects": rows,
            "truncated": truncated,
        }

    def table_schema(self, database: str, table: str) -> dict:
        if not table or len(table) > 256 or "\x00" in table:
            raise BotError("Некоректне ім’я таблиці.")
        quoted = "'" + table.replace("'", "''") + "'"
        with self._connect(database) as (connection, read_only):
            found = connection.execute(
                "SELECT type,sql FROM sqlite_schema WHERE name=? AND type IN ('table','view')",
                (table,),
            ).fetchone()
            if found is None:
                raise BotError("Таблицю або view не знайдено.")
            sections = {}
            for key, pragma in (
                ("columns", "table_xinfo"),
                ("indexes", "index_list"),
                ("foreign_keys", "foreign_key_list"),
            ):
                cursor = connection.execute(f"PRAGMA main.{pragma}({quoted})")
                rows, truncated = collect_rows(cursor, self.max_rows, max_chars=2500)
                sections[key] = {
                    "fields": [column[0] for column in cursor.description],
                    "rows": rows,
                    "truncated": truncated,
                }
        return {
            "database": database,
            "table": table,
            "read_only": read_only,
            "type": found[0],
            "sql": json_value(found[1]),
            **sections,
        }

    def execute_sql(
        self, database: str, sql: str, parameters_json: str = "[]", max_rows: int = 100
    ) -> dict:
        if not 1 <= max_rows <= self.max_rows:
            raise BotError(f"max_rows має бути від 1 до {self.max_rows}.")
        if not sql.strip() or len(sql) > MAX_SQL_CHARS:
            raise BotError("SQL має містити від 1 до 20000 символів.")
        try:
            parameters = json.loads(parameters_json)
        except ValueError as exc:
            raise BotError("parameters_json має бути JSON-масивом або об’єктом.") from exc
        if not isinstance(parameters, list | dict):
            raise BotError("parameters_json має бути JSON-масивом або об’єктом.")
        try:
            with self._connect(database) as (connection, read_only):
                before = connection.total_changes
                cursor = connection.execute(sql, parameters)
                columns = [column[0] for column in cursor.description or []]
                column_size = len(json.dumps(columns, ensure_ascii=False))
                if column_size > MAX_RESULT_CHARS // 2:
                    raise BotError("Завеликі назви колонок результату. Зменште кількість колонок.")
                rows, truncated = (
                    collect_rows(cursor, max_rows, MAX_RESULT_CHARS - column_size)
                    if columns
                    else ([], False)
                )
                row_id = cursor.lastrowid
                cursor.close()
                changed = connection.total_changes - before
                connection.commit()
        except (TypeError, OverflowError) as exc:
            raise BotError("Непідтримуваний тип або розмір SQL-параметра.") from exc
        return {
            "database": database,
            "read_only": read_only,
            "columns": columns,
            "rows": rows,
            "truncated": truncated,
            "affected_rows": changed,
            "last_row_id": row_id,
        }

    def execute_sql_script(self, database: str, sql: str) -> dict:
        if not sql.strip() or len(sql) > MAX_SQL_CHARS:
            raise BotError("SQL-скрипт має містити від 1 до 20000 символів.")
        with self._connect(database) as (connection, read_only):
            if read_only:
                raise BotError("Для session/state дозволене лише читання через execute_sql.")
            before = connection.total_changes
            connection.executescript(sql)
            connection.commit()
            changes = connection.total_changes - before
        return {
            "database": database,
            "read_only": False,
            "affected_rows": changes,
            "status": "executed",
        }

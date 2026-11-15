"""Durable operations and an outbox; accepted updates are never executed twice."""

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(path, timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.transaction_depth = 0
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(
                id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, chat_id INTEGER NOT NULL,
                text TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                result TEXT, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS updates(id INTEGER PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS outbox(
                id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL,
                text TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS audit(
                id INTEGER PRIMARY KEY, created REAL NOT NULL, user_id INTEGER NOT NULL,
                action TEXT NOT NULL, details TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS extensions(
                id TEXT NOT NULL, version INTEGER NOT NULL, manifest TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 0, deleted INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(id, version));
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_extension
                ON extensions(id) WHERE active=1;
        """)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        depth = self.transaction_depth
        savepoint = f"nested_{depth}"
        self.connection.execute(f"SAVEPOINT {savepoint}" if depth else "BEGIN IMMEDIATE")
        self.transaction_depth += 1
        try:
            yield self.connection
        except BaseException:
            if depth:
                self.connection.execute(f"ROLLBACK TO {savepoint}")
                self.connection.execute(f"RELEASE {savepoint}")
            else:
                self.connection.rollback()
            raise
        else:
            if depth:
                self.connection.execute(f"RELEASE {savepoint}")
            else:
                self.connection.commit()
        finally:
            self.transaction_depth -= 1

    def close(self) -> None:
        self.connection.close()

    def get(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value: Any) -> None:
        with self.transaction() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, json.dumps(value)))

    def claim_update(self, update_id: int) -> bool:
        with self.transaction() as db:
            return (
                db.execute("INSERT OR IGNORE INTO updates VALUES (?)", (update_id,)).rowcount == 1
            )

    def enqueue(self, update_id: int, user_id: int, chat_id: int, text: str) -> bool:
        with self.transaction() as db:
            if not db.execute("INSERT OR IGNORE INTO updates VALUES (?)", (update_id,)).rowcount:
                return False
            db.execute(
                "INSERT INTO jobs(id,user_id,chat_id,text,created) VALUES (?,?,?,?,?)",
                (update_id, user_id, chat_id, text, time.time()),
            )
        return True

    def job(self, job_id: int) -> dict:
        row = self.connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def jobs(self, user_id: int, chat_id: int) -> list[dict]:
        rows = self.connection.execute(
            "SELECT id,status,result FROM jobs WHERE user_id=? AND chat_id=? "
            "ORDER BY id DESC LIMIT 10",
            (user_id, chat_id),
        )
        return [dict(row) for row in rows]

    def has_pending_jobs(self, user_id: int, chat_id: int) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM jobs WHERE user_id=? AND chat_id=? "
                "AND status IN ('queued','running') LIMIT 1",
                (user_id, chat_id),
            ).fetchone()
            is not None
        )

    def finish(self, job_id: int, status: str, result: str) -> None:
        with self.transaction() as db:
            db.execute("UPDATE jobs SET status=?,result=? WHERE id=?", (status, result, job_id))
            chat_id = db.execute("SELECT chat_id FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
            db.execute("INSERT INTO outbox(chat_id,text) VALUES (?,?)", (chat_id, result))

    def recover(self) -> None:
        rows = self.connection.execute(
            "SELECT id FROM jobs WHERE status IN ('queued','running')"
        ).fetchall()
        for row in rows:
            self.finish(
                row[0], "interrupted", f"Задачу {row[0]} перервано. Надішліть запит повторно."
            )

    def notify(self, chat_id: int, text: str) -> None:
        with self.transaction() as db:
            db.execute("INSERT INTO outbox(chat_id,text) VALUES (?,?)", (chat_id, text))

    def pending_messages(self) -> list[dict]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM outbox WHERE sent=0 ORDER BY id LIMIT 20"
            )
        ]

    def mark_sent(self, message_id: int) -> None:
        with self.transaction() as db:
            db.execute("UPDATE outbox SET sent=1 WHERE id=?", (message_id,))

    def audit(self, user_id: int, action: str, details: str) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO audit(created,user_id,action,details) VALUES (?,?,?,?)",
                (time.time(), user_id, action, details),
            )

    def backup(self, destination: Path) -> None:
        with sqlite3.connect(destination) as target:
            self.connection.backup(target)

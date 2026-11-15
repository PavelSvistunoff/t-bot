import json
import os
import sqlite3

import pytest

from t_bot.errors import BotError
from t_bot.sqlite_tools import SQLiteTools


@pytest.fixture
def workbench(settings, store):
    with sqlite3.connect(settings.data_dir / "sessions.sqlite3") as connection:
        connection.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, state TEXT)")
        connection.execute("INSERT INTO sessions VALUES ('one','{}')")
    return SQLiteTools(settings.data_dir, settings.data_dir / "databases", 2, 100)


def test_catalog_creation_and_full_sql_lifecycle(workbench):
    initial = workbench.list_databases()["databases"]
    assert {row["database"] for row in initial} == {"state", "session"}
    assert all(row["read_only"] for row in initial)
    assert workbench.create_database("notes.db")["read_only"] is False
    workbench.execute_sql("notes.db", "CREATE TABLE notes(id INTEGER PRIMARY KEY, text TEXT)")
    workbench.execute_sql("notes.db", "INSERT INTO notes(text) VALUES (?)", '["Привіт"]')
    workbench.execute_sql(
        "notes.db",
        "UPDATE notes SET text=:text WHERE id=:id",
        json.dumps({"text": "updated", "id": 1}),
    )
    assert workbench.execute_sql("notes.db", "SELECT text FROM notes")["rows"] == [["updated"]]
    workbench.execute_sql("notes.db", "ALTER TABLE notes ADD COLUMN extra INTEGER")
    workbench.execute_sql("notes.db", "CREATE INDEX by_text ON notes(text)")
    workbench.execute_sql("notes.db", "CREATE VIEW all_notes AS SELECT * FROM notes")
    workbench.execute_sql("notes.db", "PRAGMA user_version=7")
    assert workbench.execute_sql("notes.db", "PRAGMA user_version")["rows"] == [[7]]
    workbench.execute_sql("notes.db", "DELETE FROM notes")
    assert workbench.execute_sql("notes.db", "SELECT COUNT(*) FROM notes")["rows"] == [[0]]
    workbench.execute_sql("notes.db", "VACUUM")
    workbench.execute_sql("notes.db", "DROP VIEW all_notes")
    workbench.execute_sql("notes.db", "DROP TABLE notes")
    assert any(row["database"] == "notes.db" for row in workbench.list_databases()["databases"])
    with pytest.raises(BotError, match="вже існує"):
        workbench.create_database("notes.db")


def test_schema_tables_keys_indexes_and_views(workbench):
    workbench.create_database("schema.sqlite3")
    workbench.execute_sql_script(
        "schema.sqlite3",
        """
        CREATE TABLE parents(id INTEGER PRIMARY KEY);
        CREATE TABLE children(id INTEGER PRIMARY KEY, parent_id INTEGER NOT NULL,
                              FOREIGN KEY(parent_id) REFERENCES parents(id));
        CREATE INDEX by_parent ON children(parent_id);
        CREATE VIEW child_view AS SELECT * FROM children;
    """,
    )
    schema = workbench.database_schema("schema.sqlite3")
    assert {row[0] for row in schema["objects"]} >= {"table", "index", "view"}
    table = workbench.table_schema("schema.sqlite3", "children")
    assert table["columns"]["rows"][0][1] == "id"
    assert table["indexes"]["rows"][0][1] == "by_parent"
    assert table["foreign_keys"]["rows"][0][2] == "parents"
    assert workbench.table_schema("schema.sqlite3", "child_view")["type"] == "view"
    assert workbench.table_schema("state", "jobs")["read_only"]
    assert workbench.database_schema("session")["read_only"]


@pytest.mark.parametrize("database", ["state", "session", "state.sqlite3", "sessions.sqlite3"])
@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE hacked(x)",
        "PRAGMA user_version=77",
        "PRAGMA query_only=OFF",
        "PRAGMA writable_schema=ON",
        "VACUUM",
        "DELETE FROM sqlite_schema",
        "WITH x AS (SELECT 1) DELETE FROM sqlite_schema",
    ],
)
def test_system_databases_reject_writes_and_pragma_bypasses(workbench, database, sql):
    with pytest.raises(BotError):
        workbench.execute_sql(database, sql)
    assert workbench.execute_sql(database, "PRAGMA user_version")["rows"] == [[0]]
    assert not workbench.execute_sql(
        database,
        "SELECT name FROM sqlite_schema WHERE name='hacked'",
    )["rows"]


def test_system_databases_allow_queries_and_metadata(workbench):
    assert workbench.execute_sql("state", "SELECT count(*) FROM jobs")["rows"] == [[0]]
    assert workbench.execute_sql("session", "SELECT id FROM sessions")["rows"] == [["one"]]
    assert workbench.execute_sql("state", "PRAGMA table_info(jobs)")["rows"]
    assert workbench.execute_sql("state", "SELECT sqlite_version()")["rows"]
    with pytest.raises(BotError):
        workbench.execute_sql_script("state", "SELECT 1;")


def test_cross_database_file_access_is_rejected(workbench, settings, tmp_path):
    workbench.create_database("user.db")
    state = settings.data_dir / "state.sqlite3"
    for sql in (
        f"ATTACH DATABASE '{state}' AS system",
        f"VACUUM INTO '{tmp_path / 'outside.db'}'",
        "SELECT load_extension('anything')",
        "PRAGMA temp_store_directory='/tmp'",
        "PRAGMA hard_heap_limit=1",
    ):
        with pytest.raises(BotError):
            workbench.execute_sql("user.db", sql)
    assert not (tmp_path / "outside.db").exists()
    link = workbench.database_dir / "alias.db"
    link.symlink_to(state)
    with pytest.raises(BotError):
        workbench.execute_sql("alias.db", "DELETE FROM jobs")
    link.unlink()
    os.link(state, link)
    with pytest.raises(BotError):
        workbench.execute_sql("alias.db", "DELETE FROM jobs")


@pytest.mark.parametrize(
    "name",
    [
        "../escape.db",
        "/tmp/escape.db",
        "state.db",
        "session.sqlite3",
        "sessions.db",
        "file:state.sqlite3?mode=rw",
        "notes",
    ],
)
def test_database_names_do_not_escape_directory(workbench, name):
    with pytest.raises(BotError):
        workbench.create_database(name)


def test_result_limits_and_parameter_types(workbench):
    workbench.create_database("data.db")
    workbench.execute_sql_script(
        "data.db", "CREATE TABLE data(x); INSERT INTO data VALUES (1),(2),(3);"
    )
    result = workbench.execute_sql("data.db", "SELECT x FROM data ORDER BY x", max_rows=2)
    assert result["rows"] == [[1], [2]] and result["truncated"]
    blob = workbench.execute_sql("data.db", "SELECT X'0102'")["rows"][0][0]
    assert blob["base64"] == "AQI="
    with pytest.raises(BotError):
        workbench.execute_sql("data.db", "SELECT ?", '"not an array"')
    with pytest.raises(BotError):
        workbench.execute_sql("data.db", "SELECT 1", max_rows=101)


def test_script_transactions_rollback_on_error(workbench):
    workbench.create_database("tx.db")
    workbench.execute_sql("tx.db", "CREATE TABLE data(x)")
    with pytest.raises(BotError):
        workbench.execute_sql_script(
            "tx.db", "BEGIN; INSERT INTO data VALUES (1); SELECT * FROM missing;"
        )
    assert workbench.execute_sql("tx.db", "SELECT count(*) FROM data")["rows"] == [[0]]


def test_expensive_query_is_interrupted(workbench):
    workbench.timeout_seconds = 1
    with pytest.raises(BotError, match="interrupted"):
        workbench.execute_sql(
            "state",
            """WITH RECURSIVE n(x) AS (
            VALUES(1) UNION ALL SELECT x+1 FROM n
        ) SELECT sum(x) FROM n""",
        )


def test_dynamic_attach_cannot_bypass_system_readonly(workbench, settings):
    workbench.create_database("escape.db")
    state = str(settings.data_dir / "state.sqlite3")
    with pytest.raises(BotError):
        workbench.execute_sql("escape.db", "ATTACH DATABASE ? AS protected", json.dumps([state]))
    escaped = state.replace("'", "''")
    with pytest.raises(BotError):
        workbench.execute_sql_script(
            "escape.db",
            f"ATTACH DATABASE ('{escaped}' || '') AS protected; CREATE TABLE protected.hacked(x);",
        )
    assert not workbench.execute_sql("state", "SELECT name FROM sqlite_schema WHERE name='hacked'")[
        "rows"
    ]


def test_regular_system_tables_reject_dml(workbench):
    for statement in (
        "INSERT INTO settings VALUES ('intruder','1')",
        "UPDATE settings SET value='changed'",
        "DELETE FROM settings",
    ):
        with pytest.raises(BotError):
            workbench.execute_sql("state", statement)
    with pytest.raises(BotError):
        workbench.execute_sql("session", "UPDATE sessions SET state='changed'")
    assert workbench.execute_sql("session", "SELECT state FROM sessions")["rows"] == [["{}"]]


def test_truncated_returning_still_commits_all_changes(workbench):
    workbench.create_database("returning.db")
    workbench.execute_sql_script(
        "returning.db", "CREATE TABLE t(x); INSERT INTO t VALUES (1),(2),(3);"
    )
    result = workbench.execute_sql("returning.db", "DELETE FROM t RETURNING x", max_rows=1)
    assert result["truncated"] and result["affected_rows"] == 3
    assert workbench.execute_sql("returning.db", "SELECT COUNT(*) FROM t")["rows"] == [[0]]

import pytest
from pydantic import ValidationError

from t_bot.config import Settings
from t_bot.errors import BotError


def test_config_requires_explicit_access(settings):
    settings.require_telegram()
    with pytest.raises(BotError):
        settings.authorize(99)
    with pytest.raises(BotError):
        Settings(_env_file=None).require_telegram()
    with pytest.raises(ValidationError):
        Settings(_env_file=None, allowed_user_ids={-1})


def test_duplicate_update_and_recovery_are_durable(store):
    assert store.enqueue(100, 1, 5, "first")
    assert not store.enqueue(100, 1, 5, "duplicate")
    store.recover()
    assert store.job(100)["status"] == "interrupted"
    assert len(store.pending_messages()) == 1
    store.recover()
    assert len(store.pending_messages()) == 1
    assert not store.claim_update(100)


def test_jobs_and_settings_survive_reopen(store):
    from t_bot.store import Store

    store.set("model:child", "qwen3-coder:30b")
    store.enqueue(100, 1, 5, "one")
    store.enqueue(101, 2, 5, "two")
    reopened = Store(store.path)
    try:
        assert reopened.get("model:child") == "qwen3-coder:30b"
        assert [j["id"] for j in reopened.jobs(1, 5)] == [100]
        assert reopened.jobs(1, 6) == []
    finally:
        reopened.close()


def test_nested_transaction_is_atomic(store):
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.set("first", 1)
            store.notify(5, "must roll back")
            raise RuntimeError("crash")
    assert store.get("first") is None
    assert store.pending_messages() == []


@pytest.mark.parametrize("value", ["", "   ", "[]"])
def test_empty_allowed_users_environment_allows_everyone(value, monkeypatch):
    monkeypatch.setenv("ALLOWED_USER_IDS", value)
    settings = Settings(_env_file=None, telegram_bot_token="123:test")
    settings.require_telegram()
    settings.authorize(99)
    assert settings.allowed_user_ids == set()
    assert settings.allows(99)


def test_unset_allowed_users_allows_everyone(monkeypatch):
    monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
    settings = Settings(_env_file=None, telegram_bot_token="123:test")
    settings.require_telegram()
    settings.authorize(99)


def test_nonempty_allowed_users_environment_still_restricts_access(monkeypatch):
    monkeypatch.setenv("ALLOWED_USER_IDS", "[1, 2]")
    settings = Settings(_env_file=None)
    settings.authorize(2)
    with pytest.raises(BotError):
        settings.authorize(99)


@pytest.mark.parametrize("value", ["", "   ", "[]"])
def test_empty_dependencies_environment_is_unrestricted(value, monkeypatch):
    monkeypatch.setenv("ALLOWED_DEPENDENCIES", value)
    assert Settings(_env_file=None).allowed_dependencies == set()


def test_unset_dependencies_is_unrestricted(monkeypatch):
    monkeypatch.delenv("ALLOWED_DEPENDENCIES", raising=False)
    assert Settings(_env_file=None).allowed_dependencies == set()


def test_dependencies_from_dotenv_and_explicit_list(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOWED_DEPENDENCIES", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("ALLOWED_DEPENDENCIES =\n")
    assert Settings(_env_file=dotenv).allowed_dependencies == set()
    dotenv.write_text('ALLOWED_DEPENDENCIES=["six==1.17.0"]\n')
    assert Settings(_env_file=dotenv).allowed_dependencies == {"six==1.17.0"}

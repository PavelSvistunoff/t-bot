import sys

import pytest
from pydantic import ValidationError

from t_bot.cli import main
from t_bot.config import Settings


def test_team_large_budgets_are_valid_without_removing_overall_limits():
    settings = Settings(
        _env_file=None,
        extension_team_max_rounds=30,
        extension_team_stage_attempts=20,
        extension_team_max_calls=240,
        extension_team_timeout_seconds=9000,
    )
    assert settings.extension_team_max_rounds == 30
    assert settings.extension_team_stage_attempts == 20
    assert settings.extension_team_max_calls == 240
    assert settings.extension_team_timeout_seconds == 9000


@pytest.mark.parametrize("name", ["extension_team_max_rounds", "extension_team_stage_attempts"])
@pytest.mark.parametrize("value", [0, -1])
def test_team_iteration_limits_must_remain_positive(name, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{name: value})


def test_cli_names_invalid_settings_without_exposing_credentials(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("t_bot.cli.configure_logging", lambda: None)
    monkeypatch.setattr(sys, "argv", ["t-bot", "run"])
    monkeypatch.setenv("EXTENSION_TEAM_MAX_ROUNDS", "not-a-number-PRIVATE")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://user:TOP_SECRET@example.com")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "PRIVATE_BOT_TOKEN")
    with pytest.raises(SystemExit) as exit:
        main()
    assert exit.value.code == 1
    output = capsys.readouterr().out
    assert "EXTENSION_TEAM_MAX_ROUNDS" in output and "ціле число" in output
    assert "OLLAMA_BASE_URL" in output
    assert "TOP_SECRET" not in output and "PRIVATE" not in output

import pytest

from t_bot.errors import BotError


@pytest.mark.parametrize("allowed", [set(), {"six==1.17.0"}])
async def test_dependency_installation_allowed_for_empty_or_matching_list(
    components,
    settings,
    monkeypatch,
    allowed,
):
    _, _, sandbox, _ = components
    settings.allowed_dependencies = allowed
    commands = []

    async def install(command, **kwargs):
        commands.append(command)
        return b""

    monkeypatch.setattr("t_bot.sandbox.bounded_process", install)
    await sandbox.provision(["six==1.17.0"])
    assert commands and commands[0][-1] == "six==1.17.0"
    assert sandbox.dependency_path(["six==1.17.0"]).is_dir()
    await sandbox.provision(["six==1.17.0"])
    assert len(commands) == 1


async def test_dependency_installation_rejected_for_nonmatching_list(
    components, settings, monkeypatch
):
    _, _, sandbox, _ = components
    settings.allowed_dependencies = {"six==1.17.0"}

    async def unexpected_install(*args, **kwargs):
        pytest.fail("An unapproved package reached the installer")

    monkeypatch.setattr("t_bot.sandbox.bounded_process", unexpected_install)
    with pytest.raises(BotError, match="ALLOWED_DEPENDENCIES"):
        await sandbox.provision(["requests==2.34.2"])

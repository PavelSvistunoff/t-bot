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


@pytest.mark.parametrize("allowed", [set(), {"six==1.17.0"}])
async def test_install_tool_prepares_reusable_manifest_environment(
    components, settings, monkeypatch, allowed
):
    from t_bot.runtime import Invocation, current_invocation

    runtime, _, sandbox, _ = components
    settings.allowed_dependencies = allowed
    commands = []

    async def install(command, **kwargs):
        commands.append(command)
        return b""

    monkeypatch.setattr("t_bot.sandbox.bounded_process", install)
    token = current_invocation.set(Invocation(1, 5, 10))
    try:
        result = await runtime.basic_tools.install_dependencies(["six==1.17.0", "six==1.17.0"])
        assert result["status"] == "installed"
        assert result["dependencies"] == ["six==1.17.0"]
        assert sandbox.dependency_path(result["dependencies"]).is_dir()
        await runtime.basic_tools.install_dependencies(result["dependencies"])
        assert len(commands) == 1
        assert "--only-binary=:all:" in commands[0] and "--no-deps" in commands[0]
        assert "--target" in commands[0]
    finally:
        current_invocation.reset(token)


@pytest.mark.parametrize(
    "packages", [[], ["--target=/tmp"], ["six"], ["https://example.com/a.whl"], ["a==1"] * 11]
)
async def test_install_tool_rejects_invalid_requirements_before_pip(
    components, monkeypatch, packages
):
    from t_bot.runtime import Invocation, current_invocation

    runtime, _, _, _ = components

    async def unexpected(*args, **kwargs):
        pytest.fail("Invalid arguments reached pip")

    monkeypatch.setattr("t_bot.sandbox.bounded_process", unexpected)
    token = current_invocation.set(Invocation(1, 5, 10))
    try:
        assert "error" in await runtime.basic_tools.install_dependencies(packages)
    finally:
        current_invocation.reset(token)


async def test_install_tool_checks_user_allowlist_and_recovers_after_failure(
    components, settings, monkeypatch
):
    from t_bot.runtime import Invocation, current_invocation

    runtime, _, sandbox, _ = components
    calls = []

    async def fail_install(command, **kwargs):
        calls.append(command)
        raise BotError("Package unavailable")

    monkeypatch.setattr("t_bot.sandbox.bounded_process", fail_install)
    token = current_invocation.set(Invocation(99, 5, 10))
    try:
        with pytest.raises(BotError):
            await runtime.basic_tools.install_dependencies(["six==1.17.0"])
        assert not calls
        settings.allowed_user_ids = set()
        settings.allowed_dependencies = {"other==1"}
        assert "error" in await runtime.basic_tools.install_dependencies(["six==1.17.0"])
        assert not calls
        settings.allowed_dependencies = set()
        assert "error" in await runtime.basic_tools.install_dependencies(["six==1.17.0"])
        assert not sandbox.dependency_path(["six==1.17.0"]).exists()
        assert list(sandbox.dependencies_dir.iterdir()) == []

        async def success(command, **kwargs):
            return b""

        monkeypatch.setattr("t_bot.sandbox.bounded_process", success)
        assert (await runtime.basic_tools.install_dependencies(["six==1.17.0"]))[
            "status"
        ] == "installed"
    finally:
        current_invocation.reset(token)

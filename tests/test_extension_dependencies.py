import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from google.adk.tools import FunctionTool

from t_bot import sandbox
from t_bot.errors import BotError
from t_bot.manifest import Manifest
from t_bot.runtime import Invocation, current_invocation


@pytest.mark.sandbox
@pytest.mark.parametrize("generated_dependencies", [None, [], ["wrong==1"]])
async def test_explicit_dependencies_survive_generation_and_are_importable(
    components, monkeypatch, generated_dependencies
):
    runtime, extensions, _, _ = components
    manifest = Manifest(
        id="pillow_chart", kind="tool", version=1, description="Chart", example_usage="Draw chart"
    ).model_dump()
    if generated_dependencies is None:
        manifest.pop("dependencies")
    else:
        manifest["dependencies"] = generated_dependencies
    source = '''import PIL

def invoke(payload: dict) -> dict:
    """Verify an explicitly supplied package is importable during validation and invocation."""
    return {"version": PIL.__version__}
'''
    generate = AsyncMock(return_value=json.dumps({"source": source, "manifest": manifest}))
    monkeypatch.setattr(extensions.ollama, "generate", generate)

    async def install(command, **kwargs):
        target = Path(command[command.index("--target") + 1]) / "PIL"
        target.mkdir()
        (target / "__init__.py").write_text('__version__ = "10.3.0"')
        return b""

    # Fake only the package download; use the real sandbox/import/activation path.
    original = sandbox.bounded_process

    async def run(command, **kwargs):
        if "install" in command:
            return await install(command, **kwargs)
        return await original(command, **kwargs)

    monkeypatch.setattr("t_bot.sandbox.bounded_process", run)
    token = current_invocation.set(Invocation(1, 5, 10))
    try:
        result = await runtime.create_extension(
            "Draw a graph", "tool", "pillow_chart", dependencies=["Pillow==10.3.0"]
        )
    finally:
        current_invocation.reset(token)
    assert result["status"] == "active"
    assert result["manifest"]["dependencies"] == ["Pillow==10.3.0"]
    record = extensions.get("pillow_chart")
    assert await extensions.invoke(record, {}, 1) == {"version": "10.3.0"}
    schema = generate.call_args.args[2]
    assert schema["properties"]["manifest"]["properties"]["dependencies"] == {
        "const": ["Pillow==10.3.0"]
    }


@pytest.mark.parametrize("dependencies", [["--target=/tmp"], ["Pillow"], ["Pillow==10.3.0"]])
async def test_invalid_or_disallowed_explicit_dependencies_fail_before_generation(
    components, settings, monkeypatch, dependencies
):
    _, extensions, _, _ = components
    settings.allowed_dependencies = {"six==1.17.0"}
    generate = AsyncMock()
    monkeypatch.setattr(extensions.ollama, "generate", generate)
    with pytest.raises((BotError, ValueError)):
        await extensions.create("Chart", "tool", "chart", "root", "", 1, dependencies)
    generate.assert_not_awaited()
    assert not extensions.records()


async def test_create_extension_adk_schema_exposes_dependency_list(components):
    runtime, _, _, _ = components
    declaration = FunctionTool(runtime.create_extension)._get_declaration()
    assert "dependencies" in declaration.parameters_json_schema["properties"]


@pytest.mark.parametrize("dependencies,expected", [(None, ["six==1.17.0"]), ([], [])])
async def test_omitted_dependencies_allow_generation_but_empty_list_is_authoritative(
    components, monkeypatch, dependencies, expected
):
    from types import SimpleNamespace

    _, extensions, _, _ = components
    manifest = Manifest(
        id="generated",
        kind="tool",
        version=1,
        description="Test",
        example_usage="Test",
        dependencies=["six==1.17.0"],
    )
    generate = AsyncMock(
        return_value=json.dumps(
            {
                "source": "def invoke(payload: dict) -> dict: return {}",
                "manifest": manifest.model_dump(),
            }
        )
    )
    monkeypatch.setattr(extensions.ollama, "generate", generate)

    async def install(candidate, source, user_id):
        return SimpleNamespace(manifest=candidate)

    monkeypatch.setattr(extensions, "install", install)
    result = await extensions.create("Test", "tool", "generated", "root", "", 1, dependencies)
    assert result["manifest"]["dependencies"] == expected

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from t_bot.manifest import Manifest


@pytest.mark.parametrize("permissions,generated", [(["network"], []), ([], ["network"])])
async def test_explicit_network_policy_is_authoritative(
    components, monkeypatch, permissions, generated
):
    _, extensions, _, _ = components
    manifest = Manifest(
        id="weather",
        kind="tool",
        version=1,
        description="Weather",
        example_usage="Weather",
        permissions=generated,
    )
    generate = AsyncMock(
        return_value=json.dumps(
            {
                "manifest": manifest.model_dump(),
                "source": "def invoke(payload: dict) -> dict: return {}",
            }
        )
    )
    monkeypatch.setattr(extensions.ollama, "generate", generate)

    async def install(candidate, source, user_id):
        return SimpleNamespace(manifest=candidate)

    monkeypatch.setattr(extensions, "install", install)
    result = await extensions.create(
        "Weather",
        "tool",
        "weather",
        "root",
        "",
        1,
        permissions=permissions,
    )
    assert result["manifest"]["permissions"] == permissions
    assert generate.call_args.args[2]["properties"]["manifest"]["properties"]["permissions"] == {
        "const": permissions,
    }


async def test_unknown_permission_is_rejected_before_generation(components, monkeypatch):
    _, extensions, _, _ = components
    generate = AsyncMock()
    monkeypatch.setattr(extensions.ollama, "generate", generate)
    with pytest.raises(ValueError):
        await extensions.create("Weather", "tool", "weather", "root", "", 1, permissions=["shell"])
    generate.assert_not_awaited()

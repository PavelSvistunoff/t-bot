import pytest
from pydantic import ValidationError

from t_bot.errors import BotError
from t_bot.extensions import AGENT_TEMPLATE
from t_bot.manifest import Manifest

pytestmark = pytest.mark.sandbox
SOURCE = '''def invoke(payload: dict) -> dict:
    """Count words in text."""
    return {"count": len(payload["text"].split())}
'''


def manifest(version=1, **kwargs):
    return Manifest(
        id="word_count",
        version=version,
        kind="tool",
        description="Count words",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        example={"text": "two words"},
        example_usage="Count words in hello world",
        **kwargs,
    )


async def test_install_invoke_failed_update_and_rollback(components):
    runtime, extensions, _, _ = components
    record = await extensions.install(manifest(), SOURCE, 1)
    assert await extensions.invoke(record, {"text": "one two three"}, 1) == {"count": 3}
    with pytest.raises(BotError):
        await extensions.install(manifest(2), "def broken(:", 1)
    assert extensions.get("word_count").manifest.version == 1
    await extensions.install(manifest(2), SOURCE.replace('"count":', '"words":'), 1)
    assert extensions.get("word_count").manifest.version == 2
    await extensions.manage("word_count", "rollback", 1, 1)
    assert extensions.get("word_count").manifest.version == 1
    assert "word_count" in [
        tool.name for tool in runtime.build_agent().tools if hasattr(tool, "name")
    ]
    await extensions.manage("word_count", "disable", 1)
    assert extensions.records() == []
    await extensions.manage("word_count", "enable", 1, 2)
    await extensions.manage("word_count", "delete", 1)
    assert extensions.records(active_only=False) == []


async def test_adk_agent_factory_and_graph(components, monkeypatch):
    runtime, extensions, _, _ = components

    async def available(model):
        return None

    monkeypatch.setattr(extensions.ollama, "require_model", available)
    parent = Manifest(
        id="writer",
        kind="agent",
        version=1,
        entrypoint="build_agent",
        description="Writer",
        example_usage="Write text",
    )
    await extensions.install(parent, AGENT_TEMPLATE.replace("EXTENSION_ID", "writer"), 1)
    child = parent.model_copy(update={"id": "editor", "kind": "subagent", "parent": "writer"})
    await extensions.install(child, AGENT_TEMPLATE.replace("EXTENSION_ID", "editor"), 1)
    root = runtime.build_agent()
    writer = next(tool.agent for tool in root.tools if getattr(tool, "name", "") == "writer")
    assert any(getattr(tool, "name", "") == "editor" for tool in writer.tools)
    with pytest.raises(BotError):
        await extensions.manage("writer", "disable", 1)
    cyclic = parent.model_copy(update={"kind": "subagent", "parent": "editor", "version": 2})
    with pytest.raises(BotError):
        await extensions.install(cyclic, AGENT_TEMPLATE.replace("EXTENSION_ID", "writer"), 1)
    assert extensions.get("writer").manifest.version == 1


async def test_paths_permissions_and_tampering(components):
    _, extensions, _, _ = components
    with pytest.raises(ValidationError):
        manifest().model_validate({**manifest().model_dump(), "id": "../escape"})
    with pytest.raises(BotError):
        await extensions.install(manifest(), SOURCE, 99)
    record = await extensions.install(manifest(), SOURCE, 1)
    (extensions.directory(record.manifest) / "extension.py").write_text("# tampered")
    with pytest.raises(BotError, match="змінився"):
        await extensions.invoke(record, {"text": "x"}, 1)


async def test_unapproved_dependency_denied(components, settings):
    _, _, sandbox, _ = components
    settings.allowed_dependencies = {"six==1.17.0"}
    with pytest.raises(BotError, match="ALLOWED_DEPENDENCIES"):
        await sandbox.provision(["requests==2.34.2"])


async def test_output_contract_rejects_bad_version_without_replacing_active(components):
    _, extensions, _, _ = components
    await extensions.install(manifest(), SOURCE, 1)
    next_manifest = manifest(2).model_copy(
        update={
            "output_schema": {"type": "object", "required": ["missing_field"]},
        }
    )
    with pytest.raises(BotError):
        await extensions.install(next_manifest, SOURCE, 1)
    assert extensions.get("word_count").manifest.version == 1


def test_remote_json_schema_references_are_forbidden():
    with pytest.raises(ValidationError):
        Manifest.model_validate(
            {
                **manifest().model_dump(),
                "input_schema": {"type": "object", "$ref": "http://localhost/private"},
            }
        )

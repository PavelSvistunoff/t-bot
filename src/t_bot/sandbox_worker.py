"""Runs only in the restricted subprocess; the parent never imports extension code."""

import asyncio
import contextlib
import importlib.util
import inspect
import json
import mimetypes
import os
import resource
import sys
from pathlib import Path

import jsonschema

# Use standard MIME defaults; do not inspect host web-server configuration.
mimetypes.knownfiles = []
mimetypes.init(files=[])


def load_module(directory: str):
    spec = importlib.util.spec_from_file_location(
        "extension", str(Path(directory) / "extension.py")
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def agent_spec(agent, manifest: dict) -> dict:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    if not isinstance(agent, LlmAgent) or not isinstance(agent.model, LiteLlm):
        raise ValueError("Factory must return LlmAgent using LiteLlm")
    if agent.name != manifest["id"] or agent.model.model != "ollama_chat/" + manifest["model"]:
        raise ValueError("Agent name or model differs from manifest")
    if not isinstance(agent.instruction, str) or not agent.instruction.strip():
        raise ValueError("Agent instruction must be a nonempty string")
    if len(agent.instruction) > 16000:
        raise ValueError("Agent instruction is too large")
    # Execution hooks belong to host policy. Only a declarative ADK agent is exported.
    allowed = {
        "name",
        "description",
        "instruction",
        "model",
        "disallow_transfer_to_parent",
        "disallow_transfer_to_peers",
    }
    unsupported = agent.model_fields_set - allowed
    if unsupported or agent.tools or agent.sub_agents:
        raise ValueError(f"Use manifest parent/tools, not executable ADK hooks: {unsupported}")
    return {"name": agent.name, "description": agent.description, "instruction": agent.instruction}


async def run(request: dict) -> dict:
    if request["action"] == "probe":
        return {"ready": True}
    manifest = request["manifest"]
    module = load_module(request["directory"])
    function = getattr(module, manifest["entrypoint"])
    if manifest["kind"] != "tool":
        agent = function(manifest["model"], "http://127.0.0.1:11434")
        return agent_spec(agent, manifest)
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    if len(parameters) != 1 or parameters[0].annotation is inspect.Parameter.empty:
        raise ValueError("Tool must have one annotated dict argument")
    if signature.return_annotation is inspect.Signature.empty or not inspect.getdoc(function):
        raise ValueError("Tool needs return annotation and docstring")
    jsonschema.Draft202012Validator.check_schema(manifest["input_schema"])
    payload = manifest["example"] if request["action"] == "validate" else request["payload"]
    jsonschema.validate(payload, manifest["input_schema"])
    result = function(payload)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, dict):
        raise ValueError("Tool must return a JSON object")
    jsonschema.validate(result, manifest["output_schema"])
    json.dumps(result, allow_nan=False)
    return result


def main() -> None:
    request = json.loads(sys.stdin.buffer.read(128001))
    resource.setrlimit(resource.RLIMIT_CPU, (request["timeout"], request["timeout"] + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if sys.platform.startswith("linux"):
        limit = request["memory_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    os.environ["TMPDIR"] = str(Path(request["directory"]) / "tmp")
    os.environ["HOME"] = os.environ["TMPDIR"]
    sys.path.insert(0, request["dependencies"])
    # Keep the protocol channel separate from arbitrary extension print statements.
    with contextlib.redirect_stdout(sys.stderr):
        result = asyncio.run(run(request))
    sys.stdout.write(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

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
    sys.modules["extension"] = module
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


def run_tests(directory: str) -> dict:
    """Run QA's unittest suite; require executable happy/failure/edge assertions."""
    import ast
    import io
    import unittest

    path = Path(directory) / "tests.py"
    source = path.read_text()
    tree = ast.parse(source)
    assertions = sum(
        isinstance(node, ast.Assert)
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr.startswith("assert")
        )
        for node in ast.walk(tree)
    )
    spec = importlib.util.spec_from_file_location("extension_tests", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load QA tests")
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)

    def test_ids(node):
        if isinstance(node, unittest.TestSuite):
            return [item for child in node for item in test_ids(child)]
        return [node.id()]

    identifiers = test_ids(suite)
    coverage = {
        category: any(".test_" + category + "_" in name for name in identifiers)
        for category in ("happy", "failure", "edge")
    }
    if not all(coverage.values()) or assertions < 3 or not 3 <= len(identifiers) <= 100:
        return {
            "passed": False,
            "target": "qa",
            "coverage": coverage,
            "tests_run": 0,
            "feedback": "Require 3..100 unittest tests, real assertions and test_happy_*, "
            "test_failure_*, test_edge_* methods.",
        }
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=1).run(suite)
    return {
        "passed": result.wasSuccessful() and not result.skipped and not result.expectedFailures,
        "tests_run": result.testsRun,
        "coverage": coverage,
        "failures": [text[-4000:] for _, text in result.failures[:10]],
        "errors": [text[-4000:] for _, text in result.errors[:10]],
        "skipped": len(result.skipped),
        "output": output.getvalue()[-8000:],
    }


async def run(request: dict) -> dict:
    if request["action"] == "probe":
        return {"ready": True}
    manifest = request["manifest"]
    module = load_module(request["directory"])
    if request["action"] == "test":
        return run_tests(request["directory"])
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
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(Path(os.environ["TMPDIR"]) / "matplotlib")
    if request.get("filesystem_root"):
        os.environ["FILESYSTEM_ROOT"] = request["filesystem_root"]
        os.chdir(request["filesystem_root"])
    else:
        os.environ.pop("FILESYSTEM_ROOT", None)
    sys.path.insert(0, request["dependencies"])
    # Keep the protocol channel separate from arbitrary extension print statements.
    with contextlib.redirect_stdout(sys.stderr):
        result = asyncio.run(run(request))
    sys.stdout.write(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

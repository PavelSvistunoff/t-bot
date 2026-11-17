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
    """Run QA's unittest suite; require executable happy/failure assertions."""
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
        for category in ("happy", "failure")
    }
    if not all(coverage.values()) or assertions < 2 or not 2 <= len(identifiers) <= 100:
        return {
            "passed": False,
            "target": "qa",
            "coverage": coverage,
            "tests_run": 0,
            "feedback": "Require 2..100 unittest tests, real assertions and test_happy_*, "
            "test_failure_* methods.",
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


def symbolic_references(expressions: list[str]) -> dict:
    """Evaluate only a bounded math AST with exact SymPy constants, never sympify/eval strings."""
    import ast
    import math

    import sympy

    functions = {name: getattr(sympy, name) for name in ("sin", "cos", "tan", "log", "sqrt", "exp")}
    functions["abs"] = sympy.Abs

    def evaluate(node):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
        ):
            if not math.isfinite(node.value) or abs(node.value) > 10**100:
                raise ValueError("Constant exceeds budget")
            return (
                sympy.Integer(node.value)
                if type(node.value) is int
                else sympy.Float(str(node.value))
            )
        if isinstance(node, ast.Name) and node.id in {"pi", "e"}:
            return sympy.pi if node.id == "pi" else sympy.E
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            return evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if right.is_real is not True or abs(float(right)) > 1000:
                    raise ValueError("Exponent exceeds budget")
                if (
                    left.is_Integer
                    and right.is_positive
                    and int(abs(left)).bit_length() * float(right) > 4096
                ):
                    raise ValueError("Power exceeds budget")
                return left**right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in functions and not node.keywords and 1 <= len(node.args) <= 2:
                arguments = [evaluate(arg) for arg in node.args]
                if any(arg.is_real is not True or abs(float(arg)) > 1000 for arg in arguments):
                    raise ValueError("Function argument exceeds budget")
                return functions[node.func.id](*arguments)
        raise ValueError("Unsupported expression")

    if not isinstance(expressions, list) or len(expressions) > 128:
        raise ValueError("At most 128 reference expressions")
    results: dict[str, dict] = {}
    for expression in expressions:
        try:
            if not isinstance(expression, str) or not 1 <= len(expression) <= 512:
                raise ValueError("Expression exceeds budget")
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 128:
                raise ValueError("Expression exceeds budget")
            result = evaluate(tree.body)
            if result.is_real is not True or result.is_finite is not True:
                results[expression] = {"status": "domain_error"}
                continue
            value = int(result) if result.is_Integer else float(result.evalf(17))
            results[expression] = (
                {"status": "number", "value": value}
                if math.isfinite(value)
                else {"status": "domain_error"}
            )
        except (SyntaxError, ValueError, TypeError, ArithmeticError, RecursionError):
            results[expression] = {"status": "unknown"}
    return {"semantics": "sympy", "values": results}


def inspect_image(path: str) -> dict:
    """Decode pixels in the isolated worker; never trust the extension's file metadata."""
    import hashlib
    import stat

    from PIL import Image, ImageChops

    root = Path(os.environ["FILESYSTEM_ROOT"]).resolve()
    file = Path(path)
    file = (file if file.is_absolute() else root / file).resolve()
    if not file.is_relative_to(root):
        raise ValueError("Image must be saved inside FILESYSTEM_ROOT, not TMPDIR")
    info = file.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 8 * 1024 * 1024:
        raise ValueError("Image must be a bounded regular file")
    Image.MAX_IMAGE_PIXELS = 4_000_000
    with Image.open(file) as original:
        if original.width * original.height > 4_000_000:
            raise ValueError("Image exceeds pixel budget")
        format_name = original.format
        original.verify()
    with Image.open(file) as original:
        rgba = original.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
        image.load()
    background = Image.new("RGB", image.size, image.getpixel((0, 0)))
    bounds = ImageChops.difference(image, background).getbbox()
    return {
        "canvas_size": list(image.size),
        "format": format_name,
        "nonblank": bounds is not None,
        "content_bbox": list(bounds) if bounds else None,
        "pixel_digest": hashlib.sha256(image.tobytes()).hexdigest(),
    }


async def probe_images(module, manifest: dict) -> dict:
    """Sample controlled draws in disposable workspace and compare actual decoded pixels."""
    import random

    state = random.getstate()
    samples = []
    paths = set()
    try:
        for seed in (11, 29, 47):
            random.seed(seed)
            result = getattr(module, manifest["entrypoint"])(dict(manifest["example"]))
            if inspect.isawaitable(result):
                result = await result
            path = next(
                (result[key] for key in ("file_path", "image_path", "path") if key in result), None
            )
            if not isinstance(path, str) or result.get("error") or result.get("status") == "error":
                raise ValueError("Generator did not return a successful image path")
            samples.append(inspect_image(path))
            root = Path(os.environ["FILESYSTEM_ROOT"])
            paths.add(str((root / path).resolve()))
    finally:
        random.setstate(state)
    passed = (
        all(sample["nonblank"] for sample in samples)
        and len({sample["pixel_digest"] for sample in samples}) == 3
        and len(paths) == 3
    )
    return {
        "passed": passed,
        "target": "developer",
        "samples": samples,
        "distinct_paths": len(paths),
        "distinct_pixels": len({sample["pixel_digest"] for sample in samples}),
        "feedback": ""
        if passed
        else "Images must contain drawn pixels and differ across controlled seeds; "
        "unique filenames alone are insufficient.",
    }


async def run(request: dict) -> dict:
    if request["action"] == "probe":
        return {"ready": True}
    if request["action"] == "inspect_image":
        return inspect_image(request["payload"]["path"])
    if request["action"] == "reference":
        return symbolic_references(request["payload"]["expressions"])
    manifest = request["manifest"]
    module = load_module(request["directory"])
    if request["action"] == "image_probe":
        return await probe_images(module, manifest)
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

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from t_bot.errors import BotError
from t_bot.extension_team import SPEC_SCHEMA, TeamFailure
from t_bot.filesystem import FileSystem
from t_bot.image_quality import audit_image_tests, image_contract, validate_size_claims
from t_bot.manifest import Manifest

pytestmark = pytest.mark.team
REQUEST = (
    "Створи унікальне зображення кота через Pillow: випадкові колір хутра, розмір голови, "
    "вигляд вух, очей та вусів."
)
SOURCE = '''import os
import random
import uuid
from pathlib import Path
from PIL import Image, ImageDraw

def sample_traits():
    return {"fur": [random.randint(30, 220) for _ in range(3)], "size": random.randint(140, 220),
            "ears": random.randint(20, 55), "eyes": random.randint(4, 12),
            "whiskers": random.randint(15, 45)}

def invoke(payload: dict) -> dict:
    """Draw independently sampled cat features and return a persistent PNG."""
    try:
        traits = sample_traits()
        image = Image.new("RGB", (500, 500), "white")
        draw = ImageDraw.Draw(image)
        radius = traits["size"] // 2
        left, top, right, bottom = 250-radius, 250-radius, 250+radius, 250+radius
        color = tuple(traits["fur"])
        draw.ellipse((left, top, right, bottom), fill=color, outline="black")
        draw.polygon([(left+5, top+15), (left+30, top-traits["ears"]), (left+55, top+15)],
                     fill=color, outline="black")
        draw.polygon([(right-55, top+15), (right-30, top-traits["ears"]), (right-5, top+15)],
                     fill=color, outline="black")
        eye = traits["eyes"]
        for x in (220, 280):
            draw.ellipse((x-eye, 230-eye, x+eye, 230+eye), fill="black")
        draw.ellipse((245, 250, 255, 260), fill="pink")
        draw.line((220, 260, 220-traits["whiskers"], 250), fill="black", width=2)
        draw.line((280, 260, 280+traits["whiskers"], 250), fill="black", width=2)
        path = OUTPUT_ROOT / ("cat_" + uuid.uuid4().hex + ".png")
        image.save(path)
        return {"status": "success", "file_path": str(path), "traits": traits}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
'''
GOOD = SOURCE.replace("OUTPUT_ROOT", 'Path(os.environ["FILESYSTEM_ROOT"])')
BAD_TMP = SOURCE.replace("OUTPUT_ROOT", '__import__("tempfile").gettempdir_path()')
BAD_TMP = BAD_TMP.replace(
    '__import__("tempfile").gettempdir_path()', 'Path(__import__("tempfile").gettempdir())'
)


def specification():
    result = {
        name: "Procedural cat" if name == "objective" else ["Draw random cat features"]
        for name in SPEC_SCHEMA["required"]
    }
    result["happy_path"] = ["Випадкові параметри, наприклад розмір голови 200x200."]
    return result


def manifest():
    return Manifest(
        id="random_cat",
        kind="tool",
        version=1,
        description="Procedural cat",
        example_usage="Draw cat",
        dependencies=["pillow==10.3.0"],
        output_schema={
            "type": "object",
            "properties": {"file_path": {"type": "string"}, "error": {"type": "string"}},
        },
    ).model_dump()


def qa_tests():
    tests = """import unittest
import json
import random
from unittest.mock import patch
from PIL import Image
import extension

class CatTests(unittest.TestCase):
    def test_happy_decode(self):
        result = extension.invoke({})
        with Image.open(result["file_path"]) as image:
            image.load()
            self.assertGreater(image.width, 0)
            self.assertGreater(len(set(image.getdata())), 1)
    def test_failure_save(self):
        with patch("PIL.Image.Image.save", side_effect=OSError("no space")):
            self.assertEqual(extension.invoke({})["status"], "error")
"""
    variants = {
        "fur": [[50, 100, 150], [200, 80, 40]],
        "size": [140, 220],
        "ears": [20, 55],
        "eyes": [4, 12],
        "whiskers": [15, 45],
    }
    regions = {
        "fur": (245, 280, 255, 290),
        "size": (120, 120, 380, 380),
        "ears": (120, 70, 380, 190),
        "eyes": (200, 210, 300, 245),
        "whiskers": (150, 247, 350, 272),
    }
    for feature in variants:
        tests += f"""
    def test_edge_{feature}(self):
        sampled = []
        for seed in (11, 29, 47):
            random.seed(seed)
            sampled.append(json.dumps(extension.sample_traits()[{feature!r}]))
        self.assertGreater(len(set(sampled)), 1)
        pixels = []
        for value in {variants[feature]!r}:
            traits = {{"fur": [100, 100, 100], "size": 180, "ears": 20, "eyes": 5, "whiskers": 15}}
            traits[{feature!r}] = value
            with patch("extension.sample_traits", return_value=traits):
                result = extension.invoke({{}})
            with Image.open(result["file_path"]) as image:
                pixels.append(image.crop({regions[feature]!r}).tobytes())
        self.assertNotEqual(pixels[0], pixels[1])
"""
    return tests


async def prepare(sandbox, monkeypatch):
    pins = ["pillow==10.3.0"]
    sandbox.dependency_path(pins).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sandbox, "prepare_dependencies", AsyncMock(return_value=pins))


def research(team, monkeypatch):
    monkeypatch.setattr(
        team.web,
        "search",
        lambda *args: {"results": [{"title": "Pillow", "url": "https://pillow.readthedocs.io/"}]},
    )


def test_head_example_does_not_define_canvas_constraint():
    contract = image_contract(REQUEST, specification())
    assert contract["random_features"] == ["fur", "size", "ears", "eyes", "whiskers"]
    assert contract["geometry_facts"][0]["subject"] == "head"
    assert not contract["geometry_facts"][0]["binding"]
    claim = {"subject": "canvas", "size": [200, 200], "section": "happy_path", "index": 0}
    with pytest.raises(BotError, match="Полотно й голова"):
        validate_size_claims({"size_claims": [claim]}, contract)
    spec = specification()
    spec["requirements"] = ["Розмір полотна 500x500."]
    validate_size_claims(
        {"size_claims": [{**claim, "size": [500, 500], "section": "requirements"}]},
        image_contract(REQUEST, spec),
    )


def test_review_cannot_omit_evidence_for_a_canvas_requirement():
    contract = image_contract(REQUEST, specification())
    with pytest.raises(BotError, match="size_claims"):
        validate_size_claims(
            {"decision": "qa", "findings": ["Canvas must be 200x200"], "size_claims": []}, contract
        )


def test_explicit_canvas_and_example_head_in_same_sentence_stay_separate():
    spec = specification()
    spec["requirements"] = ["Розмір полотна 500x500, наприклад голова 200x200."]
    facts = image_contract(REQUEST, spec)["geometry_facts"]
    assert any(fact["subject"] == "canvas" and fact["binding"] for fact in facts)
    assert all(not fact["binding"] for fact in facts if fact["subject"] == "head")


def test_unrelated_words_do_not_invent_random_cat_features():
    assert (
        image_contract("Хочу унікальне зображення квітки, a clear image.", {})["random_features"]
        == []
    )


def test_path_only_tests_do_not_cover_random_features():
    weak = """class Tests:
    def test_happy(self):
        response = invoke({})
        self.assertTrue(response["file_path"].endswith(".png"))
"""
    contract = image_contract(REQUEST, specification())
    assert len(audit_image_tests(weak, contract)["findings"]) == 5
    assert audit_image_tests(qa_tests(), contract)["passed"]
    # Merely reading pixels without asserting anything about them is insufficient.
    unused = qa_tests().replace(
        "self.assertNotEqual(pixels[0], pixels[1])", "self.assertTrue(True)"
    )
    assert not audit_image_tests(unused, contract)["passed"]


@pytest.mark.sandbox
async def test_persistent_png_survives_invocation_and_is_readable_as_attachment(
    components, settings, monkeypatch, tmp_path
):
    _, _, sandbox, _ = components
    await prepare(sandbox, monkeypatch)
    directory = tmp_path / "code"
    directory.mkdir()
    (directory / "extension.py").write_text(GOOD)
    result = await sandbox.execute(directory, manifest(), "invoke", {})
    filename, data = FileSystem(settings.filesystem_root, 1000000, ()).read_attachment(
        result["file_path"], 1000000
    )
    assert filename.endswith(".png")
    assert await asyncio.to_thread(Path(result["file_path"]).exists)
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (500, 500)
    (directory / "extension.py").write_text(BAD_TMP)
    with pytest.raises(BotError, match="FILESYSTEM_ROOT"):
        await sandbox.execute(directory, manifest(), "invoke", {})


@pytest.mark.sandbox
async def test_distinct_filenames_with_identical_pixels_fail_native_probe(
    components, monkeypatch, tmp_path
):
    _, _, sandbox, _ = components
    await prepare(sandbox, monkeypatch)
    directory = tmp_path / "code"
    directory.mkdir()
    static = GOOD.replace(
        "traits = sample_traits()",
        'traits = {"fur": [100,100,100], "size":180, "ears":20, "eyes":5, "whiskers":15}',
    )
    (directory / "extension.py").write_text(static)
    result = await sandbox.execute(directory, manifest(), "image_probe", {})
    assert not result["passed"] and result["target"] == "developer"
    assert len({sample["pixel_digest"] for sample in result["samples"]}) == 1


@pytest.mark.sandbox
async def test_pipeline_repairs_tmp_path_weak_qa_and_canvas_head_confusion(
    components, settings, monkeypatch
):
    _, extensions, sandbox, _ = components
    await prepare(sandbox, monkeypatch)
    research(extensions.team, monkeypatch)
    settings.extension_team_fallback_models = ["qwen3-coder:30b"]
    counts = {"developer": 0, "qa": 0, "test_review": 0}

    async def ask(role, model, context, schema):
        if role == "Developer":
            counts["developer"] += 1
            return {"manifest": manifest(), "source": BAD_TMP if counts["developer"] == 1 else GOOD}
        if "queries" in schema["properties"]:
            return {"queries": ["Pillow drawing documentation"]}
        if "tests" in schema["properties"]:
            counts["qa"] += 1
            return {
                "tests": "import unittest\nclass Weak(unittest.TestCase):\n    pass"
                if counts["qa"] == 1
                else qa_tests(),
                "notes": "Pixel assertions",
            }
        if "decision" in schema["properties"]:
            if json.JSONDecoder().raw_decode(context)[0].get("phase") == "qa_expectations_review":
                counts["test_review"] += 1
                if counts["test_review"] == 1:
                    return {
                        "decision": "qa",
                        "findings": ["Canvas must be 200x200"],
                        "size_claims": [
                            {
                                "subject": "canvas",
                                "size": [200, 200],
                                "section": "happy_path",
                                "index": 0,
                            }
                        ],
                    }
                assert model == "qwen3-coder:30b"
            return {"decision": "approve", "findings": [], "size_claims": []}
        return specification()

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    result = await extensions.create(REQUEST, "tool", "random_cat", "root", "", 1)
    assert result["status"] == "active" and result["qa"]["tests_run"] == 7
    assert counts == {"developer": 2, "qa": 2, "test_review": 2}
    report = extensions.team.report(result["team_run_id"], 1)
    assert report["candidate"]["smoke_result"]["image"]["canvas_size"] == [500, 500]
    assert any("Полотно й голова" in event["data"].get("error", "") for event in report["events"])


@pytest.mark.sandbox
async def test_static_ear_feature_is_detected_by_executable_qa(components, settings, monkeypatch):
    _, extensions, sandbox, _ = components
    await prepare(sandbox, monkeypatch)
    research(extensions.team, monkeypatch)
    settings.extension_team_max_rounds = 1
    source = GOOD.replace('"ears": random.randint(20, 55)', '"ears": 20')

    async def ask(role, model, context, schema):
        if role == "Developer":
            return {"manifest": manifest(), "source": source}
        if "queries" in schema["properties"]:
            return {"queries": ["Pillow"]}
        if "tests" in schema["properties"]:
            return {"tests": qa_tests(), "notes": "Features"}
        if "decision" in schema["properties"]:
            return {
                "decision": "developer" if role == "QA" else "approve",
                "findings": [],
                "size_claims": [],
            }
        return specification()

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    with pytest.raises(TeamFailure) as failed:
        await extensions.create(REQUEST, "tool", "random_cat", "root", "", 1)
    executions = [
        event["data"] for event in failed.value.report["events"] if event["stage"] == "QAExecution"
    ]
    assert len(executions) == 1 and not executions[0]["passed"]
    assert any("test_edge_ears" in failure for failure in executions[0]["failures"])
    assert not extensions.records()

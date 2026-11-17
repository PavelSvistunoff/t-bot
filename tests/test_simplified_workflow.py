import json

import pytest

from t_bot.extension_team import SPEC_SCHEMA
from t_bot.manifest import Manifest

pytestmark = pytest.mark.team
SOURCE = '''def invoke(payload: dict) -> dict:
    """Double a valid integer or reject invalid input."""
    number = payload.get("number")
    if not isinstance(number, int) or isinstance(number, bool):
        raise ValueError("Expected integer")
    return {"doubled": number * 2}
'''
TESTS = """import unittest
import extension
class Tests(unittest.TestCase):
    def test_happy_double(self):
        self.assertEqual(extension.invoke({"number": 3})["doubled"], EXPECTED)
    def test_failure_invalid(self):
        with self.assertRaises(ValueError):
            extension.invoke({"number": "bad"})
"""


def artifact():
    return {
        "manifest": Manifest(
            id="simple_double",
            kind="tool",
            version=1,
            description="Double an integer",
            example_usage="Double 3",
            example={"number": 3},
            input_schema={
                "type": "object",
                "properties": {"number": {"type": "integer"}},
                "required": ["number"],
            },
        ).model_dump(),
        "source": SOURCE,
    }


def spec():
    return {
        **(
            {
                name: "Double integers" if name == "objective" else ["Preserve requirements"]
                for name in SPEC_SCHEMA["required"]
            }
        ),
        "source_checks": [
            {
                "conclusion": "Use official documentation",
                "url": "https://docs.python.org/3/",
                "quote": "Official Python documentation",
            }
        ],
    }


@pytest.mark.sandbox
@pytest.mark.parametrize(
    "mode",
    ["happy", "qa_repairs_own_expectation", "reviewer_requests_qa", "reviewer_grades_assertion"],
)
async def test_only_qa_owns_tests_and_reviewer_sees_developer_code(
    components, settings, monkeypatch, mode
):
    _, extensions, _, _ = components
    settings.extension_team_fallback_models = ["qwen3-coder:30b"]
    monkeypatch.setattr(
        extensions.team.web,
        "search",
        lambda *args: {"results": [{"title": "Python", "url": "https://docs.python.org/3/"}]},
    )
    counts = {"developer": 0, "qa": 0, "reviewer": 0}

    async def ask(role, model, context, schema):
        if role == "Developer":
            counts["developer"] += 1
            return artifact()
        if "queries" in schema["properties"]:
            return {"queries": ["Python integers"]}
        if "tests" in schema["properties"]:
            counts["qa"] += 1
            expected = "7" if mode == "qa_repairs_own_expectation" and counts["qa"] == 1 else "6"
            return {"tests": TESTS.replace("EXPECTED", expected), "notes": "QA owns assertions"}
        if "decision" in schema["properties"]:
            data = json.JSONDecoder().raw_decode(context)[0]
            if role == "QA":
                return {
                    "decision": "approve" if data["test_results"]["passed"] else "qa",
                    "findings": ["Correct my expected value"],
                }
            assert role == "Reviewer"
            counts["reviewer"] += 1
            assert (
                not {"tests", "test_results", "qa_notes", "qa_triage", "trusted_checks"}
                & data.keys()
            )
            assert set(data["implementation"]) == {"manifest", "source"}
            assert data["implementation"]["source"] == SOURCE
            assert "qa" not in schema["properties"]["decision"]["enum"]
            if counts["reviewer"] == 1 and mode == "reviewer_requests_qa":
                return {"decision": "qa", "findings": ["Rewrite test coverage"]}
            if counts["reviewer"] == 1 and mode == "reviewer_grades_assertion":
                return {
                    "decision": "developer",
                    "findings": ["test_happy_double should use assertAlmostEqual"],
                }
            return {"decision": "approve", "findings": []}
        assert role != "Supervisor", "Straightforward runs do not need LLM arbitration"
        return spec()

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    result = await extensions.create("Double an integer", "tool", "simple_double", "root", "", 1)
    assert result["status"] == "active" and result["qa"]["tests_run"] == 2
    assert result["qa"]["coverage"] == {"happy": True, "failure": True}
    assert counts["developer"] == 1
    assert counts["qa"] == (2 if mode == "qa_repairs_own_expectation" else 1)
    assert counts["reviewer"] == (2 if mode.startswith("reviewer_") else 1)
    report = extensions.team.report(result["team_run_id"], 1)
    assert "edge_cases" not in report["final_spec"]
    assert not any(event["stage"] in {"TestReview", "QAExpectations"} for event in report["events"])
    stages = [event["stage"] for event in report["events"]]
    assert stages.index("QAExecution") < stages.index("Reviewer")
    if mode == "qa_repairs_own_expectation":
        executions = [
            event["data"]["passed"] for event in report["events"] if event["stage"] == "QAExecution"
        ]
        assert executions == [False, True]


@pytest.mark.sandbox
@pytest.mark.parametrize("missing", ["happy", "failure"])
async def test_both_remaining_categories_are_required(components, tmp_path, missing):
    _, _, sandbox, _ = components
    directory = tmp_path / "qa"
    directory.mkdir()
    (directory / "extension.py").write_text(SOURCE)
    tests = TESTS.replace("EXPECTED", "6").replace(
        "def test_" + missing + "_", "def helper_" + missing + "_"
    )
    (directory / "tests.py").write_text(tests)
    result = await sandbox.execute(directory, artifact()["manifest"], "test", {})
    assert not result["passed"] and result["target"] == "qa"
    assert not result["coverage"][missing]
    assert "edge" not in result["coverage"]

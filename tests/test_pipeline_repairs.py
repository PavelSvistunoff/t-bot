import copy
import json
from unittest.mock import AsyncMock

import pytest

from t_bot.extension_team import SPEC_SCHEMA, TeamFailure
from t_bot.manifest import Manifest

pytestmark = pytest.mark.team
REQUEST = "Build a calculator using eval() and math sin/cos; return a number or error message."
SOURCE = '''import math

def invoke(payload: dict) -> dict:
    """Calculate a Python math expression or return an error description."""
    expression = payload.get("expression")
    if not isinstance(expression, str):
        return {"result": "Помилка: очікується рядок"}
    names = {name: getattr(math, name) for name in ("sin", "cos", "sqrt", "pi", "e")}
    try:
        return {"result": eval(expression, {"__builtins__": {}, **names}, {})}
    except (ArithmeticError, SyntaxError, NameError, ValueError, TypeError):
        return {"result": "Помилка обчислення"}
'''
TESTS = """import unittest
import extension

class CalculatorTests(unittest.TestCase):
    def test_happy_expression(self):
        payload = {"expression": "7+5+sin(7)-1+cos(5)"}
        result = extension.invoke(payload)
        self.assertAlmostEqual(result["result"], EXPECTED, places=5)
    def test_failure_syntax(self):
        self.assertIsInstance(extension.invoke({"expression": "5+*2"})["result"], str)
    def test_edge_invalid_type(self):
        self.assertIsInstance(extension.invoke({"expression": 123})["result"], str)
"""


def spec(bad=False):
    data = {
        key: "Calculator" if key == "objective" else ["Check behavior"]
        for key in SPEC_SCHEMA["required"]
    }
    if bad:
        data["failure_path"] = ["Синтаксична помилка Python: '5++2'"]
    return data


def candidate():
    manifest = Manifest(
        id="calc_repair",
        kind="tool",
        version=1,
        description="Calculator",
        example_usage="Calculate",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        output_schema={
            "type": "object",
            "properties": {"result": {"type": ["number", "string"]}},
            "required": ["result"],
        },
        example={"expression": "2+3"},
    ).model_dump()
    manifest["dependencies"] = ["math==3.12.0"]
    return {"manifest": manifest, "source": SOURCE}


def research(team, monkeypatch):
    monkeypatch.setattr(
        team.web,
        "search",
        lambda *args: {
            "results": [
                {
                    "url": "https://docs.python.org/3/library/math.html",
                    "title": "math",
                }
            ]
        },
    )


@pytest.mark.sandbox
async def test_report_case_repairs_pm_and_qa_without_rewriting_valid_code(components, monkeypatch):
    _, extensions, _, _ = components
    research(extensions.team, monkeypatch)
    counts = {"pm_revisions": 0, "developer": 0, "qa_write": 0}

    async def ask(role, model, context, schema):
        if role == "PM":
            if "current_spec" in context:
                counts["pm_revisions"] += 1
                assert "5++2" in context
                return spec()
            return spec(bad=True)
        if "queries" in schema["properties"]:
            return {"queries": ["Python math official documentation"]}
        if role == "Analyst":
            return spec(bad=True)
        if role == "Developer":
            counts["developer"] += 1
            return candidate()
        if role == "QA" and "tests" in schema["properties"]:
            counts["qa_write"] += 1
            expected = "12.534569" if counts["qa_write"] == 1 else "11.940648784182015"
            if counts["qa_write"] > 1:
                assert "11.940648784182015" in context
            return {"tests": TESTS.replace("EXPECTED", expected), "notes": "Check numeric oracle"}
        if role == "QA" and "test_results" in context:
            execution = json.JSONDecoder().raw_decode(context)[0]["test_results"]
            return {"decision": "approve" if execution["passed"] else "qa", "findings": []}
        return {"decision": "approve", "findings": []}

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    result = await extensions.create(REQUEST, "tool", "calc_repair", "root", "", 1)
    assert result["status"] == "active"
    assert counts == {"pm_revisions": 1, "developer": 1, "qa_write": 2}
    assert result["manifest"]["dependencies"] == []
    assert result["qa"]["tests_run"] == 3 and result["qa"]["passed"]
    report = extensions.team.report(result["team_run_id"], 1)
    assert len(report["spec_revisions"]) == 2
    assert report["last_valid_candidate"]["source"] == SOURCE
    assert any(e["stage"] == "QAExecution" and not e["data"]["passed"] for e in report["events"])


async def test_invalid_attempt_does_not_replace_last_valid_candidate(
    components, settings, monkeypatch
):
    _, extensions, _, _ = components
    settings.extension_team_max_rounds = 2
    research(extensions.team, monkeypatch)
    attempts = []

    async def ask(role, model, context, schema):
        if role == "Developer":
            result = candidate()
            result["manifest"]["dependencies"] = []
            attempts.append(result)
            if len(attempts) > 1:
                result = copy.deepcopy(result)
                result["source"] += "\n# invalid new proposal"
            return result
        if "queries" in schema["properties"]:
            return {"queries": ["Python docs"]}
        if "tests" in schema["properties"]:
            return {"tests": TESTS.replace("EXPECTED", "11.940648784182015"), "notes": "Tests"}
        if "decision" in schema["properties"]:
            data = json.JSONDecoder().raw_decode(context)[0]
            decision = "approve" if data.get("phase") == "qa_expectations_review" else "developer"
            return {"decision": decision, "findings": ["Fix the failing implementation assertion"]}
        return spec()

    async def validate(value):
        if len(attempts) > 1:
            raise ValueError("Invalid new manifest")
        return copy.deepcopy(value)

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    with pytest.raises(TeamFailure) as failed:
        await extensions.team.run(
            request=REQUEST,
            extension_id="calc_repair",
            user_id=1,
            model="fake",
            developer_context="Implement",
            developer_schema={},
            validate=validate,
            execute_tests=AsyncMock(
                return_value={"passed": False, "failures": ["AssertionError: bug"]}
            ),
        )
    report = extensions.team.report(failed.value.report["id"], 1)
    assert report["candidate"]["source"] == SOURCE
    assert report["last_valid_candidate"]["source"] == SOURCE
    assert report["attempt_candidate"]["source"].endswith("# invalid new proposal")


async def test_repeated_unresolved_failures_stop_before_round_limit(
    components, settings, monkeypatch
):
    _, extensions, _, _ = components
    settings.extension_team_max_rounds = 30
    settings.extension_team_max_calls = 100
    settings.extension_team_stagnation_limit = 2
    research(extensions.team, monkeypatch)

    async def ask(role, model, context, schema):
        if role == "Developer":
            return candidate()
        if "queries" in schema["properties"]:
            return {"queries": ["Python docs"]}
        if "tests" in schema["properties"]:
            return {"tests": TESTS.replace("EXPECTED", "11.940648784182015"), "notes": "Tests"}
        if "decision" in schema["properties"]:
            data = json.JSONDecoder().raw_decode(context)[0]
            return {
                "decision": "approve"
                if data.get("phase") == "qa_expectations_review"
                else "developer",
                "findings": ["Fix the failing implementation assertion"],
            }
        return spec()

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    execute = AsyncMock(return_value={"passed": False, "failures": ["AssertionError: same defect"]})
    with pytest.raises(TeamFailure) as failed:
        await extensions.team.run(
            request=REQUEST,
            extension_id="calc_repair",
            user_id=1,
            model="fake",
            developer_context="Implement",
            developer_schema={},
            validate=AsyncMock(side_effect=lambda value: value),
            execute_tests=execute,
        )
    assert "Зациклення" in failed.value.report["error"]
    assert execute.await_count == 6
    assert failed.value.report["last_valid_candidate"]
    assert [
        e["data"]["target"] for e in failed.value.report["events"] if e["stage"] == "Stagnation"
    ] == ["qa", "pm", "pm"]


@pytest.mark.sandbox
@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_pm_semantic_failure_uses_fallback_instead_of_repeating_report_loop(
    components,
    settings,
    monkeypatch,
    repair_succeeds,
):
    _, extensions, _, _ = components
    research(extensions.team, monkeypatch)
    settings.extension_team_fallback_models = ["qwen3-coder:30b"]
    revisions = []
    bad = spec(bad=True)
    bad["acceptance_criteria"] = ["Вираз 'sqrt(16) + (2 * 3)' повертає 8.0."]
    good = copy.deepcopy(bad)
    good["failure_path"] = ["Синтаксична помилка Python: '5+*2'"]
    good["acceptance_criteria"] = ["Вираз 'sqrt(16) + (2 * 3)' повертає 10.0."]

    async def ask(role, model, context, schema):
        if role == "PM":
            if "current_spec" not in context:
                return copy.deepcopy(bad)
            revisions.append(model)
            if len(revisions) == 1:
                # The report's failure: add a true requirement, leave the false examples intact.
                result = copy.deepcopy(bad)
                result["requirements"].append("5++2 is valid and returns 7")
                return result
            assert "PREVIOUS ATTEMPT FAILED" in context
            assert "failure_path[0]" in context
            assert "corrections" in context and "10.0" in context
            return copy.deepcopy(good if repair_succeeds else bad)
        if "queries" in schema["properties"]:
            return {"queries": ["Python math documentation"]}
        if role == "Analyst":
            return copy.deepcopy(bad)
        if role == "Developer":
            final_task = context.split("\nFINAL TECHNICAL TASK:\n", 1)[1]
            final_task = final_task.split("\nREPOSITORY CLEAN CODE RULES:\n", 1)[0]
            assert json.loads(final_task) == good
            return candidate()
        if "tests" in schema["properties"]:
            return {"tests": TESTS.replace("EXPECTED", "11.940648784182015"), "notes": "Tests"}
        if role == "QA" and "test_results" in context:
            execution = json.JSONDecoder().raw_decode(context)[0]["test_results"]
            return {"decision": "approve" if execution["passed"] else "qa", "findings": []}
        return {"decision": "approve", "findings": []}

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    arguments = (
        "Create a scientific calculator supporting sin, cos, sqrt and arithmetic expressions.",
        "tool",
        "calc_repair",
        "root",
        "",
        1,
    )
    if not repair_succeeds:
        settings.extension_team_max_rounds = 30
        with pytest.raises(TeamFailure) as failed:
            await extensions.create(*arguments)
        report = failed.value.report
        assert report["calls"] == 6
        assert "PM: усі спроби" in report["error"]
        assert not report.get("candidate")
        assert not any(event["stage"] == "Developer" for event in report["events"])
        assert revisions == [settings.default_model, "qwen3-coder:30b"]
        return
    result = await extensions.create(*arguments)
    assert result["status"] == "active" and result["qa"]["passed"]
    assert revisions == [settings.default_model, "qwen3-coder:30b"]
    report = extensions.team.report(result["team_run_id"], 1)
    assert report["final_spec"] == good
    assert len(report["spec_revisions"]) == 2
    assert not any(event["stage"] == "Stagnation" for event in report["events"])
    assert any(
        event["stage"] == "PM" and "ТЗ не виправлено" in event["data"].get("error", "")
        for event in report["events"]
    )


@pytest.mark.sandbox
async def test_ukrainian_spec_and_dynamic_oracle_repairs_activate_valid_candidate(
    components,
    monkeypatch,
):
    _, extensions, _, _ = components
    research(extensions.team, monkeypatch)
    bad = spec()
    bad["happy_path"] = ["Користувач вводить '2 + 2 * 5', інструмент повертає 15."]
    bad["failure_path"] = ["Невалідний синтаксис '5 ++ 2' повертає помилку."]
    good = copy.deepcopy(bad)
    good["happy_path"][0] = good["happy_path"][0].replace("15", "12")
    good["failure_path"][0] = good["failure_path"][0].replace("5 ++ 2", "5+*2")
    counts = {"developer": 0, "qa": 0, "pm_revision": 0}

    async def ask(role, model, context, schema):
        if role == "PM":
            if "current_spec" in context:
                counts["pm_revision"] += 1
                return copy.deepcopy(good)
            return copy.deepcopy(bad)
        if "queries" in schema["properties"]:
            return {"queries": ["Python math documentation"]}
        if role == "Analyst":
            return copy.deepcopy(bad)
        if role == "Developer":
            counts["developer"] += 1
            return candidate()
        if "tests" in schema["properties"]:
            counts["qa"] += 1
            expected = "7 + 5 + math.sin(7) - 1 + math.cos(5)"
            if counts["qa"] == 1:
                expected += " + 1"
            return {"tests": "import math\n" + TESTS.replace("EXPECTED", expected), "notes": "Math"}
        if role == "QA" and "test_results" in context:
            execution = json.JSONDecoder().raw_decode(context)[0]["test_results"]
            return {"decision": "approve" if execution["passed"] else "qa", "findings": []}
        return {"decision": "approve", "findings": []}

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    result = await extensions.create(
        "Створи інструмент для обчислення складних математичних виразів із функціями math.",
        "tool",
        "calc_repair",
        "root",
        "",
        1,
    )
    assert result["status"] == "active" and result["qa"]["tests_run"] == 3
    assert counts == {"developer": 1, "qa": 2, "pm_revision": 1}

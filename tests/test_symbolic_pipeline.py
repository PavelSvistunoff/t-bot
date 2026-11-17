import copy
import json
from unittest.mock import AsyncMock

import pytest

from t_bot.candidate_validation import verify_smoke_result
from t_bot.extension_team import SPEC_SCHEMA, TeamFailure
from t_bot.manifest import Manifest
from t_bot.qa_expectations import NoProgressGuard, audit_tests
from t_bot.sandbox_worker import symbolic_references

pytestmark = pytest.mark.team
REQUEST = "Створи інструмент для обчислення математичних виразів через SymPy."


def specification():
    return {
        name: "Math calculator" if name == "objective" else ["Math requirements"]
        for name in SPEC_SCHEMA["required"]
    }


def artifact():
    return {
        "manifest": Manifest(
            id="symbolic_calc",
            kind="tool",
            version=1,
            description="Calculator",
            example_usage="Calculate sqrt(16)",
            dependencies=["sympy==1.13.0"],
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            output_schema={
                "type": "object",
                "properties": {"result": {"type": "number"}, "error": {"type": "string"}},
            },
            example={"text": "sqrt(16)"},
        ).model_dump(),
        "source": '''import sympy

def invoke(payload: dict) -> dict:
    """Evaluate controlled regression cases using exact SymPy values."""
    values = {"sqrt(16)": sympy.sqrt(16), "tan(pi/2)": sympy.tan(sympy.pi/2)}
    result = values.get(payload.get("text"), sympy.nan).evalf()
    try:
        BROKEN_CHECK
        if result.is_real is not True or result.is_finite is not True:
            return {"error": "domain error"}
        return {"result": float(result)}
    except Exception as exc:
        return {"error": str(exc)}
''',
    }


TESTS = """import unittest
import extension

class RegressionTests(unittest.TestCase):
    def test_happy_number(self):
        response = extension.invoke({"text": "sqrt(16)"})
        self.assertEqual(response["result"], 4.0)
    def test_failure_symbolic_singularity(self):
        response = extension.invoke({"text": "tan(pi/2)"})
        self.assertIn("error", response)
    def test_edge_empty_expression(self):
        self.assertIn("error", extension.invoke({"text": ""}))
"""


def research(team, monkeypatch):
    monkeypatch.setattr(
        team.web,
        "search",
        lambda *args: {
            "results": [{"title": "SymPy", "url": "https://docs.sympy.org/latest/"}],
        },
    )


def test_exact_sympy_domain_and_constants_are_used_without_eval(tmp_path):
    marker = tmp_path / "not-written"
    unsafe = f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')"
    results = symbolic_references(
        [
            "tan(pi/2)",
            "sqrt(-1)",
            "1/0",
            "log(e)",
            "sqrt(16)",
            "sin(pi)",
            "7+5+sin(7)-1+cos(5)",
            "2**(2**100)",
            unsafe,
        ]
    )["values"]
    assert all(
        results[expression]["status"] == "domain_error"
        for expression in ("tan(pi/2)", "sqrt(-1)", "1/0")
    )
    assert results["log(e)"]["value"] == 1
    assert results["sqrt(16)"]["value"] == 4
    assert results["sin(pi)"]["value"] == 0
    assert results["7+5+sin(7)-1+cos(5)"]["value"] == pytest.approx(11.940648784182015)
    assert results["2**(2**100)"]["status"] == results[unsafe]["status"] == "unknown"
    assert not marker.exists()


def test_audit_does_not_replace_symbolic_singularity_with_float_approximation():
    oracle = symbolic_references(["sqrt(16)", "tan(pi/2)"])
    audit = audit_tests(
        REQUEST, specification(), TESTS, semantics="sympy", references=oracle["values"]
    )
    assert audit["passed"]
    assert audit["reference_checks"][0]["reference"] == 4
    assert audit["semantics"] == "sympy"
    # Missing reference data is unknown, never a claim that tan(pi/2) is a valid finite value.
    assert audit_tests(REQUEST, specification(), TESTS)["passed"]


@pytest.mark.parametrize(
    "result", [{"error": "Float has no attribute is_nan"}, {"success": False}, {"status": "failed"}]
)
def test_error_object_is_not_a_successful_example(result):
    from t_bot.errors import BotError

    with pytest.raises(BotError, match="Happy-path"):
        verify_smoke_result(REQUEST, artifact()["manifest"], result, {"semantics": "sympy"})


def test_new_milestone_resets_no_progress_but_artifact_changes_do_not():
    guard = NoProgressGuard(2)
    assert guard.observe(1) is None
    assert guard.observe(1) is None
    assert guard.observe(2) is None
    assert guard.observe(2) is None
    assert guard.observe(2) == {"failed_rounds": 2, "milestone": 2}
    assert guard.observe(3) is None


@pytest.mark.sandbox
async def test_is_nan_error_is_repaired_before_qa_and_symbolic_tests_pass(components, monkeypatch):
    _, extensions, sandbox, _ = components
    research(extensions.team, monkeypatch)
    # Use the pinned dev SymPy in the real worker, replacing only package download.
    pins = ["sympy==1.13.0"]
    sandbox.dependency_path(pins).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sandbox, "prepare_dependencies", AsyncMock(return_value=pins))
    counts = {"developer": 0, "qa": 0}

    async def ask(role, model, context, schema):
        if role == "Developer":
            counts["developer"] += 1
            candidate = artifact()
            candidate["source"] = candidate["source"].replace(
                "BROKEN_CHECK",
                "result.is_nan" if counts["developer"] == 1 else "pass",
            )
            return candidate
        if "queries" in schema["properties"]:
            return {"queries": ["SymPy finite number"]}
        if "tests" in schema["properties"]:
            counts["qa"] += 1
            assert counts["developer"] == 2
            return {"tests": TESTS, "notes": "Exact symbolic domain"}
        if "decision" in schema["properties"]:
            data = json.loads(context)
            if data.get("phase") == "qa_expectations_review":
                assert data["trusted_checks"]["semantics"] == "sympy"
            return {"decision": "approve", "findings": []}
        return specification()

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    result = await extensions.create(REQUEST, "tool", "symbolic_calc", "root", "", 1)
    assert result["status"] == "active" and result["qa"]["tests_run"] == 3
    assert counts == {"developer": 2, "qa": 1}
    report = extensions.team.report(result["team_run_id"], 1)
    validations = [event["data"] for event in report["events"] if event["stage"] == "Validation"]
    assert "is_nan" in validations[0]["validation_error"]
    assert validations[1]["smoke_result"]["numeric_result"] == 4
    assert report["last_valid_candidate"]["source"].count("is_nan") == 0


async def test_changed_candidates_and_tests_cannot_hide_global_stall(
    components, settings, monkeypatch
):
    _, extensions, _, _ = components
    research(extensions.team, monkeypatch)
    settings.extension_team_max_rounds = 30
    settings.extension_team_max_calls = 100
    settings.extension_team_no_progress_limit = 3
    counts = {"developer": 0, "qa": 0, "review": 0}

    async def ask(role, model, context, schema):
        if role == "Developer":
            counts["developer"] += 1
            candidate = artifact()
            candidate["manifest"]["dependencies"] = []
            candidate["source"] = f"VERSION = {counts['developer']}\n" + candidate[
                "source"
            ].replace("BROKEN_CHECK", "pass")
            return candidate
        if "queries" in schema["properties"]:
            return {"queries": ["Math documentation"]}
        if "tests" in schema["properties"]:
            counts["qa"] += 1
            return {
                "tests": f"QA_VERSION = {counts['qa']}\n" + TESTS.replace("tan(pi/2)", "1/0"),
                "notes": "Tests",
            }
        if "decision" in schema["properties"]:
            counts["review"] += 1
            return {"decision": "pm", "findings": [f"Different wording {counts['review']}"]}
        return specification()

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    execute = AsyncMock()
    with pytest.raises(TeamFailure) as failed:
        await extensions.team.run(
            request=REQUEST,
            extension_id="symbolic_calc",
            user_id=1,
            model="fake",
            developer_context="Implement",
            developer_schema={},
            validate=AsyncMock(side_effect=lambda value: copy.deepcopy(value)),
            execute_tests=execute,
        )
    report = failed.value.report
    assert "Немає прогресу" in report["error"]
    assert counts == {"developer": 4, "qa": 4, "review": 4}
    assert not any(event["stage"] == "Stagnation" for event in report["events"])
    assert report["last_valid_candidate"]
    assert report["stage"] == "NoProgress"
    assert execute.await_count == 0


def test_number_expectation_for_symbolic_singularity_is_rejected():
    tests = """class Tests:
    def test_edge_singularity(self):
        response = invoke({"text": "tan(pi/2)"})
        self.assertEqual(response["result"], 1.633123935319537e16)
"""
    oracle = symbolic_references(["tan(pi/2)"])
    audit = audit_tests(REQUEST, specification(), tests, references=oracle["values"])
    assert not audit["passed"] and audit["target"] == "qa"
    assert "SymPy" in audit["findings"][0]


def test_smoke_checks_numeric_value_independently_of_result_field_name():
    manifest = artifact()["manifest"]
    oracle = symbolic_references(["sqrt(16)"])
    assert verify_smoke_result(REQUEST, manifest, {"answer": 4}, oracle)["numeric_result"] == 4
    from t_bot.errors import BotError

    with pytest.raises(BotError, match="контрольний результат"):
        verify_smoke_result(REQUEST, manifest, {"result": 5}, oracle)

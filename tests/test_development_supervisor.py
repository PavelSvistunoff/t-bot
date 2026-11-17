import copy
import json
from unittest.mock import AsyncMock

import pytest

from t_bot.development_supervisor import DevelopmentSupervisor, validate_advice
from t_bot.errors import BotError
from t_bot.extension_team import SPEC_SCHEMA, TeamFailure
from t_bot.manifest import Manifest

pytestmark = pytest.mark.team
SOURCE = '''def invoke(payload: dict) -> dict:
    """Double every integer, including zero."""
    number = payload.get("number")
    if not isinstance(number, int) or isinstance(number, bool):
        raise ValueError("Expected integer")
    return {"doubled": number * 2}
'''
TESTS = """import unittest
import extension
class Tests(unittest.TestCase):
    def test_happy_positive(self):
        self.assertEqual(extension.invoke({"number": 3})["doubled"], 6)
    def test_failure_string(self):
        with self.assertRaises(ValueError):
            extension.invoke({"number": "three"})
    def test_edge_negative(self):
        self.assertEqual(extension.invoke({"number": -1})["doubled"], -2)
"""


def candidate(source=SOURCE):
    return {
        "source": source,
        "manifest": Manifest(
            id="supervised_double",
            kind="tool",
            version=1,
            description="Double integers",
            example_usage="Double 3",
            example={"number": 3},
            input_schema={
                "type": "object",
                "properties": {"number": {"type": "integer"}},
                "required": ["number"],
            },
        ).model_dump(),
    }


def spec():
    return {
        key: "Double every integer including zero"
        if key == "objective"
        else ["Preserve the requested behavior"]
        for key in SPEC_SCHEMA["required"]
    }


def advice(action="reviewer", pointer="/test_results/passed", quote="true"):
    return {
        "action": action,
        "reason": "Resolve the disputed finding using actual evidence",
        "evidence": [{"pointer": pointer, "quote": quote}],
    }


@pytest.mark.parametrize(
    "invalid",
    [
        advice("activate"),
        {**advice(), "reason": " "},
        advice(pointer="/absent"),
        advice(quote="fabricated"),
        advice(pointer="/items/-1", quote="first"),
        advice(pointer="/items/00", quote="first"),
    ],
)
def test_model_advice_cannot_invent_evidence_or_activation(invalid):
    with pytest.raises(BotError):
        validate_advice(
            invalid, {"test_results": {"passed": True}, "items": ["first"]}, {"reviewer", "qa"}
        )


def test_advice_is_restricted_by_stage_and_supports_json_pointer_escaping():
    validate_advice(advice(pointer="/a~1b/0", quote="real"), {"a/b": ["real proof"]}, {"reviewer"})
    with pytest.raises(BotError, match="перехід"):
        validate_advice(advice("developer"), {"test_results": {"passed": True}}, {"qa", "pm"})


@pytest.mark.parametrize(
    "execution,qa,review",
    [
        ({"passed": False, "tests_run": 3}, "approve", "approve"),
        ({"passed": True, "tests_run": 0}, "approve", "approve"),
        ({"passed": True, "tests_run": 3, "errors": ["failure"]}, "approve", "approve"),
        ({"passed": True, "tests_run": 3, "skipped": 1}, "approve", "approve"),
        ({"passed": True, "tests_run": 3}, "qa", "approve"),
        ({"passed": True, "tests_run": 3}, "approve", "developer"),
    ],
)
def test_python_activation_policy_requires_actual_checks(execution, qa, review):
    assert not DevelopmentSupervisor.may_activate(execution, {"decision": qa}, {"decision": review})
    assert DevelopmentSupervisor.may_activate(
        {"passed": True, "tests_run": 3}, {"decision": "approve"}, {"decision": "approve"}
    )


def research(team, monkeypatch):
    monkeypatch.setattr(
        team.roles.ollama, "models", AsyncMock(return_value=["gemma4:12b", "qwen3-coder:30b"])
    )
    monkeypatch.setattr(
        team.web,
        "search",
        lambda *args: {"results": [{"title": "Python", "url": "https://docs.python.org/3/"}]},
    )


def scripted_roles(*, mode, calls):
    async def ask(role, model, context, schema):
        calls.append((role, context))
        data = json.JSONDecoder().raw_decode(context)[0] if context.startswith("{") else {}
        if role == "Supervisor":
            if mode == "invalid_advice":
                return advice("activate")
            if mode == "stop":
                return advice("stop")
            if mode == "code_bug":
                return advice(
                    "developer", "/implementation/source", 'if number == 0: return {"doubled": 1}'
                )
            return advice(quote="false" if mode == "failed_tests" else "true")
        if role == "Developer":
            developer_count = sum(name == "Developer" for name, _ in calls)
            source = SOURCE
            if mode == "code_bug" and developer_count > 1:
                assert "supervisor_advice" in context
                assert "if number == 0" in context
            if mode == "code_bug" and developer_count == 1:
                source = source.replace(
                    '    return {"doubled": number * 2}',
                    '    if number == 0: return {"doubled": 1}\n    return {"doubled": number * 2}',
                )
            return candidate(source)
        if "queries" in schema.get("properties", {}):
            return {"queries": ["Python integers documentation"]}
        if "tests" in schema.get("properties", {}):
            return {
                "tests": TESTS.replace("], -2)", "], -99)") if mode == "failed_tests" else TESTS,
                "notes": "Three scenarios",
            }
        if "decision" in schema.get("properties", {}):
            if role == "QA":
                return {
                    "decision": "developer" if mode == "failed_tests" else "approve",
                    "findings": ["Actual assertion failure"] if mode == "failed_tests" else [],
                }
            if data.get("phase") == "qa_expectations_review":
                if mode == "empty_review" and sum(name == "Reviewer" for name, _ in calls) == 1:
                    return {"decision": "qa", "findings": []}
                return {"decision": "approve", "findings": []}
            if data.get("supervisor_feedback"):
                assert model == "qwen3-coder:30b"
                return {"decision": "approve", "findings": []}
            final_reviews = sum(
                name == "Reviewer" and '"qa_triage"' in ctx and '"supervisor_feedback"' not in ctx
                for name, ctx in calls
            )
            if mode == "code_bug" and final_reviews > 1:
                return {"decision": "approve", "findings": []}
            return {
                "decision": "qa" if mode == "failed_tests" else "developer",
                "findings": ["Concrete finding disputed by QA"],
            }
        return spec()

    return ask


@pytest.mark.sandbox
@pytest.mark.parametrize("mode", ["empty_review", "repair_review", "code_bug"])
async def test_supervisor_repairs_review_or_routes_real_code_bug(components, monkeypatch, mode):
    _, extensions, _, _ = components
    research(extensions.team, monkeypatch)
    calls = []
    callback = scripted_roles(mode=mode, calls=calls)

    # In the empty-review case, all later reviews are accepted without LLM arbitration.
    async def ask(role, model, context, schema):
        result = await callback(role, model, context, schema)
        if mode == "empty_review" and role == "Reviewer" and '"qa_triage"' in context:
            return {"decision": "approve", "findings": []}
        return result

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    events = []
    result = await extensions.create(
        "Double every integer including zero",
        "tool",
        "supervised_double",
        "root",
        "",
        1,
        progress=events.append,
    )
    assert result["status"] == "active" and result["qa"]["tests_run"] == 3
    report = extensions.team.report(result["team_run_id"], 1)
    decisions = [
        event["data"] for event in report["events"] if event["stage"] == "SupervisorDecision"
    ]
    assert decisions[-1]["action"] == "activate" and decisions[-1]["mode"] == "python"
    if mode == "empty_review":
        assert not any(role == "Supervisor" for role, _ in calls)
        assert any(item["action"] == "reviewer" and item["mode"] == "python" for item in decisions)
    elif mode == "repair_review":
        assert sum(role == "Developer" for role, _ in calls) == 1
        assert sum(role == "QA" and '"previous_tests"' in context for role, context in calls) == 1
        assert any(event.phase == "supervisor" for event in events)
        assert any(event.phase == "reviewer_rework" for event in events)
    else:
        assert sum(role == "Developer" for role, _ in calls) == 2
        assert await extensions.invoke(extensions.get("supervised_double"), {"number": 0}, 1) == {
            "doubled": 0
        }


@pytest.mark.sandbox
@pytest.mark.parametrize("mode", ["invalid_advice", "failed_tests", "stop"])
async def test_supervisor_cannot_bypass_failed_checks(components, settings, monkeypatch, mode):
    _, extensions, _, _ = components
    settings.extension_team_max_rounds = 1
    research(extensions.team, monkeypatch)
    calls = []
    monkeypatch.setattr(extensions.team.roles, "ask", scripted_roles(mode=mode, calls=calls))
    with pytest.raises(TeamFailure) as failed:
        await extensions.create("Double integers", "tool", "supervised_double", "root", "", 1)
    assert not extensions.records()
    assert failed.value.report["last_valid_candidate"]
    if mode == "invalid_advice":
        assert sum(role == "Supervisor" for role, _ in calls) == 2
        assert "контракт" in failed.value.report["error"]
    elif mode == "stop":
        assert "Supervisor зупинив" in failed.value.report["error"]
    else:
        assert any(
            event["stage"] == "QAExecution" and not event["data"]["passed"]
            for event in failed.value.report["events"]
        )


async def test_supervisor_does_not_reset_progress_for_changing_artifacts(
    components, settings, monkeypatch
):
    _, extensions, _, _ = components
    research(extensions.team, monkeypatch)
    settings.extension_team_max_rounds = 30
    settings.extension_team_max_calls = 100
    settings.extension_team_no_progress_limit = 2
    rounds = []

    async def ask(role, model, context, schema):
        if role == "Supervisor":
            return advice("developer")
        if role == "Developer":
            rounds.append(len(rounds))
            return candidate(f"VERSION = {len(rounds)}\n" + SOURCE)
        if "queries" in schema.get("properties", {}):
            return {"queries": ["Python docs"]}
        if "tests" in schema.get("properties", {}):
            return {"tests": f"VERSION = {len(rounds)}\n" + TESTS, "notes": "Tests"}
        if "decision" in schema.get("properties", {}):
            data = json.loads(context)
            return {
                "decision": "approve"
                if role == "QA" or data.get("phase") == "qa_expectations_review"
                else "developer",
                "findings": ["Concrete disputed implementation finding"],
            }
        return spec()

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    with pytest.raises(TeamFailure) as failed:
        await extensions.team.run(
            request="Double integers",
            extension_id="supervised_double",
            user_id=1,
            model="fake",
            developer_context="Implement",
            developer_schema={},
            validate=AsyncMock(side_effect=copy.deepcopy),
            execute_tests=AsyncMock(return_value={"passed": True, "tests_run": 3}),
        )
    assert len(rounds) == 3
    assert failed.value.report["stage"] == "NoProgress"
    assert failed.value.report["last_valid_candidate"]


@pytest.mark.sandbox
async def test_supervisor_shares_existing_model_call_budget(components, settings, monkeypatch):
    _, extensions, _, _ = components
    research(extensions.team, monkeypatch)
    settings.extension_team_max_calls = 10
    calls = []
    monkeypatch.setattr(
        extensions.team.roles, "ask", scripted_roles(mode="repair_review", calls=calls)
    )
    with pytest.raises(TeamFailure) as failed:
        await extensions.create("Double integers", "tool", "supervised_double", "root", "", 1)
    assert failed.value.report["calls"] == 10
    assert "бюджет" in failed.value.report["error"]
    assert not extensions.records()


def test_verified_numeric_objection_is_arbitrated_but_ordinary_feedback_is_not():
    context = {
        "trusted_checks": {"reference_checks": [{"test": "test_happy_numeric", "matched": True}]}
    }
    review = {"decision": "qa", "findings": ["test_happy_numeric has a precision problem"]}
    assert DevelopmentSupervisor.needs_arbitration(review, context)
    assert not DevelopmentSupervisor.needs_arbitration(
        {"decision": "qa", "findings": ["Missing failure coverage"]}, context
    )


@pytest.mark.parametrize(
    "findings", [[], ["notes_for_qa_fix"], ["action_items_for_developer", " "]]
)
def test_placeholder_rejections_are_returned_to_reviewer(findings):
    with pytest.raises(BotError, match="Reviewer"):
        DevelopmentSupervisor.check_review({"decision": "qa", "findings": findings})


@pytest.mark.sandbox
async def test_persistent_reviewer_conflict_is_bounded_without_rewriting_qa(
    components, monkeypatch
):
    _, extensions, _, _ = components
    research(extensions.team, monkeypatch)
    calls = []
    callback = scripted_roles(mode="repair_review", calls=calls)

    async def ask(role, model, context, schema):
        result = await callback(role, model, context, schema)
        if role == "Reviewer" and '"supervisor_feedback"' in context:
            return {"decision": "developer", "findings": ["Unresolved disputed finding"]}
        return result

    monkeypatch.setattr(extensions.team.roles, "ask", ask)
    with pytest.raises(TeamFailure) as failed:
        await extensions.create("Double integers", "tool", "supervised_double", "root", "", 1)
    assert "не узгодив" in failed.value.report["error"]
    assert sum(role == "Supervisor" for role, _ in calls) == 2
    assert sum(role == "Developer" for role, _ in calls) == 1
    assert sum(role == "QA" and '"previous_tests"' in context for role, context in calls) == 1
    assert not extensions.records()

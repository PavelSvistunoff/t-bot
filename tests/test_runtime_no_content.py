from unittest.mock import AsyncMock

import pytest
from google.adk.events import Event
from google.genai import types

from t_bot.errors import BotError


def response(name, data):
    return Event(
        author="root",
        content=types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(name=name, response=data),
            ],
        ),
    )


@pytest.mark.parametrize("success", [True, False])
async def test_empty_model_reply_reports_actual_tool_outcome_without_retry(
    components, monkeypatch, success
):
    runtime, _, _, _ = components
    monkeypatch.setattr(runtime.ollama, "require_model", AsyncMock())
    if success:
        outcome = response(
            "create_extension",
            {
                "status": "active",
                "manifest": {"id": "plotter", "version": 1},
            },
        )
    else:
        outcome = response(
            "install_dependencies",
            {
                "error": "Traceback: internal details\n"
                "ERROR: No matching distribution found for kiwisolver==1.4.1",
            },
        )
    events = [
        outcome,
        Event(
            author="root",
            error_code="MODEL_RETURNED_NO_CONTENT",
            content=types.Content(
                role="model", parts=[types.Part(text="SECRET THOUGHT", thought=True)]
            ),
        ),
    ]
    calls = []

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        async def run_async(self, **kwargs):
            calls.append(kwargs)
            for event in events:
                yield event

    monkeypatch.setattr("t_bot.runtime.Runner", FakeRunner)
    if success:
        result = await runtime.run(1, 5, 1, "Make plot")
        assert "Створено plotter v1" in result
    else:
        with pytest.raises(BotError) as exc:
            await runtime.run(1, 5, 1, "Make plot")
        result = str(exc.value)
        assert "kiwisolver==1.4.1" in result and "Traceback" not in result
    assert "MODEL_RETURNED_NO_CONTENT" in result
    assert "SECRET THOUGHT" not in result
    assert len(calls) == 1


async def test_recovered_tool_failure_does_not_fail_successful_empty_reply():
    from t_bot.task_progress import TaskProgress

    progress = TaskProgress()
    progress.observe(response("create_extension", {"error": "First attempt failed"}))
    progress.observe(
        response(
            "create_extension",
            {
                "status": "active",
                "manifest": {"id": "plotter", "version": 2},
            },
        )
    )
    result = progress.missing_answer()
    assert "Створено plotter v2" in result


async def test_no_content_without_tool_outcomes_is_an_explicit_error():
    from t_bot.task_progress import TaskProgress

    with pytest.raises(BotError, match="Підтверджених результатів інструментів немає"):
        TaskProgress().missing_answer()


async def test_team_result_is_reported_without_model_summary(components, monkeypatch):
    runtime, _, _, _ = components
    monkeypatch.setattr(runtime.ollama, "require_model", AsyncMock())
    outcome = response(
        "create_extension",
        {
            "status": "active",
            "manifest": {"id": "plotter", "version": 1},
            "team_run_id": "a" * 32,
            "qa": {"tests_run": 3},
            "usage": "Use plotter",
            "report_command": "/build " + "a" * 32,
        },
    )
    calls = []

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        async def run_async(self, **kwargs):
            calls.append(kwargs)
            yield outcome

    monkeypatch.setattr("t_bot.runtime.Runner", FakeRunner)
    result = await runtime.run(1, 5, 1, "Create plotter")
    assert "Створено plotter v1" in result and "QA: 3" in result
    assert "/build " in result and "MODEL_RETURNED_NO_CONTENT" not in result
    assert len(calls) == 1

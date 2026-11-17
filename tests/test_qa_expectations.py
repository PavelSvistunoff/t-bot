import math

import pytest

from t_bot.qa_expectations import StagnationGuard, audit_specification, reference_math

REQUEST = (
    "Calculator using Python eval() with math sin, cos and constants; number or error message."
)


@pytest.mark.parametrize(
    "expression,value",
    [
        ("7+5+sin(7)-1+cos(5)", 11.940648784182015),
        ("5++2", 7),
        ("sqrt(9)+log10(100)", 5),
        ("2**10", 1024),
        ("sin(pi/2)", 1),
    ],
)
def test_reference_math_checks_known_oracles(expression, value):
    assert math.isclose(reference_math(expression), value)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo BAD')",
        "().__class__",
        "open('/tmp/secret')",
        "10**1000000000",
        "9**9**9",
        "sqrt(-1)",
        "5+*2",
        "True",
        "1e309",
    ],
)
def test_reference_oracle_rejects_unsupported_or_expensive_expressions(expression):
    assert reference_math(expression) is None


def test_wrong_python_syntax_assumption_goes_back_to_pm():
    result = audit_specification(REQUEST, {"failure_path": ["Синтаксична помилка: '5++2'"]})
    assert not result["passed"] and result["target"] == "pm"
    assert "7" in result["findings"][0]
    assert audit_specification(REQUEST, {"failure_path": ["Синтаксична помилка: '5+*2'"]})["passed"]


def test_repeated_failures_escalate_qa_then_pm_and_stop():
    guard = StagnationGuard(limit=2)
    targets = []
    for i in range(6):
        result = guard.observe(
            {"failures": [f'File "/tmp/run-{i}/tests.py", line {i}\nAssertionError: wrong value']},
            "developer",
        )
        if result:
            targets.append(result)
    assert [r["target"] for r in targets] == ["qa", "pm", "pm"]
    assert targets[-1]["stop"] and targets[-1]["occurrences"] == 6


@pytest.mark.parametrize(
    "description",
    [
        "Вираз 'sqrt(16) + (2 * 3)' повертає 8.0.",
        "'sqrt(16) + (2 * 3)' returns 8.0.",
    ],
)
def test_numeric_acceptance_criteria_are_checked_before_developer(description):
    audit = audit_specification(
        "Create a scientific calculator",
        {
            "acceptance_criteria": [description],
            "failure_path": ["Синтаксичні помилки: '5++2' або '10 /'."],
        },
    )
    assert not audit["passed"]
    assert {(c["section"], c["index"]) for c in audit["corrections"]} == {
        ("acceptance_criteria", 0),
        ("failure_path", 0),
    }
    corrected = {c["section"]: [c["suggested"]] for c in audit["corrections"]}
    assert "10.0" in corrected["acceptance_criteria"][0]
    assert "'10 /'" in corrected["failure_path"][0]
    assert audit_specification("Create a scientific calculator", corrected)["passed"]


def test_custom_grammar_is_not_overridden_by_python_math_rules():
    audit = audit_specification(
        "Calculator with custom grammar",
        {
            "failure_path": ["Syntax error: '5++2'"],
        },
    )
    assert audit["passed"]


def test_absolute_value_reference_is_supported():
    assert reference_math("abs(-7) + sqrt(16)") == 11


def test_scientific_notation_in_correct_requirement_is_not_truncated():
    assert audit_specification(
        "Scientific calculator",
        {
            "acceptance_criteria": ["'10**5' returns 1e5."],
        },
    )["passed"]


@pytest.mark.parametrize(
    "user_request",
    [
        "Створи інструмент для обчислення складних математичних виразів. Використовуй math.",
        "ОБЧИСЛИ ВИРАЗ 2+2",
        "Реалізуй арифметичні операції з дужками",
        "Evaluate mathematical expressions using sin and cos",
    ],
)
def test_ukrainian_and_english_math_intent_without_word_calculator(user_request):
    audit = audit_specification(
        user_request,
        {
            "happy_path": ["Користувач вводить '2 + 2 * 5', інструмент повертає 15."],
            "failure_path": ["Невалідний синтаксис: '5 ++ 2'."],
        },
    )
    assert not audit["passed"]
    assert {c["reference"] for c in audit["corrections"]} == {12, 7}


@pytest.mark.parametrize(
    "user_request",
    [
        "Переклади документацію math українською",
        "Порахуй частоту літер у файлі math.txt",
        "Калькулятор: тригонометричні функції у градусах",
        "Обчислюй математичні вирази, використовуючи власну граматику",
    ],
)
def test_unrelated_or_custom_semantics_do_not_enable_python_oracle(user_request):
    assert audit_specification(user_request, {"failure_path": ["Syntax: '5++2'"]})["passed"]


def test_artifact_loop_ignores_review_prose_comments_and_whitespace():
    from t_bot.qa_expectations import artifact_fingerprint

    guard = StagnationGuard(limit=2)
    routes = []
    for i in range(6):
        candidate = {"source": f"# Attempt {i}\ndef invoke(payload):\n    return 1\n"}
        tests = f"# Retry {i}\ndef test_happy():\n    assert 1 == 1\n"
        state = {"phase": "test_review", **artifact_fingerprint(candidate, tests)}
        outcome = guard.observe({"findings": [f"Different wording {i}"]}, "qa", state=state)
        if outcome:
            routes.append(outcome)
    assert [item["target"] for item in routes] == ["pm", "pm", "pm"]
    assert routes[-1]["stop"] and routes[-1]["basis"] == "artifacts"


def test_changed_code_tests_or_contract_count_as_distinct_progress():
    from t_bot.qa_expectations import artifact_fingerprint

    guard = StagnationGuard(limit=2)
    candidate = {"source": "def invoke(payload):\n    return 1", "manifest": {}}
    base = artifact_fingerprint(candidate, "assert 1")
    states = [
        base,
        artifact_fingerprint(
            {**candidate, "source": "def invoke(payload):\n    return 2"}, "assert 1"
        ),
        artifact_fingerprint(candidate, "assert 2"),
        artifact_fingerprint(
            {**candidate, "manifest": {"output_schema": {"type": "number"}}}, "assert 1"
        ),
    ]
    assert len({str(state) for state in states}) == 4
    assert all(
        guard.observe({"error": "same prose"}, "developer", state=state) is None for state in states
    )

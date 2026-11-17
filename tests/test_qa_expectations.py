import math

import pytest

from t_bot.qa_expectations import StagnationGuard, audit_specification, audit_tests, reference_math

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


def test_report_regressions_are_detected_in_qa_expectations():
    tests = """class Tests:
    def test_happy_expression(self):
        payload = {"expression": "7+5+sin(7)-1+cos(5)"}
        result = invoke(payload)
        self.assertAlmostEqual(result["result"], 12.534569, places=5)
    def test_failure_syntax(self):
        payload = {"expression": "5++2"}
        result = invoke(payload)
        self.assertIn("Помилка", str(result["result"]))
    def test_edge_invalid_type(self):
        payload = {"expression": 123}
        result = invoke(payload)
        self.assertEqual(result["result"], "Помилка: Вирази повинні бути рядком")
"""
    audit = audit_tests(REQUEST, {}, tests)
    assert not audit["passed"] and audit["target"] == "qa"
    assert len(audit["findings"]) == 3
    assert audit["reference_checks"][0]["reference"] == 11.940648784182015
    fixed = tests.replace("12.534569", "11.94065").replace("5++2", "5+*2")
    fixed = fixed.replace(
        'self.assertEqual(result["result"], "Помилка: Вирази повинні бути рядком")',
        'self.assertIsInstance(result["result"], str)',
    )
    assert audit_tests(REQUEST, {}, fixed)["passed"]


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


def test_reused_result_variable_keeps_each_expression_oracle():
    tests = """class Tests:
    def test_happy_expressions(self):
        result = invoke({"expression": "2+3"})
        self.assertEqual(result["result"], 5)
        result = invoke({"expression": "4+5"})
        self.assertEqual(result["result"], 9)
"""
    audit = audit_tests(REQUEST, {}, tests)
    assert audit["passed"]
    assert [r["reference"] for r in audit["reference_checks"]] == [5, 9]


def test_incorrect_numeric_requirement_is_returned_to_pm():
    tests = """class Tests:
    def test_happy_expression(self):
        result = invoke({"expression": "7+5+sin(7)-1+cos(5)"})
        self.assertAlmostEqual(result["result"], 12.534569, places=5)
"""
    audit = audit_tests(
        REQUEST,
        {"acceptance_criteria": ["7+5+sin(7)-1+cos(5) returns 12.534569"]},
        tests,
    )
    assert not audit["passed"] and audit["target"] == "pm"


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


DYNAMIC_TEST = """import math as maths
from math import cos as cosine

class Tests:
    def test_happy_expression(self):
        payload = {"text": "7+5+sin(7)-1+cos(5)"}
        response = invoke(payload)
        base = 7 + 5
        expected = base + maths.sin(7) - 1 + cosine(5)
        self.assertAlmostEqual(response["result"], expected, places=5)
"""


def test_dynamic_math_oracle_is_verified_with_import_aliases_and_variables():
    audit = audit_tests("Обчисли математичний вираз", {}, DYNAMIC_TEST)
    assert audit["passed"]
    assert audit["reference_checks"][0]["matched"] is True
    assert audit["reference_checks"][0]["expected"] == 11.940648784182015
    wrong = DYNAMIC_TEST.replace("base = 7 + 5", "base = 7 + 6")
    audit = audit_tests("Обчисли математичний вираз", {}, wrong)
    assert not audit["passed"] and audit["target"] == "qa"
    assert audit["reference_checks"][0]["matched"] is False


def test_dynamic_inline_oracle_and_direct_invocation():
    tests = """import math
class Tests:
    def test_happy(self):
        self.assertAlmostEqual(invoke({"text": "sqrt(16)"})["result"], math.sqrt(16))
"""
    audit = audit_tests(REQUEST, {}, tests)
    assert audit["passed"] and audit["reference_checks"][0]["matched"]


def test_delta_is_respected_when_verifying_rounded_expected_number():
    tests = DYNAMIC_TEST.replace("base + maths.sin(7) - 1 + cosine(5)", "11.94").replace(
        "places=5", "delta=0.01"
    )
    assert audit_tests(REQUEST, {}, tests)["passed"]
    assert not audit_tests(REQUEST, {}, tests.replace("delta=0.01", "delta=0.00001"))["passed"]


def test_valid_expression_cannot_be_expected_to_return_error_dictionary():
    tests = """class Tests:
    def test_failure(self):
        response = invoke({"text": "5 ++ 2"})
        self.assertIn("error", response)
        self.assertNotIn("result", response)
"""
    audit = audit_tests("Інструмент обчислення математичних виразів", {}, tests)
    assert not audit["passed"] and audit["target"] == "qa"


def test_unknown_reassignment_does_not_reuse_stale_numeric_oracle(tmp_path):
    marker = tmp_path / "not-created"
    tests = DYNAMIC_TEST.replace("expected = base", "expected = 123\n        expected = base")
    tests = tests.replace(
        "base + maths.sin(7) - 1 + cosine(5)",
        f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')",
    )
    audit = audit_tests(REQUEST, {}, tests)
    assert not audit["reference_checks"]
    assert not marker.exists()


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

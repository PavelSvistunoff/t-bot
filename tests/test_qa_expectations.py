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

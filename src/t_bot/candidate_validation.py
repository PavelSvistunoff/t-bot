"""Check semantic success of a generated tool's example before accepting its candidate."""

import math

from t_bot.errors import BotError
from t_bot.qa_expectations import python_math_task, reference_value


def verify_smoke_result(request: str, manifest: dict, result: dict, oracle: dict) -> dict:
    if (
        result.get("error")
        or result.get("success") is False
        or result.get("status")
        in (
            "error",
            "failed",
            "failure",
        )
    ):
        raise BotError(
            "Happy-path приклад повернув помилку замість успішного результату: "
            + str(result.get("error", result.get("status", "success=false")))[:500]
        )
    summary = {"passed": True, "result_keys": sorted(result)[:32]}
    if not python_math_task(request):
        return summary
    numbers = [value for value in result.values() if type(value) in {int, float}]
    value = result.get("result", result.get("value", numbers[0] if len(numbers) == 1 else None))
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise BotError("Happy-path математичного інструмента має повертати скінченне число.")
    example = manifest["example"]
    expression = next(
        (example[key] for key in ("text", "expression", "expr") if key in example), None
    )
    if expression is None and len(example) == 1:
        expression = next(iter(example.values()))
    semantics = oracle["semantics"]
    expected = (
        reference_value(expression, semantics, oracle.get("values"))
        if isinstance(expression, str)
        else None
    )
    if (
        isinstance(expression, str)
        and oracle.get("values", {}).get(expression, {}).get("status") == "domain_error"
    ):
        raise BotError("Happy-path приклад містить математичну помилку; виберіть валідний приклад.")
    if expected is not None:
        try:
            matches = math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9)
        except OverflowError:
            matches = value == expected
        if not matches:
            raise BotError(f"Happy-path повернув {value}, контрольний результат {expected}.")
    return {**summary, "semantics": semantics, "numeric_result": value, "reference": expected}

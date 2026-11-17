"""Bounded mathematical references, specification checks and progress guards."""

import ast
import hashlib
import json
import math
import operator
import re
from collections import Counter

FUNCTIONS = {
    name: getattr(math, name)
    for name in (
        "sin",
        "cos",
        "tan",
        "log",
        "sqrt",
        "log10",
        "exp",
    )
}
FUNCTIONS["abs"] = abs

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def reference_math(expression: str) -> int | float | None:
    """Evaluate a small, bounded math AST, without eval, imports, attributes or Python code."""

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = node.value
        elif isinstance(node, ast.Name) and node.id in {"pi", "e"}:
            value = getattr(math, node.id)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            value = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 1000 or (
                    isinstance(left, int) and right > 0 and left.bit_length() * right > 4096
                ):
                    raise ValueError("Power exceeds oracle budget")
            value = OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.Call) and not node.keywords and len(node.args) <= 2:
            name = node.func.id if isinstance(node.func, ast.Name) else ""
            if name not in FUNCTIONS:
                raise ValueError("Unsupported function")
            value = FUNCTIONS[name](*(evaluate(arg) for arg in node.args))
        else:
            raise ValueError("Unsupported expression")
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError("Not a finite real number")
        if isinstance(value, int) and value.bit_length() > 4096:
            raise ValueError("Integer exceeds oracle budget")
        return value

    try:
        if not expression or len(expression) > 512:
            return None
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            return None
        return evaluate(tree.body)
    except (ValueError, SyntaxError, TypeError, ArithmeticError, RecursionError):
        return None


def python_math_task(request: str) -> bool:
    text = request.casefold()
    mathematical_expressions = bool(
        re.search(r"математичн|арифметичн|mathematical|arithmetic", text)
        and re.search(r"вираз|обчисл|операц|expression|evaluat|calculat", text)
    )
    expression_calculation = "обчисл" in text and "вираз" in text
    return (
        ("eval" in text and "math" in text)
        or any(word in text for word in ("calculator", "калькулятор"))
        or mathematical_expressions
        or expression_calculation
    ) and not re.search(
        r"градус|degrees?|custom\s+grammar|(?:власн|нестандартн)\w*\s+грамат",
        text,
    )


def math_semantics(request: str, manifest: dict | None = None) -> str:
    if manifest is not None:
        return (
            "sympy"
            if any(
                dependency.split("==", 1)[0].casefold() == "sympy"
                for dependency in manifest.get("dependencies", [])
            )
            else "math"
        )
    return "sympy" if "sympy" in request.casefold() else "math"


def math_expressions(spec: dict, example: dict) -> list[str]:
    expressions = [value for value in example.values() if isinstance(value, str)]
    for cases in spec.values():
        if isinstance(cases, list):
            for case in cases:
                if not isinstance(case, str):
                    continue
                expressions.extend(re.findall(r"['\"`]([^'\"`]+)['\"`]", case))
    return list(dict.fromkeys(value for value in expressions if 0 < len(value) <= 512))[:128]


def reference_value(expression: str, semantics: str, references: dict | None = None):
    if semantics == "sympy":
        # Never substitute binary-float math for a symbolic domain check (e.g. tan(pi/2)).
        result = (references or {}).get(expression, {})
        return result.get("value") if result.get("status") == "number" else None
    return reference_math(expression)


def audit_specification(
    request: str,
    spec: dict,
    *,
    semantics: str | None = None,
    references: dict | None = None,
) -> dict:
    """Locate independently verifiable contradictions before implementation or QA."""
    findings = []
    corrections = []
    semantics = semantics or math_semantics(request)
    if python_math_task(request):
        for section in ("requirements", "acceptance_criteria", "happy_path", "failure_path"):
            for index, case in enumerate(spec.get(section, [])):
                for match in re.finditer(r"['\"`]([^'\"`]+)['\"`]", case):
                    expression = match.group(1)
                    value = reference_value(expression, semantics, references)
                    if value is None:
                        continue
                    location = f"{section}[{index}]"
                    if section == "failure_path" and any(
                        word in case.casefold() for word in ("синтакс", "syntax")
                    ):
                        findings.append(
                            f"{location}: {expression!r} є валідним виразом зі значенням {value}, "
                            "а не синтаксичною помилкою. Виправте саме цей пункт, "
                            "замінивши приклад на справді невалідний '5+*2'."
                        )
                        corrections.append(
                            {
                                "section": section,
                                "index": index,
                                "original": case,
                                "suggested": case[: match.start(1)] + "5+*2" + case[match.end(1) :],
                                "expression": expression,
                                "reference": value,
                            }
                        )
                        continue
                    expected = re.match(
                        r"\s*(?:,\s*(?:інструмент|система|програма)\s+)?"
                        r"(?:поверт\w*|returns?|equals?|дорівню\w*|=)\s+"
                        r"(-?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)(?![\w,])",
                        case[match.end() :],
                        re.IGNORECASE,
                    )
                    if expected is None:
                        continue
                    number = float(expected.group(1).replace(",", "."))
                    if math.isclose(number, value, rel_tol=1e-7, abs_tol=1e-7):
                        continue
                    findings.append(
                        f"{location}: неправильний результат {expression!r}: "
                        f"задано {number}, контрольний результат {value}."
                    )
                    first = match.end() + expected.start(1)
                    last = match.end() + expected.end(1)
                    corrections.append(
                        {
                            "section": section,
                            "index": index,
                            "original": case,
                            "suggested": case[:first] + str(float(value)) + case[last:],
                            "expression": expression,
                            "reference": value,
                        }
                    )
    return {
        "passed": not findings,
        "target": "pm",
        "findings": findings,
        "corrections": corrections,
    }


def json_text(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False)


def artifact_fingerprint(candidate: dict, tests: str) -> dict[str, str]:
    """Hash executable syntax and contracts; formatting, comments and review prose are ignored."""

    def code_digest(source: str) -> str:
        try:
            normalized = ast.dump(ast.parse(source), include_attributes=False)
        except (SyntaxError, ValueError, RecursionError):
            normalized = source.strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    manifest = candidate.get("manifest", {})
    contract = {
        key: manifest.get(key)
        for key in (
            "kind",
            "entrypoint",
            "input_schema",
            "output_schema",
            "dependencies",
            "model",
            "parent",
            "permissions",
        )
    }
    return {
        "candidate_digest": code_digest(candidate.get("source", "")),
        "tests_digest": code_digest(tests),
        "contract_digest": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest(),
    }


class StagnationGuard:
    """Escalate repeated failures from Developer to QA, then PM, then stop explicitly."""

    def __init__(self, limit: int = 3):
        self.limit = limit
        self.counts: Counter[str] = Counter()
        self.escalations: Counter[str] = Counter()

    def observe(self, evidence: dict, target: str, *, state: dict | None = None) -> dict | None:
        if state is not None:
            text = "artifacts:" + json.dumps(state, sort_keys=True)
        elif evidence.get("failures") or evidence.get("errors"):
            text = json_text(
                {
                    kind: [
                        item.strip().splitlines()[-1]
                        for item in evidence.get(kind, [])
                        if item.strip()
                    ]
                    for kind in ("failures", "errors")
                }
            )
        else:
            text = json_text(evidence)
        text = re.sub(r"/(?:private/var/folders|tmp|var/tmp)/[^\s\"']+", "<temporary>", text)
        text = re.sub(r"line \d+", "line N", text)
        text = re.sub(r"test_\w+", "<test>", text)
        self.counts[text] += 1
        if self.counts[text] % self.limit:
            return None
        self.escalations[text] += 1
        level = self.escalations[text]
        route = ("qa" if target == "developer" else "pm") if level == 1 else "pm"
        return {
            "occurrences": self.counts[text],
            "target": route,
            "stop": level >= 3,
            "basis": "artifacts" if state is not None else "error",
            "state": state,
        }


class NoProgressGuard:
    """Bound failed rounds across different artifacts; only a new milestone resets the counter."""

    def __init__(self, limit: int):
        self.limit = limit
        self.best_milestone = -1
        self.failed_rounds = 0

    def observe(self, milestone: int) -> dict | None:
        if milestone > self.best_milestone:
            self.best_milestone = milestone
            self.failed_rounds = 0
        else:
            self.failed_rounds += 1
        if self.failed_rounds < self.limit:
            return None
        return {"failed_rounds": self.failed_rounds, "milestone": self.best_milestone}

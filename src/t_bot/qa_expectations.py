"""Deterministic checks of test oracles; never execute generated Python in the host."""

import ast
import copy
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


def audit_specification(request: str, spec: dict) -> dict:
    """Locate independently verifiable contradictions before implementation or QA."""
    findings = []
    corrections = []
    if python_math_task(request):
        for section in ("requirements", "acceptance_criteria", "happy_path", "failure_path"):
            for index, case in enumerate(spec.get(section, [])):
                for match in re.finditer(r"['\"`]([^'\"`]+)['\"`]", case):
                    expression = match.group(1)
                    value = reference_math(expression)
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


def _literal(node, values: dict):
    if isinstance(node, ast.Name):
        return values.get(node.id)
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _math_imports(tree: ast.Module) -> tuple[set[str], dict[str, str]]:
    modules: set[str] = set()
    functions = {"abs": "abs"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.asname or "math" for alias in node.names if alias.name == "math")
        elif isinstance(node, ast.ImportFrom) and node.module == "math":
            functions.update(
                {
                    alias.asname or alias.name: alias.name
                    for alias in node.names
                    if alias.name in FUNCTIONS or alias.name in {"pi", "e"}
                }
            )
    return modules, functions


class _OracleNames(ast.NodeTransformer):
    """Normalize whitelisted math imports and numeric variables, without importing test code."""

    def __init__(self, values: dict, modules: set[str], functions: dict[str, str]):
        self.values = values
        self.modules = modules
        self.functions = functions

    def visit_Name(self, node: ast.Name) -> ast.AST:
        value = self.values.get(node.id)
        if type(value) in {int, float}:
            return ast.Constant(value=value)
        return ast.Name(id=self.functions.get(node.id, node.id), ctx=ast.Load())

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id in self.modules:
            if node.attr in FUNCTIONS or node.attr in {"pi", "e"}:
                return ast.Name(id=node.attr, ctx=ast.Load())
        return node


def _expected_value(node: ast.AST, values: dict, modules: set[str], functions: dict[str, str]):
    literal = _literal(node, values)
    if literal is not None:
        return literal
    nodes = list(ast.walk(node))
    if len(nodes) > 128 or any(
        isinstance(child, ast.Name)
        and child.id not in modules
        and child.id not in functions
        and type(values.get(child.id)) not in {int, float}
        for child in nodes
    ):
        return None
    normalized = _OracleNames(values, modules, functions).visit(copy.deepcopy(node))
    try:
        return reference_math(ast.unparse(ast.fix_missing_locations(normalized)))
    except (ValueError, RecursionError):
        return None


def _result_expression(node, results: dict, values: dict):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str":
        node = node.args[0] if node.args else None
    if isinstance(node, ast.Subscript):
        return _result_expression(node.value, results, values)
    if isinstance(node, ast.Name):
        return results.get(node.id)
    if isinstance(node, ast.Call) and node.args:
        function = node.func
        if (isinstance(function, ast.Name) and function.id == "invoke") or (
            isinstance(function, ast.Attribute) and function.attr == "invoke"
        ):
            payload = _literal(node.args[0], values)
            if isinstance(payload, dict):
                return next(
                    (payload[k] for k in ("expression", "expr", "text") if k in payload), None
                )
    return None


def _numeric_agreement(
    call: ast.Call, expected: int | float, reference: int | float, values: dict
) -> bool | None:
    """Apply unittest numeric assertion semantics, including explicit delta or places."""
    if not isinstance(call.func, ast.Attribute):
        return None
    if expected == reference:
        return True
    if call.func.attr == "assertEqual":
        return False
    arguments = {item.arg: _literal(item.value, values) for item in call.keywords}
    delta = arguments.get("delta", _literal(call.args[4], values) if len(call.args) > 4 else None)
    if delta is not None:
        return abs(expected - reference) <= delta if type(delta) in {int, float} else None
    places = arguments.get("places", _literal(call.args[2], values) if len(call.args) > 2 else 7)
    if places is None:
        places = 7
    if type(places) is not int or abs(places) > 308:
        return None
    return round(abs(expected - reference), places) == 0


def audit_tests(request: str, spec: dict, tests: str) -> dict:
    """Check directly analyzable unittest assertions; unknown forms still require review."""
    specification = audit_specification(request, spec)
    if not specification["passed"]:
        return specification
    findings = []
    references = []
    target = "qa"
    tree = ast.parse(tests)
    imported_modules, imported_functions = _math_imports(tree)
    for method in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        values: dict = {}
        results: dict = {}
        modules, functions = set(imported_modules), dict(imported_functions)
        assertions: list[tuple[ast.Call, dict, dict, set[str], dict[str, str]]] = []
        for node in method.body:
            # Branches/loops need independent review; do not merge their variable scopes.
            if not isinstance(node, ast.Assign | ast.Expr):
                continue
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                name = node.targets[0].id
                literal = _expected_value(node.value, values, modules, functions)
                values.pop(name, None)
                results.pop(name, None)
                modules.discard(name)
                functions.pop(name, None)
                if literal is not None:
                    values[name] = literal
                if isinstance(node.value, ast.Call) and node.value.args:
                    function = node.value.func
                    is_invoke = (isinstance(function, ast.Name) and function.id == "invoke") or (
                        isinstance(function, ast.Attribute) and function.attr == "invoke"
                    )
                    payload = _literal(node.value.args[0], values)
                    if is_invoke and isinstance(payload, dict):
                        results[name] = next(
                            (payload[k] for k in ("expression", "expr", "text") if k in payload),
                            None,
                        )
            assertions.extend(
                (call, dict(values), dict(results), set(modules), dict(functions))
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            )
        for call, values, results, modules, functions in assertions:
            if len(call.args) < 2:
                continue
            if not isinstance(call.func, ast.Attribute):
                continue
            assertion = call.func.attr
            actual = _result_expression(call.args[0], results, values)
            expected = _expected_value(call.args[1], values, modules, functions)
            if assertion in {"assertEqual", "assertAlmostEqual"}:
                if actual is None:
                    actual = _result_expression(call.args[1], results, values)
                    expected = _expected_value(call.args[0], values, modules, functions)
                if (
                    type(expected) in {int, float}
                    and isinstance(actual, str)
                    and python_math_task(request)
                ):
                    reference = reference_math(actual)
                    agreement = (
                        _numeric_agreement(call, expected, reference, values)
                        if reference is not None
                        else None
                    )
                    if reference is not None:
                        references.append(
                            {
                                "test": method.name,
                                "expression": actual,
                                "expected": expected,
                                "reference": reference,
                                "matched": agreement,
                            }
                        )
                    if agreement is False:
                        if actual in json_text(spec) and str(expected) in json_text(spec):
                            target = "pm"
                        findings.append(
                            f"{method.name}: неправильний еталон для {actual!r}: "
                            f"очікується {expected}, контрольний результат {reference}."
                        )
                if isinstance(expected, str) and (
                    expected.casefold().startswith(("помилка:", "помилка обчислення:", "error:"))
                ):
                    if expected not in request and expected not in json_text(spec):
                        findings.append(
                            f"{method.name}: текст помилки {expected!r} не заданий контрактом; "
                            "перевіряйте тип/ознаку помилки, а не довільне формулювання."
                        )
            if assertion in {"assertIn", "assertNotIn"} and isinstance(call.args[0], ast.Constant):
                marker = call.args[0].value
                actual = _result_expression(call.args[1], results, values)
                if (
                    isinstance(marker, str)
                    and (
                        assertion == "assertIn"
                        and any(s in marker.casefold() for s in ("помил", "error"))
                        or assertion == "assertNotIn"
                        and marker == "result"
                    )
                    and isinstance(actual, str)
                    and python_math_task(request)
                    and reference_math(actual) is not None
                ):
                    findings.append(
                        f"{method.name}: для валідного {actual!r} помилково очікується помилка."
                    )
    return {
        "passed": not findings,
        "target": target,
        "findings": list(dict.fromkeys(findings)),
        "reference_checks": references,
        "scope": "Bounded Python-math/literal checks; other oracles need independent review.",
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

"""Image artifact contracts and evidence for procedural-image QA and review."""

import ast
import re

from t_bot.errors import BotError

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
FEATURE_PATTERNS = {
    "fur": r"\bfur\b|\bfurry\b|\bхутр\w*|\bшерст\w*",
    "size": r"\bsize\b|\bрозмір\w*",
    "ears": r"\bears?\b|\bвух\w*",
    "eyes": r"\beyes?\b|\bоч(?:і|ей|има)\b",
    "whiskers": r"\bwhiskers?\b|\bвус\w*",
}

DIMENSION = re.compile(r"(\d{1,5})\s*[xх×]\s*(\d{1,5})", re.IGNORECASE)


def random_image_request(request: str) -> bool:
    text = request.casefold()
    return bool(
        re.search(r"зображ|малю|\bimage\b|\bcat\b|кот", text)
        and re.search(r"випадков|унікальн|random|unique", text)
    )


def image_contract(request: str, spec: dict) -> dict:
    facts = []
    sources = [("user_request", 0, request)]
    for section, cases in spec.items():
        if isinstance(cases, list):
            sources.extend((section, index, text) for index, text in enumerate(cases))
    for section, index, text in sources:
        for match in DIMENSION.finditer(text):
            anchors = list(
                re.finditer(
                    r"голов\w*|head|полотн\w*|canvas|зображенн\w*|image",
                    text[: match.start()],
                    re.IGNORECASE,
                )
            )
            if not anchors:
                continue
            subject = (
                "head" if re.search(r"голов|head", anchors[-1].group(), re.IGNORECASE) else "canvas"
            )
            facts.append(
                {
                    "subject": subject,
                    "size": [int(match[1]), int(match[2])],
                    "binding": not bool(
                        re.search(
                            r"наприклад|приклад|example|e\.g\.",
                            text[: match.start()],
                            re.IGNORECASE,
                        )
                    ),
                    "section": section,
                    "index": index,
                    "quote": text,
                }
            )
    return {
        "enabled": random_image_request(request),
        "random_features": [
            name
            for name, pattern in FEATURE_PATTERNS.items()
            if re.search(pattern, request, re.IGNORECASE)
        ],
        "geometry_facts": facts,
        "guidance": (
            "Canvas/image size and head size are different quantities. "
            "Examples are not fixed constraints. For each random_features item write "
            "a test_*_<feature>* method: control seed/mock, assert sampler variation and "
            "compare decoded pixels/crops while holding other features fixed. "
            "File existence, status and unique names alone are insufficient."
        ),
    }


def validate_size_claims(review: dict, contract: dict) -> None:
    """Reject a claimed size requirement unless its cited geometry fact actually establishes it."""
    claims = review.get("size_claims", [])
    for finding in review.get("findings", []):
        if not re.search(r"полотн|canvas|зображенн|image", finding, re.IGNORECASE):
            continue
        if not re.search(
            r"вказано|вимага|повин|must|required|specifies|specified", finding, re.IGNORECASE
        ):
            continue
        for match in DIMENSION.finditer(finding):
            size = [int(match[1]), int(match[2])]
            if not any(
                claim.get("subject") == "canvas" and claim.get("size") == size for claim in claims
            ):
                raise BotError(
                    "Твердження про обов'язковий розмір полотна потребує size_claims "
                    "із binding geometry_facts. Приклад розміру голови не є вимогою до полотна."
                )
    for claim in claims:
        if not any(
            fact["binding"]
            and all(claim.get(key) == fact[key] for key in ("subject", "size", "section", "index"))
            for fact in contract["geometry_facts"]
        ):
            raise BotError(
                "Непідтверджена вимога до розміру. Полотно й голова — різні величини; "
                "приклад розміру голови не визначає розмір полотна. "
                "Використайте binding geometry_facts або приберіть хибне зауваження."
            )


def image_review_schema(base: dict) -> dict:
    import copy

    schema = copy.deepcopy(base)
    schema["properties"]["size_claims"] = {
        "type": "array",
        "maxItems": 10,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "subject": {"enum": ["canvas", "head"]},
                "size": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "section": {"type": "string"},
                "index": {"type": "integer", "minimum": 0},
            },
            "required": ["subject", "size", "section", "index"],
        },
    }
    schema["required"] = [*schema["required"], "size_claims"]
    return schema


def _from_pixels(node: ast.AST, variables: set[str]) -> bool:
    return any(
        isinstance(child, ast.Name)
        and child.id in variables
        or isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr in {"getpixel", "getdata", "tobytes", "histogram", "getbbox"}
        for child in ast.walk(node)
    )


def audit_image_tests(tests: str, contract: dict) -> dict:
    """Require executable pixel assertions and controlled per-feature tests, beyond path checks.

    This is a coverage gate, not a proof of visual anatomy.
    QA execution and code review remain required.
    """
    tree = ast.parse(tests)
    findings = []
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    for feature in contract["random_features"]:
        covered = False
        for method in methods:
            if feature not in method.name.casefold():
                continue
            calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
            names = [ast.unparse(call.func) for call in calls]
            pixel_variables: set[str] = set()
            nodes = list(ast.walk(method))
            for _ in range(len(nodes)):
                before = len(pixel_variables)
                for node in nodes:
                    if isinstance(node, ast.Assign) and _from_pixels(node.value, pixel_variables):
                        pixel_variables.update(
                            target.id for target in node.targets if isinstance(target, ast.Name)
                        )
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        if node.func.attr in {"append", "add", "extend"} and isinstance(
                            node.func.value, ast.Name
                        ):
                            if any(_from_pixels(arg, pixel_variables) for arg in node.args):
                                pixel_variables.add(node.func.value.id)
                if len(pixel_variables) == before:
                    break
            pixels = any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr.startswith("assert")
                and any(_from_pixels(arg, pixel_variables) for arg in call.args)
                for call in calls
            )
            controlled = any(
                name.rsplit(".", 1)[-1] in {"seed", "patch", "object"} for name in names
            )
            asserted = any(
                name.rsplit(".", 1)[-1]
                in {"assertNotEqual", "assertGreater", "assertTrue", "assertEqual"}
                for name in names
            )
            if pixels and controlled and asserted:
                covered = True
                break
        if not covered:
            findings.append(
                f"Потрібен test_*_{feature}*: контрольований seed/mock "
                "та assertions про пікселі/область цієї ознаки. "
                "Імена файлів і статус success не перевіряють ознаку."
            )
    return {"passed": not findings, "target": "qa", "findings": findings}

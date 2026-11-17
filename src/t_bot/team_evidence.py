"""Ground role decisions in exact, scoped source evidence."""

import json
import re
from difflib import SequenceMatcher

from t_bot.errors import BotError

EVIDENCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pointer": {"type": "string", "minLength": 1, "maxLength": 300},
        "quote": {"type": "string", "minLength": 1, "maxLength": 1000},
    },
    "required": ["pointer", "quote"],
}
QUALITY_RULES = (
    "Clean Code: focused responsibilities, meaningful names, explicit contracts, "
    "safe resource lifetime, concrete error handling. Explain a concrete consequence; "
    "stylistic preferences alone do not block activation."
)


class EvidenceError(BotError):
    def __init__(self, message: str, failure_kind: str):
        super().__init__(message)
        self.failure_kind = failure_kind


def evidence_text(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def evidence_examples(context: dict) -> list[dict]:
    """Offer short, copyable leaf quotes; never normalize them into fabricated evidence."""
    examples: list[dict] = []

    def visit(value: object, pointer: str) -> None:
        if len(examples) >= 8:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for index, item in enumerate(value[:2]):
                visit(item, pointer + f"/{index}")
        else:
            text = evidence_text(value)
            if text.strip():
                examples.append({"pointer": pointer, "quote": text[:160]})

    visit(context, "")
    return examples


def normalized_quote(text: str) -> str:
    return " ".join(text.split())


def nearest_excerpt(text: str, quote: str) -> str:
    """Return original text around the closest word-sequence match, not navigation by default."""
    words = list(re.finditer(r"\S+", text))
    if not words:
        return text[:600]
    match = SequenceMatcher(
        None, quote.split()[:200], [word.group() for word in words], autojunk=False
    ).find_longest_match()
    start = words[max(0, match.b - 8)].start()
    return text[start : start + 600]


def validate_evidence(item: dict, context: dict, *, page_quote: bool = False) -> None:
    pointer = item.get("pointer", "")
    try:
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise ValueError("Expected slash-prefixed JSON Pointer")
        value = context
        for part in pointer[1:].split("/"):
            if re.search(r"~(?![01])", part):
                raise ValueError("Invalid escape")
            key = part.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not re.fullmatch(r"0|[1-9][0-9]*", key):
                    raise ValueError("Invalid index")
                value = value[int(key)]
            else:
                value = value[key]
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        raise EvidenceError(
            f"Неіснуючий pointer {pointer!r}. Корінь — переданий evidence, без /evidence. "
            "Використай /implementation/source, а не implementation.source; "
            "масиви адресуй /spec/requirements/0. Скопіюй відповідний pointer і точну цитату "
            "з цих реальних прикладів: "
            + json.dumps(evidence_examples(context), ensure_ascii=False),
            "pointer",
        ) from exc
    text = evidence_text(value)
    quote = item.get("quote")
    matches = (
        isinstance(quote, str)
        and bool(quote.strip())
        and (normalized_quote(quote) in normalized_quote(text) if page_quote else quote in text)
    )
    if not matches:
        location = f" URL: {context.get('url', '')}." if page_quote else ""
        requirement = (
            "Пробільні символи нормалізуються; слова, числа й пунктуація мають збігатися. "
            if page_quote
            else "Збережи лапки, пробіли й переноси рядків коду. "
        )
        raise EvidenceError(
            f"Цитата за {pointer} не відповідає фактичному доказу.{location} "
            "Скопіюй дослівний уривок. "
            + requirement
            + "Не використовуй переказ або '...'. Найближчий реальний уривок: "
            + json.dumps(
                nearest_excerpt(text, quote if isinstance(quote, str) else ""), ensure_ascii=False
            ),
            "quote",
        )


def validate_analysis(
    analysis: dict, pages: list[dict], *, expected_objective: str | None = None
) -> None:
    """Each research conclusion must cite a read URL and an exact excerpt."""
    if expected_objective is not None and normalized_quote(
        analysis.get("objective", "")
    ) != normalized_quote(expected_objective):
        raise EvidenceError(
            "Analyst: початкова мета незмінна. Поверни objective дослівно: "
            + json.dumps(expected_objective, ensure_ascii=False)
            + ". Уточнюй реалізацію початкового запиту, не замінюй задачу "
            "обробкою вебсторінок або виправленням цитат.",
            "analyst_objective",
        )
    checks = analysis.get("source_checks")
    if not isinstance(checks, list) or not checks:
        raise EvidenceError(
            "Analyst: додай source_checks: [{conclusion, url, quote}]. "
            "Кожен висновок спирається на прочитану web_pages сторінку; "
            "quote — дослівний уривок її text, не snippet.",
            "analyst_sources",
        )
    for check in checks:
        if not isinstance(check, dict) or not str(check.get("conclusion", "")).strip():
            raise EvidenceError("Analyst: потрібен змістовний conclusion.", "analyst_sources")
        page = next((page for page in pages if page["url"] == check.get("url")), None)
        if page is None:
            raise EvidenceError(
                "Analyst: url має належати успішно прочитаним web_pages: "
                + json.dumps([page["url"] for page in pages]),
                "analyst_sources",
            )
        validate_evidence({"pointer": "/text", "quote": check.get("quote")}, page, page_quote=True)


def validate_review_justifications(review: dict, context: dict) -> None:
    """Rejections need a code location, a requirement/rule and concrete impact per finding."""
    if review["decision"] == "approve":
        return
    findings = review.get("findings", [])
    justifications = review.get("justifications", [])
    if not findings or len(justifications) != len(findings):
        raise EvidenceError(
            "Reviewer: для кожного finding додай justifications із finding_index, "
            "code={pointer,quote}, basis={pointer,quote}, impact. "
            "code має посилатися лише на /implementation/source або /implementation/manifest; "
            "basis — /spec/..., /user_request або /quality_rules. "
            "Поясни конкретний наслідок, не лише стилістичну перевагу.",
            "review_basis",
        )
    indices = set()
    for item in justifications:
        index = item.get("finding_index")
        if type(index) is not int or index not in range(len(findings)) or index in indices:
            raise EvidenceError(
                "Reviewer: кожен finding_index має бути унікальним.", "review_basis"
            )
        indices.add(index)
        code = item.get("code", {})
        basis = item.get("basis", {})
        if not (
            code.get("pointer") == "/implementation/source"
            or code.get("pointer", "").startswith("/implementation/manifest/")
        ) or not (
            basis.get("pointer") in {"/user_request", "/quality_rules"}
            or basis.get("pointer", "").startswith("/spec/")
        ):
            raise EvidenceError(
                "Reviewer: докази обмежені кодом Developer та вимогами; QA недоступний.",
                "review_scope",
            )
        validate_evidence(code, context)
        validate_evidence(basis, context)
        impact = item.get("impact", "")
        if not isinstance(impact, str) or not impact.strip():
            raise EvidenceError("Reviewer: поясни конкретний impact дефекту.", "review_basis")

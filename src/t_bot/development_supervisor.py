"""Python development policy; model advice can route work but cannot approve activation."""

import re

from t_bot.errors import BotError
from t_bot.qa_expectations import NoProgressGuard, StagnationGuard
from t_bot.team_evidence import validate_evidence

SUPERVISOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"enum": ["developer", "qa", "pm", "reviewer", "stop"]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 2000},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pointer": {"type": "string", "minLength": 1, "maxLength": 300},
                    "quote": {"type": "string", "minLength": 1, "maxLength": 1000},
                },
                "required": ["pointer", "quote"],
            },
        },
    },
    "required": ["action", "reason", "evidence"],
}


def validate_advice(advice: dict, context: dict, allowed: set[str]) -> None:
    """Validate evidence and allowed routes; models cannot approve activation."""
    import jsonschema

    try:
        jsonschema.validate(advice, SUPERVISOR_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise BotError("Supervisor повернув некоректний контракт рішення.") from exc
    if not advice["reason"].strip():
        raise BotError("Supervisor має надати змістовне пояснення рішення.")
    if advice["action"] not in allowed:
        raise BotError("Supervisor запропонував недопустимий перехід для цього етапу.")
    for evidence in advice["evidence"]:
        validate_evidence(evidence, context)


class DevelopmentSupervisor:
    def __init__(self, stagnation_limit: int, no_progress_limit: int):
        self.stagnation = StagnationGuard(stagnation_limit)
        self.progress = NoProgressGuard(no_progress_limit)

    def observe_rework(
        self, evidence: dict, proposed: str, state: dict | None, milestone: int
    ) -> dict | None:
        stalled = self.progress.observe(milestone)
        if stalled is not None:
            return {"stage": "NoProgress", **stalled, "stop": True}
        repeated = self.stagnation.observe(evidence, proposed, state=state)
        return {"stage": "Stagnation", **repeated} if repeated else None

    @staticmethod
    def check_review(review: dict, source: str = "") -> None:
        if review["decision"] not in {"approve", "developer", "pm"}:
            raise BotError("Reviewer оцінює лише код Developer; якість тестів належить QA.")
        for finding in review.get("findings", []):
            qa_tokens = re.findall(
                r"\btest_(?:happy|failure|edge)_\w+|\bassert(?:Equal|AlmostEqual|Raises|In|NotEqual)\b",
                finding,
            )
            qa_quality = re.search(
                r"(?:QA|tests?|тест\w*).{0,45}(?:quality|coverage|якіст|покрит|перепис|rewrite)"
                r"|(?:quality|coverage|якіст|покрит|перепис|rewrite).{0,45}(?:QA|tests?|тест\w*)",
                finding,
                re.IGNORECASE,
            )
            if qa_quality or any(token not in source for token in qa_tokens):
                raise BotError(
                    "Reviewer має переглянути код Developer; assertions і тести оцінює QA."
                )
        if review["decision"] == "approve":
            return
        meaningful = [
            finding
            for finding in review.get("findings", [])
            if finding.strip()
            and not re.match(
                r"^(notes_for_|action_items?_for_|status_code_for_)",
                finding.strip(),
                re.IGNORECASE,
            )
        ]
        if not meaningful:
            raise BotError(
                "Supervisor повертає зауваження Reviewer: потрібен конкретний дефект "
                "із посиланням на вимогу або код Developer. Порожні findings "
                "та службові placeholders не є підставою для доопрацювання."
            )

    @staticmethod
    def needs_arbitration(review: dict, context: dict) -> bool:
        if review["decision"] == "approve":
            return False
        execution = context.get("test_results", {})
        qa = context.get("qa_triage", {})
        if execution.get("passed") is True and qa.get("decision") == "approve":
            return True
        if qa.get("decision") in {"developer", "qa", "pm"} and qa["decision"] != review["decision"]:
            return True
        return False

    @staticmethod
    def may_activate(execution: dict, qa: dict, review: dict) -> bool:
        return (
            execution.get("passed") is True
            and execution.get("tests_run", 0) >= 2
            and not execution.get("failures")
            and not execution.get("errors")
            and not execution.get("skipped")
            and not execution.get("expectedFailures")
            and qa.get("decision") == "approve"
            and review.get("decision") == "approve"
        )

"""Python development policy; model advice can route work but cannot approve activation."""

import json
import re

from t_bot.errors import BotError
from t_bot.qa_expectations import NoProgressGuard, StagnationGuard

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
        pointer = evidence["pointer"]
        try:
            if not pointer.startswith("/"):
                raise ValueError("Expected JSON pointer")
            value = context
            for part in pointer[1:].split("/"):
                key = part.replace("~1", "/").replace("~0", "~")
                if isinstance(value, list):
                    if not re.fullmatch(r"0|[1-9][0-9]*", key):
                        raise ValueError("Invalid array pointer")
                    value = value[int(key)]
                else:
                    value = value[key]
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            raise BotError("Supervisor послався на неіснуючий доказ.") from exc
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if not evidence["quote"].strip() or evidence["quote"] not in text:
            raise BotError("Цитата Supervisor не відповідає фактичному доказу.")


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
    def check_review(review: dict) -> None:
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
                "із посиланням на вимогу, assertion або результати. Порожні findings "
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
        references = context.get("trusted_checks", {}).get("reference_checks", [])
        verified = {item["test"] for item in references if item.get("matched") is True}
        findings = " ".join(review.get("findings", [])).casefold()
        numerical = re.search(
            r"точн|округл|радіан|precision|rounding|radian|oracle|еталон", findings
        )
        return bool(numerical and any(name.casefold() in findings for name in verified))

    @staticmethod
    def may_activate(execution: dict, qa: dict, review: dict) -> bool:
        return (
            execution.get("passed") is True
            and execution.get("tests_run", 0) >= 3
            and not execution.get("failures")
            and not execution.get("errors")
            and not execution.get("skipped")
            and not execution.get("expectedFailures")
            and qa.get("decision") == "approve"
            and review.get("decision") == "approve"
        )

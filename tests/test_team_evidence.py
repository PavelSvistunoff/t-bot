import copy

import pytest

from t_bot.development_supervisor import validate_advice
from t_bot.team_evidence import (
    EvidenceError,
    evidence_examples,
    validate_analysis,
    validate_review_justifications,
)

CONTEXT = {
    "implementation": {
        "source": 'if operator == "divide":\n    return a / b',
        "manifest": {"output_schema": {"properties": {"result": {"type": "number"}}}},
    },
    "spec": {"requirements": ["Reject division by zero"]},
    "user_request": "Implement arithmetic",
}


def justification():
    return {
        "decision": "developer",
        "findings": ["Division by zero is not handled"],
        "justifications": [
            {
                "finding_index": 0,
                "code": {"pointer": "/implementation/source", "quote": "return a / b"},
                "basis": {"pointer": "/spec/requirements/0", "quote": "Reject division by zero"},
                "impact": "Zero input raises an unhandled exception",
            }
        ],
    }


def test_supervisor_pointer_error_provides_copyable_paths_and_quotes():
    with pytest.raises(EvidenceError) as failure:
        validate_advice(
            {
                "action": "developer",
                "reason": "Fix division",
                "evidence": [{"pointer": "implementation.source", "quote": "invented"}],
            },
            CONTEXT,
            {"developer"},
        )
    assert failure.value.failure_kind == "pointer"
    assert "/implementation/source" in str(failure.value)
    assert "/spec/requirements/0" in str(failure.value)
    assert "return a / b" in str(failure.value)
    for item in evidence_examples(CONTEXT):
        validate_advice(
            {"action": "developer", "reason": "Inspect", "evidence": [item]}, CONTEXT, {"developer"}
        )


def test_supervisor_quote_error_shows_actual_value_without_accepting_paraphrase():
    with pytest.raises(EvidenceError) as failure:
        validate_advice(
            {
                "action": "developer",
                "reason": "Fix",
                "evidence": [{"pointer": "/implementation/source", "quote": "elif ... else ..."}],
            },
            CONTEXT,
            {"developer"},
        )
    assert failure.value.failure_kind == "quote"
    assert "return a / b" in str(failure.value)
    assert "дослівний" in str(failure.value)


def test_reviewer_requires_code_requirement_and_impact_for_each_finding():
    review = justification()
    validate_review_justifications(review, CONTEXT)
    review["findings"].append("Another unsubstantiated objection")
    with pytest.raises(EvidenceError, match="кожного finding"):
        validate_review_justifications(review, CONTEXT)


@pytest.mark.parametrize(
    "pointer", ["/test_results/passed", "/tests", "/qa_triage", "/implementation/source_extra"]
)
def test_reviewer_cannot_use_qa_or_out_of_scope_evidence(pointer):
    review = justification()
    review["justifications"][0]["code"]["pointer"] = pointer
    with pytest.raises(EvidenceError, match="QA недоступний"):
        validate_review_justifications(review, CONTEXT)


def test_reviewer_cannot_invent_required_output_field_or_duplicate_code():
    review = justification()
    review["justifications"][0]["code"] = {
        "pointer": "/implementation/manifest/output_schema",
        "quote": '"required": ["result"]',
    }
    with pytest.raises(EvidenceError, match="Цитата"):
        validate_review_justifications(review, CONTEXT)
    review["justifications"][0]["code"] = {
        "pointer": "/implementation/source",
        "quote": "else: result = a / b else:",
    }
    with pytest.raises(EvidenceError, match="Цитата"):
        validate_review_justifications(review, CONTEXT)


@pytest.mark.parametrize("defect", ["missing", "unread_url", "snippet_quote", "empty_conclusion"])
def test_analyst_must_cite_read_page_content(defect):
    pages = [{"url": "https://docs.python.org/3/", "text": "Division raises ZeroDivisionError."}]
    analysis = {
        "source_checks": [
            {
                "conclusion": "Handle division by zero",
                "url": pages[0]["url"],
                "quote": "Division raises ZeroDivisionError.",
            }
        ]
    }
    validate_analysis(analysis, pages)
    invalid = copy.deepcopy(analysis)
    if defect == "missing":
        invalid.pop("source_checks")
    elif defect == "unread_url":
        invalid["source_checks"][0]["url"] = "https://unread.example/"
    elif defect == "snippet_quote":
        invalid["source_checks"][0]["quote"] = "Found only in the search snippet"
    else:
        invalid["source_checks"][0]["conclusion"] = " "
    with pytest.raises(EvidenceError):
        validate_analysis(invalid, pages)


@pytest.mark.parametrize("finding", ["Increase QA coverage", "Покращити якість тестів"])
def test_reviewer_cannot_grade_qa_in_freeform_prose(finding):
    from t_bot.development_supervisor import DevelopmentSupervisor
    from t_bot.errors import BotError

    with pytest.raises(BotError, match="QA"):
        DevelopmentSupervisor.check_review({"decision": "developer", "findings": [finding]})

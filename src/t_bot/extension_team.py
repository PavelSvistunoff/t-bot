"""Persistent, bounded PM/Analyst/Developer/QA/Reviewer extension workflow."""

import asyncio
import copy
import hashlib
import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import uuid4

from t_bot.config import Settings
from t_bot.development_progress import DevelopmentEvent, ProgressCallback, ProgressReporter
from t_bot.errors import BotError
from t_bot.image_quality import (
    audit_image_tests,
    image_contract,
    image_review_schema,
    validate_size_claims,
)
from t_bot.ollama import Ollama
from t_bot.qa_expectations import (
    NoProgressGuard,
    StagnationGuard,
    artifact_fingerprint,
    audit_specification,
    audit_tests,
)
from t_bot.team_agents import RoleAgents
from t_bot.web_search import WebSearch

logger = logging.getLogger(__name__)

SPEC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "objective": {"type": "string", "minLength": 1, "maxLength": 2000},
        **{
            name: {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "items": {"type": "string", "maxLength": 1000},
            }
            for name in (
                "requirements",
                "acceptance_criteria",
                "happy_path",
                "failure_path",
                "edge_cases",
            )
        },
    },
    "required": [
        "objective",
        "requirements",
        "acceptance_criteria",
        "happy_path",
        "failure_path",
        "edge_cases",
    ],
}
QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
        }
    },
    "required": ["queries"],
}
QA_SCHEMA = {
    "type": "object",
    "properties": {
        "tests": {
            "type": "array",
            "minItems": 1,
            "maxItems": 300,
            "items": {"type": "string", "maxLength": 500},
        },
        "notes": {"type": "string", "maxLength": 4000},
    },
    "required": ["tests", "notes"],
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"enum": ["approve", "developer", "qa", "pm"]},
        "findings": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "maxLength": 1000},
        },
    },
    "required": ["decision", "findings"],
}


TEST_REVIEW_SCHEMA: dict = copy.deepcopy(REVIEW_SCHEMA)
TEST_REVIEW_SCHEMA["properties"]["decision"] = {"enum": ["approve", "qa", "pm"]}


class TeamFailure(BotError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__(
            f"Конвеєр {report['id']} не завершив розширення: "
            f"{report.get('error', 'невідома помилка')}. "
            f"Артефакти збережено; /build {report['id']}"
        )


class ExtensionTeam:
    def __init__(self, settings: Settings, ollama: Ollama):
        self.settings = settings
        self.roles = RoleAgents(ollama)
        self.web = WebSearch(settings.web_search_timeout_seconds, settings.web_search_max_results)
        self.directory = settings.data_dir / "team_runs"

    async def save(self, report: dict) -> None:
        def write() -> None:
            directory = self.directory / report["id"]
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary = directory / "report.tmp"
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            temporary.chmod(0o600)
            os.replace(temporary, directory / "report.json")

        await asyncio.to_thread(write)

    @staticmethod
    def _load(path: Path) -> dict:
        try:
            report = json.loads(path.read_text())
            required = {"id", "user_id", "status", "stage", "extension_id"}
            if not isinstance(report, dict) or not required <= report.keys():
                raise ValueError("Missing report fields")
            if report["id"] != path.parent.name:
                raise ValueError("Report ID differs from directory")
            return report
        except (OSError, ValueError) as exc:
            raise BotError("Збережений звіт конвеєра пошкоджений або недоступний.") from exc

    def reports(self, user_id: int) -> list[dict]:
        self.settings.authorize(user_id)
        if not self.directory.exists():
            return []
        reports = []
        for path in sorted(
            self.directory.glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                data = self._load(path)
            except BotError:
                logger.warning("team_report.unreadable")
                continue
            if data["user_id"] == user_id:
                reports.append(
                    {k: data.get(k) for k in ("id", "extension_id", "status", "stage", "error")}
                )
        return reports[:20]

    def report(self, build_id: str, user_id: int) -> dict:
        self.settings.authorize(user_id)
        if len(build_id) != 32 or any(c not in "0123456789abcdef" for c in build_id):
            raise BotError("Некоректний ID конвеєра.")
        path = self.directory / build_id / "report.json"
        if not path.is_file():
            raise BotError("Конвеєр не знайдено.")
        report = self._load(path)
        if report["user_id"] != user_id:
            raise BotError("Доступ до конвеєра заборонено.")
        return report

    async def recover(self, active_extensions: dict[tuple[str, int], str] | None = None) -> None:
        if not await asyncio.to_thread(self.directory.exists):
            return
        for path in await asyncio.to_thread(lambda: list(self.directory.glob("*/report.json"))):
            try:
                report = await asyncio.to_thread(self._load, path)
            except BotError:
                logger.warning("team_report.unreadable")
                continue
            if report["status"] in {"running", "approved"}:
                candidate = report.get("candidate", {})
                manifest = candidate.get("manifest", {})
                key = (manifest.get("id", ""), manifest.get("version", 0))
                digest = hashlib.sha256(candidate.get("source", "").encode()).hexdigest()
                if (active_extensions or {}).get(key) == digest:
                    report.update(status="active", stage="completed", version=key[1])
                    await self.save(report)
                    continue
                report.update(
                    status="interrupted", error="Процес бота завершився під час конвеєра."
                )
                await self.save(report)

    async def run(
        self,
        *,
        request: str,
        extension_id: str,
        user_id: int,
        model: str,
        developer_context: str,
        developer_schema: dict,
        validate: Callable[[dict], Awaitable[dict]],
        execute_tests: Callable[[dict, str], Awaitable[dict]],
        progress: ProgressCallback | None = None,
        reference_values: Callable[[dict, str, dict], Awaitable[dict]] | None = None,
    ) -> dict:
        self.settings.authorize(user_id)
        report: dict = {
            "id": uuid4().hex,
            "extension_id": extension_id,
            "user_id": user_id,
            "request": request,
            "status": "running",
            "stage": "PM",
            "events": [],
            "calls": 0,
        }
        await self.save(report)
        reporter = ProgressReporter(progress)
        reporter.emit(DevelopmentEvent("pipeline", "started", report["id"]))

        async def record(stage: str, data: dict) -> None:
            report["stage"] = stage
            report["events"].append({"stage": stage, "data": data})
            await self.save(report)

        async def ask(
            role: str,
            context: str,
            schema: dict,
            phase: str,
            check: Callable[[dict], Awaitable[None]] | None = None,
        ) -> dict:
            errors: list[str] = []
            models = [
                model,
                *[m for m in self.settings.extension_team_fallback_models if m != model],
            ]
            for attempt in range(self.settings.extension_team_stage_attempts):
                if report["calls"] >= self.settings.extension_team_max_calls:
                    raise BotError("Вичерпано бюджет викликів команди.")
                selected = models[min(attempt, len(models) - 1)]
                report["calls"] += 1
                report["stage"] = role
                await self.save(report)
                event = DevelopmentEvent(
                    phase,
                    "started",
                    report["id"],
                    selected,
                    attempt + 1,
                    self.settings.extension_team_stage_attempts,
                )
                reporter.emit(event)
                try:
                    async with asyncio.timeout(self.settings.model_timeout_seconds):
                        retry_feedback = (
                            "\nPREVIOUS ATTEMPT FAILED; FIX THIS ERROR:\n" + errors[-1]
                            if errors
                            else ""
                        )
                        result = await reporter.wait(
                            self.roles.ask(role, selected, context + retry_feedback, schema), event
                        )
                    await record(role, {"model": selected, "output": result})
                    if check is not None:
                        await check(result)
                    detail = ""
                    if "decision" in result:
                        decision = {
                            "approve": "Схвалено.",
                            "developer": "Потрібні правки Developer.",
                            "qa": "Потрібні правки QA.",
                            "pm": "Потрібне уточнення ТЗ від PM.",
                        }
                        detail = decision.get(result["decision"], "")
                    reporter.emit(DevelopmentEvent(phase, "completed", report["id"], detail=detail))
                    return result
                except (BotError, ValueError, TimeoutError) as exc:
                    reporter.emit(
                        DevelopmentEvent(
                            phase,
                            "failed",
                            report["id"],
                            selected,
                            attempt + 1,
                            self.settings.extension_team_stage_attempts,
                            "Причину збережено у звіті конвеєра.",
                        )
                    )
                    errors.append(str(exc)[:1000] or type(exc).__name__)
                    await record(
                        role,
                        {
                            "attempt": attempt + 1,
                            "error": errors[-1],
                            "response": getattr(exc, "response", ""),
                        },
                    )
                    if (
                        len(models) == 1
                        and attempt + 1 < self.settings.extension_team_stage_attempts
                    ):
                        try:
                            available = await self.roles.ollama.models()
                            alternatives = [
                                m for m in available if m != model and "embedding" not in m
                            ]
                            alternatives.sort(key=lambda m: ("coder" not in m.casefold(), m))
                            models.extend(alternatives)
                        except BotError:
                            pass  # Keep retrying the primary model if discovery is unavailable.
            raise BotError(f"{role}: усі спроби завершилися помилкою. {errors[-1]}")

        guard = StagnationGuard(self.settings.extension_team_stagnation_limit)
        progress_guard = NoProgressGuard(self.settings.extension_team_no_progress_limit)
        milestone = 0

        async def advance_progress(level: int, stage: str) -> None:
            nonlocal milestone
            if level > milestone:
                milestone = level
                report["progress"] = {"milestone": level, "stage": stage}
                await record("Milestone", report["progress"])

        async def checked_route(
            evidence: dict,
            proposed: str,
            *,
            candidate: dict | None = None,
            tests: str = "",
            phase: str = "",
        ) -> str:
            stalled = progress_guard.observe(milestone)
            if stalled is not None:
                await record("NoProgress", stalled)
                raise BotError(
                    "Немає прогресу між версіями коду та тестів: "
                    f"{stalled['failed_rounds']} раундів без нового контрольного етапу. "
                    "Усі спроби та останній валідний кандидат (якщо він є) збережено."
                )
            state = (
                {"phase": phase, **artifact_fingerprint(candidate, tests)}
                if candidate is not None
                else None
            )
            escalation = guard.observe(evidence, proposed, state=state)
            if escalation is None:
                return proposed
            await record("Stagnation", escalation)
            if escalation["stop"]:
                artifacts = (
                    "Останній валідний кандидат і всі спроби збережено."
                    if report.get("last_valid_candidate")
                    else "Кандидата ще не створено; ТЗ і всі спроби збережено."
                )
                raise BotError(
                    "Зациклення: повторюються однакові помилки після доопрацювань. " + artifacts
                )
            return escalation["target"]

        async def check_specification(specification: dict) -> None:
            current_candidate = report.get("last_valid_candidate")
            oracle = (
                await reference_values(current_candidate, "", specification)
                if reference_values is not None and current_candidate
                else {}
            )
            audit = audit_specification(
                request,
                specification,
                semantics=oracle.get("semantics"),
                references=oracle.get("values"),
            )
            await record("SpecificationAudit", audit)
            if not audit["passed"]:
                raise BotError(
                    "ТЗ не виправлено. Змініть конкретні пункти з corrections, "
                    "не додавайте протилежні твердження в інші розділи. "
                    + json.dumps(audit, ensure_ascii=False)
                )

        try:
            async with asyncio.timeout(self.settings.extension_team_timeout_seconds):
                draft = await ask(
                    "PM",
                    "USER REQUEST:\n"
                    + request
                    + "\nHOST CONTRACT AND CONSTRAINTS:\n"
                    + developer_context,
                    SPEC_SCHEMA,
                    "pm_draft",
                )
                plan = await ask(
                    "Analyst",
                    "Plan official documentation searches:\n" + json.dumps(draft),
                    QUERY_SCHEMA,
                    "analyst_plan",
                )
                reporter.emit(DevelopmentEvent("research", "started", report["id"]))
                research = []
                for query in plan["queries"]:
                    try:
                        search = await asyncio.to_thread(
                            self.web.search, query, min(3, self.settings.web_search_max_results)
                        )
                        research.append(search)
                    except (BotError, OSError) as exc:
                        research.append({"query": query, "error": str(exc)[:1000]})
                await record("WebSearch", {"research": research})
                sources = sum(len(item.get("results", [])) for item in research)
                reporter.emit(
                    DevelopmentEvent(
                        "research",
                        "completed" if sources else "failed",
                        report["id"],
                        detail=f"Знайдено джерел: {sources}.",
                    )
                )
                if not any(item.get("results") for item in research):
                    raise BotError("Аналітик не отримав джерел: вебпошук недоступний або порожній.")
                analysis = await ask(
                    "Analyst",
                    json.dumps({"draft": draft, "web_results": research}),
                    SPEC_SCHEMA,
                    "analyst_revision",
                )
                final_spec = await ask(
                    "PM",
                    "Approve corrections and finalize:\n"
                    + json.dumps(
                        {
                            "user_request": request,
                            "draft": draft,
                            "analyst_revision": analysis,
                        }
                    ),
                    SPEC_SCHEMA,
                    "pm_final",
                )
                report["final_spec"] = final_spec
                report["spec_revisions"] = [copy.deepcopy(final_spec)]
                await self.save(report)
                guidelines = Path.cwd() / "AGENTS.md"
                clean_code = (
                    await asyncio.to_thread(guidelines.read_text) if guidelines.is_file() else ""
                )
                feedback: dict = {}
                candidate: dict = {}
                last_valid: dict = {}
                tests = ""
                specification_audit = audit_specification(request, final_spec)
                await record("SpecificationAudit", specification_audit)
                route = "developer" if specification_audit["passed"] else "pm"
                feedback = (
                    {} if specification_audit["passed"] else {"specification": specification_audit}
                )
                for iteration in range(self.settings.extension_team_max_rounds):
                    if route == "pm":
                        final_spec = await ask(
                            "PM",
                            json.dumps(
                                {
                                    "instruction": "Repair contradictions in the task or tests. "
                                    "Keep the original request and requirements. "
                                    "Never change them to fit bugs. "
                                    "Use reference evidence. "
                                    "Apply the supplied corrections to the EXACT section/index. "
                                    "Do not leave the old contradictory example elsewhere. "
                                    "Preserve unrelated requirements and return the full task.",
                                    "user_request": request,
                                    "current_spec": final_spec,
                                    "feedback": feedback,
                                }
                            ),
                            SPEC_SCHEMA,
                            "pm_revision",
                            check=check_specification,
                        )
                        report["final_spec"] = final_spec
                        report["spec_revisions"].append(copy.deepcopy(final_spec))
                        await self.save(report)
                        route = "developer"
                        tests = ""
                    if route == "developer":
                        proposal = await ask(
                            "Developer",
                            developer_context
                            + "\nFINAL TECHNICAL TASK:\n"
                            + json.dumps(final_spec)
                            + "\nREPOSITORY CLEAN CODE RULES:\n"
                            + clean_code[:20000]
                            + "\nPREVIOUS SOURCE AND FEEDBACK:\n"
                            + json.dumps({"candidate": candidate, "feedback": feedback}),
                            developer_schema,
                            "developer",
                        )
                        report["attempt_candidate"] = proposal
                        await self.save(report)
                        reporter.emit(DevelopmentEvent("validation", "started", report["id"]))
                        try:
                            candidate = await validate(proposal)
                            last_valid = copy.deepcopy(candidate)
                            report["last_valid_candidate"] = last_valid
                            report["candidate"] = copy.deepcopy(candidate)
                            await self.save(report)
                            await record(
                                "Validation",
                                {
                                    "passed": True,
                                    "smoke_result": candidate.get("smoke_result"),
                                },
                            )
                            await advance_progress(1, "candidate_validated")
                            reporter.emit(DevelopmentEvent("validation", "completed", report["id"]))
                        except (BotError, ValueError, OSError) as exc:
                            candidate = copy.deepcopy(last_valid)
                            feedback = {"validation_error": str(exc)[-3000:]}
                            report["candidate"] = candidate or proposal
                            await record("Validation", feedback)
                            route = await checked_route(
                                feedback,
                                "developer",
                                candidate=proposal,
                                phase="validation",
                            )
                            if route == "qa":
                                route = "pm"
                            reporter.emit(DevelopmentEvent("validation", "failed", report["id"]))
                            reporter.emit(
                                DevelopmentEvent(
                                    "rework",
                                    "rework",
                                    report["id"],
                                    detail=f"Передано {'PM' if route == 'pm' else 'Developer'}. "
                                    f"Раунд {iteration + 1}.",
                                )
                            )
                            continue
                    qa = await ask(
                        "QA",
                        json.dumps(
                            {
                                "spec": final_spec,
                                "implementation": candidate,
                                "feedback": feedback,
                                "previous_tests": tests,
                                "image_contract": image_contract(request, final_spec),
                            }
                        ),
                        QA_SCHEMA,
                        "qa_write",
                    )
                    tests = qa["tests"]
                    report["candidate"] = candidate
                    report["tests"] = tests
                    await self.save(report)
                    oracle = (
                        await reference_values(candidate, tests, final_spec)
                        if reference_values is not None
                        else {}
                    )
                    expectations = audit_tests(
                        request,
                        final_spec,
                        tests,
                        semantics=oracle.get("semantics"),
                        references=oracle.get("values"),
                    )
                    contract = oracle.get("image_contract", image_contract(request, final_spec))
                    if contract["enabled"] and expectations["passed"]:
                        image_audit = audit_image_tests(tests, contract)
                        expectations["image_checks"] = image_audit
                        if not image_audit["passed"]:
                            expectations.update(
                                passed=False, target="qa", findings=image_audit["findings"]
                            )
                    await record("QAExpectations", expectations)
                    if not expectations["passed"]:
                        feedback = {"expectations": expectations}
                        route = await checked_route(
                            expectations,
                            expectations["target"],
                            candidate=candidate,
                            tests=tests,
                            phase="qa_expectations",
                        )
                        target = "PM" if route == "pm" else "QA"
                        await record(
                            "Rework",
                            {"target": route, "round": iteration + 1, "feedback": feedback},
                        )
                        reporter.emit(
                            DevelopmentEvent(
                                "rework",
                                "rework",
                                report["id"],
                                detail=f"Передано {target}: суперечності в очікуваннях тестів.",
                            )
                        )
                        continue
                    await advance_progress(2, "expectations_audit_passed")

                    async def check_image_review(review: dict, facts: dict = contract) -> None:
                        if facts["enabled"]:
                            validate_size_claims(review, facts)

                    test_review = await ask(
                        "Reviewer",
                        json.dumps(
                            {
                                "phase": "qa_expectations_review",
                                "user_request": request,
                                "spec": final_spec,
                                "manifest": candidate["manifest"],
                                "tests": tests,
                                "trusted_checks": expectations,
                                "image_contract": contract,
                                "instruction": "Review ONLY test oracles against requirements. "
                                "Verify numbers and exception contracts independently. "
                                "Do not require unspecified prose. Route bad tests to qa, "
                                "spec contradictions to pm. Approval here is NOT code approval. "
                                "Use trusted_checks as evidence. Do not reject a verified numeric "
                                "oracle because of rounded comments or hypothetical float errors. "
                                "Any disputed assertion needs a concrete conflicting reference.",
                            }
                        ),
                        image_review_schema(TEST_REVIEW_SCHEMA)
                        if contract["enabled"]
                        else TEST_REVIEW_SCHEMA,
                        "qa_expectations",
                        check=check_image_review,
                    )
                    await record("TestReview", test_review)
                    if test_review["decision"] != "approve":
                        feedback = {"test_review": test_review}
                        route = await checked_route(
                            feedback,
                            test_review["decision"],
                            candidate=candidate,
                            tests=tests,
                            phase="test_review",
                        )
                        await record(
                            "Rework",
                            {"target": route, "round": iteration + 1, "feedback": feedback},
                        )
                        reporter.emit(
                            DevelopmentEvent(
                                "rework",
                                "rework",
                                report["id"],
                                detail=f"Передано {'PM' if route == 'pm' else 'QA'}: "
                                "зауваження до тестів.",
                            )
                        )
                        continue
                    await advance_progress(3, "test_review_approved")
                    reporter.emit(DevelopmentEvent("qa_execute", "started", report["id"]))
                    execution = await execute_tests(candidate, tests)
                    await record("QAExecution", execution)
                    if execution.get("tests_run", 0) > 0:
                        await advance_progress(4, "tests_executed")
                    if execution.get("passed"):
                        await advance_progress(5, "tests_passed")
                    reporter.emit(
                        DevelopmentEvent(
                            "qa_execute",
                            "completed" if execution.get("passed") else "failed",
                            report["id"],
                            detail=f"Виконано тестів: {execution.get('tests_run', 0)}.",
                        )
                    )
                    if execution.get("target") == "qa":
                        route = await checked_route(
                            execution,
                            "qa",
                            candidate=candidate,
                            tests=tests,
                            phase="qa_execution",
                        )
                        feedback = {"execution": execution}
                        await record(
                            "Rework",
                            {"target": route, "round": iteration + 1, "feedback": feedback},
                        )
                        reporter.emit(
                            DevelopmentEvent(
                                "rework",
                                "rework",
                                report["id"],
                                detail=f"Передано {'PM' if route == 'pm' else 'QA'}. "
                                f"Раунд {iteration + 1}.",
                            )
                        )
                        continue
                    triage = await ask(
                        "QA",
                        json.dumps(
                            {
                                "phase": "triage",
                                "spec": final_spec,
                                "implementation": candidate,
                                "tests": tests,
                                "test_results": execution,
                                "image_contract": contract,
                                "instruction": "Approve only if tests passed. Route code bugs to "
                                "developer, test defects to qa, and contradictory requirements "
                                "to pm; explain findings.",
                            }
                        ),
                        image_review_schema(REVIEW_SCHEMA)
                        if contract["enabled"]
                        else REVIEW_SCHEMA,
                        "qa_triage",
                        check=check_image_review,
                    )
                    review = await ask(
                        "Reviewer",
                        json.dumps(
                            {
                                "spec": final_spec,
                                "implementation": candidate,
                                "tests": tests,
                                "test_results": execution,
                                "image_contract": contract,
                                "qa_notes": qa["notes"],
                                "qa_triage": triage,
                            }
                        ),
                        image_review_schema(REVIEW_SCHEMA)
                        if contract["enabled"]
                        else REVIEW_SCHEMA,
                        "reviewer",
                        check=check_image_review,
                    )
                    feedback = {"execution": execution, "qa": triage, "review": review}
                    if (
                        execution.get("passed")
                        and triage["decision"] == "approve"
                        and review["decision"] == "approve"
                    ):
                        report.update(
                            status="approved", stage="completed", candidate=candidate, tests=tests
                        )
                        await self.save(report)
                        return {
                            "candidate": candidate,
                            "tests": tests,
                            "team_run_id": report["id"],
                            "qa": execution,
                            "review": review,
                            "qa_triage": triage,
                        }
                    route = (
                        "pm"
                        if "pm" in {review["decision"], triage["decision"]}
                        else review["decision"]
                    )
                    if route == "approve":
                        route = (
                            triage["decision"] if triage["decision"] != "approve" else "developer"
                        )
                    route = await checked_route(
                        execution,
                        route,
                        candidate=candidate,
                        tests=tests,
                        phase="code_review" if execution.get("passed") else "qa_execution",
                    )
                    await record(
                        "Rework", {"target": route, "round": iteration + 1, "feedback": feedback}
                    )
                    target = {"qa": "QA", "pm": "PM", "developer": "Developer"}[route]
                    reporter.emit(
                        DevelopmentEvent(
                            "rework",
                            "rework",
                            report["id"],
                            detail=f"Передано {target}. Раунд {iteration + 1}.",
                        )
                    )
                raise BotError(
                    "Вичерпано раунди доопрацювання; QA або reviewer не підтвердили якість."
                )
        except asyncio.CancelledError:
            report.update(
                status="cancelled", error="Конвеєр скасовано; поточні артефакти збережено."
            )
            await self.save(report)
            reporter.emit(DevelopmentEvent("pipeline", "cancelled", report["id"]))
            raise
        except Exception as exc:
            error = (
                "Перевищено загальний час конвеєра."
                if isinstance(exc, TimeoutError)
                else str(exc)[-2000:] or type(exc).__name__
            )
            report.update(status="failed", error=error)
            await self.save(report)
            reporter.emit(
                DevelopmentEvent(
                    "pipeline",
                    "failed",
                    report["id"],
                    detail="Роботу зупинено; артефакти та причину збережено.",
                )
            )
            raise TeamFailure(report) from exc

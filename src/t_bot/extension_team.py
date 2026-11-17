"""Persistent, bounded PM/Analyst/Developer/QA/Reviewer extension workflow."""

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import uuid4

from t_bot.config import Settings
from t_bot.errors import BotError
from t_bot.ollama import Ollama
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
        "decision": {"enum": ["approve", "developer", "qa"]},
        "findings": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "maxLength": 1000},
        },
    },
    "required": ["decision", "findings"],
}


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

        async def record(stage: str, data: dict) -> None:
            report["stage"] = stage
            report["events"].append({"stage": stage, "data": data})
            await self.save(report)

        async def ask(role: str, context: str, schema: dict) -> dict:
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
                try:
                    async with asyncio.timeout(self.settings.model_timeout_seconds):
                        retry_feedback = (
                            "\nPREVIOUS ATTEMPT FAILED; FIX THIS ERROR:\n" + errors[-1]
                            if errors
                            else ""
                        )
                        result = await self.roles.ask(
                            role, selected, context + retry_feedback, schema
                        )
                    await record(role, {"model": selected, "output": result})
                    return result
                except (BotError, ValueError, TimeoutError) as exc:
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

        try:
            async with asyncio.timeout(self.settings.extension_team_timeout_seconds):
                draft = await ask(
                    "PM",
                    "USER REQUEST:\n"
                    + request
                    + "\nHOST CONTRACT AND CONSTRAINTS:\n"
                    + developer_context,
                    SPEC_SCHEMA,
                )
                plan = await ask(
                    "Analyst",
                    "Plan official documentation searches:\n" + json.dumps(draft),
                    QUERY_SCHEMA,
                )
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
                if not any(item.get("results") for item in research):
                    raise BotError("Аналітик не отримав джерел: вебпошук недоступний або порожній.")
                analysis = await ask(
                    "Analyst", json.dumps({"draft": draft, "web_results": research}), SPEC_SCHEMA
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
                )
                report["final_spec"] = final_spec
                await self.save(report)
                guidelines = Path.cwd() / "AGENTS.md"
                clean_code = (
                    await asyncio.to_thread(guidelines.read_text) if guidelines.is_file() else ""
                )
                feedback: dict = {}
                candidate: dict = {}
                tests = ""
                route = "developer"
                for iteration in range(self.settings.extension_team_max_rounds):
                    if route == "developer":
                        candidate = await ask(
                            "Developer",
                            developer_context
                            + "\nFINAL TECHNICAL TASK:\n"
                            + json.dumps(final_spec)
                            + "\nREPOSITORY CLEAN CODE RULES:\n"
                            + clean_code[:20000]
                            + "\nPREVIOUS SOURCE AND FEEDBACK:\n"
                            + json.dumps({"candidate": candidate, "feedback": feedback}),
                            developer_schema,
                        )
                        report["candidate"] = candidate
                        await self.save(report)
                        try:
                            candidate = await validate(candidate)
                            await record("Validation", {"passed": True})
                        except (BotError, ValueError, OSError) as exc:
                            feedback = {"validation_error": str(exc)[-3000:]}
                            await record("Validation", feedback)
                            continue
                    qa = await ask(
                        "QA",
                        json.dumps(
                            {
                                "spec": final_spec,
                                "implementation": candidate,
                                "feedback": feedback,
                                "previous_tests": tests,
                            }
                        ),
                        QA_SCHEMA,
                    )
                    tests = qa["tests"]
                    report["candidate"] = candidate
                    report["tests"] = tests
                    await self.save(report)
                    execution = await execute_tests(candidate, tests)
                    await record("QAExecution", execution)
                    if execution.get("target") == "qa":
                        route = "qa"
                        feedback = {"execution": execution}
                        await record(
                            "Rework",
                            {"target": route, "round": iteration + 1, "feedback": feedback},
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
                                "instruction": "Approve only if tests passed. Route code bugs to "
                                "developer and test defects to qa; explain findings.",
                            }
                        ),
                        REVIEW_SCHEMA,
                    )
                    review = await ask(
                        "Reviewer",
                        json.dumps(
                            {
                                "spec": final_spec,
                                "implementation": candidate,
                                "tests": tests,
                                "test_results": execution,
                                "qa_notes": qa["notes"],
                                "qa_triage": triage,
                            }
                        ),
                        REVIEW_SCHEMA,
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
                    route = review["decision"]
                    if route == "approve":
                        route = (
                            triage["decision"] if triage["decision"] != "approve" else "developer"
                        )
                    await record(
                        "Rework", {"target": route, "round": iteration + 1, "feedback": feedback}
                    )
                raise BotError(
                    "Вичерпано раунди доопрацювання; QA або reviewer не підтвердили якість."
                )
        except asyncio.CancelledError:
            report.update(
                status="cancelled", error="Конвеєр скасовано; поточні артефакти збережено."
            )
            await self.save(report)
            raise
        except Exception as exc:
            error = (
                "Перевищено загальний час конвеєра."
                if isinstance(exc, TimeoutError)
                else str(exc)[-2000:] or type(exc).__name__
            )
            report.update(status="failed", error=error)
            await self.save(report)
            raise TeamFailure(report) from exc

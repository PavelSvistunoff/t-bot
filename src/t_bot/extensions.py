"""Versioned extension lifecycle. Only validated descriptions enter the ADK graph."""

import ast
import asyncio
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Literal

import jsonschema
from pydantic import BaseModel, Field, ValidationError

from t_bot.config import Settings
from t_bot.errors import BotError
from t_bot.manifest import Manifest
from t_bot.ollama import Ollama
from t_bot.sandbox import Sandbox
from t_bot.store import Store

TOOL_TEMPLATE = '''def invoke(payload: dict) -> dict:
    """Describe this operation and its returned fields."""
    return {"result": payload["text"].upper()}
'''
AGENT_TEMPLATE = """from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm


def build_agent(model: str, api_base: str) -> LlmAgent:
    return LlmAgent(
        name="EXTENSION_ID",
        description="Describe the role",
        instruction="Describe the task and expected answer",
        model=LiteLlm(model="ollama_chat/" + model, api_base=api_base),
    )
"""


class AgentDescription(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    description: str = Field(max_length=2000)
    instruction: str = Field(min_length=1, max_length=16000)


class ExtensionRecord(BaseModel):
    manifest: Manifest
    digest: str
    agent: AgentDescription | None = None
    active: bool = True


class Extensions:
    def __init__(self, settings: Settings, store: Store, sandbox: Sandbox, ollama: Ollama):
        self.settings = settings
        self.store = store
        self.sandbox = sandbox
        self.ollama = ollama
        self.lock = asyncio.Lock()

    def records(self, *, active_only: bool = True) -> list[ExtensionRecord]:
        query = "SELECT manifest, active FROM extensions WHERE deleted=0"
        if active_only:
            query += " AND active=1"
        return [
            ExtensionRecord.model_validate_json(row[0]).model_copy(
                update={"active": bool(row[1])},
            )
            for row in self.store.connection.execute(query)
        ]

    def get(self, extension_id: str, version: int | None = None) -> ExtensionRecord:
        clause = "active=1" if version is None else "version=?"
        args = (extension_id,) if version is None else (extension_id, version)
        row = self.store.connection.execute(
            f"SELECT manifest,active FROM extensions WHERE id=? AND deleted=0 AND {clause}",
            args,
        ).fetchone()
        if row is None:
            raise BotError("Розширення або версію не знайдено.")
        return ExtensionRecord.model_validate_json(row[0]).model_copy(
            update={"active": bool(row[1])},
        )

    def next_version(self, extension_id: str) -> int:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", extension_id):
            raise BotError("Некоректний ID розширення.")
        row = self.store.connection.execute(
            "SELECT COALESCE(MAX(version),0)+1 FROM extensions WHERE id=?",
            (extension_id,),
        ).fetchone()
        next_version = int(row[0])
        directory = self.settings.extensions_dir / extension_id
        if directory.exists() and not directory.is_symlink():
            versions = [int(path.name) for path in directory.iterdir() if path.name.isdecimal()]
            if versions:
                next_version = max(next_version, max(versions) + 1)
        return next_version

    def directory(self, manifest: Manifest) -> Path:
        path = self.settings.extensions_dir / manifest.id / str(manifest.version)
        if not path.resolve().is_relative_to(self.settings.extensions_dir):
            raise BotError("Шлях розширення виходить за межі сховища.")
        if any(part.is_symlink() for part in (path.parent, path)):
            raise BotError("Символічні посилання в сховищі розширень заборонені.")
        return path

    def check_graph(self, records: list[ExtensionRecord]) -> None:
        nodes = {record.manifest.id: record.manifest for record in records}
        for node in nodes.values():
            depth, current, seen = 0, node, set()
            while current.parent != "root":
                if current.id in seen:
                    raise BotError("Цикл у графі субагентів.")
                seen.add(current.id)
                parent = nodes.get(current.parent)
                if parent is None or parent.kind == "tool":
                    raise BotError(f"Батьківський агент {current.parent} не активний.")
                current = parent
                depth += 1
                if depth >= self.settings.max_delegation_depth:
                    raise BotError("Перевищено глибину делегування.")

    async def validate_source(self, manifest: Manifest, source: str, directory: Path) -> dict:
        if len(source.encode()) > 64000:
            raise BotError("Код розширення завеликий.")
        try:
            tree = ast.parse(source)
            if not any(
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name == manifest.entrypoint
                for node in tree.body
            ):
                raise ValueError("Entrypoint must be a top-level function")
            jsonschema.Draft202012Validator.check_schema(manifest.input_schema)
            jsonschema.Draft202012Validator.check_schema(manifest.output_schema)
            (directory / "extension.py").write_text(source)
            return await self.sandbox.execute(directory, manifest.model_dump(), "validate", {})
        except (SyntaxError, ValueError, jsonschema.SchemaError) as exc:
            raise BotError(f"Некоректний контракт розширення: {str(exc)[:500]}") from exc

    async def install(self, manifest: Manifest, source: str, user_id: int) -> ExtensionRecord:
        self.settings.authorize(user_id)
        async with self.lock:
            if manifest.version != self.next_version(manifest.id):
                raise BotError("Версія вже існує або не є наступною.")
            if manifest.kind != "tool":
                await self.ollama.require_model(manifest.model)
            target = self.directory(manifest)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with tempfile.TemporaryDirectory(dir=target.parent, prefix="staging-") as temporary:
                staging = Path(temporary)
                result = await self.validate_source(manifest, source, staging)
                agent = AgentDescription.model_validate(result) if manifest.kind != "tool" else None
                if agent and agent.name != manifest.id:
                    raise BotError("Ім’я агента не відповідає маніфесту.")
                record = ExtensionRecord(
                    manifest=manifest,
                    digest=hashlib.sha256(source.encode()).hexdigest(),
                    agent=agent,
                )
                self.check_graph(
                    [r for r in self.records() if r.manifest.id != manifest.id] + [record]
                )
                (staging / "manifest.json").write_text(manifest.model_dump_json(indent=2))
                await asyncio.to_thread(staging.rename, target)
            with self.store.transaction() as db:
                db.execute("UPDATE extensions SET active=0 WHERE id=?", (manifest.id,))
                db.execute(
                    "INSERT INTO extensions(id,version,manifest,active) VALUES (?,?,?,1)",
                    (manifest.id, manifest.version, record.model_dump_json()),
                )
            self.store.audit(user_id, "extension.install", f"{manifest.id}:{manifest.version}")
            return record

    async def invoke(self, record: ExtensionRecord, payload: dict, user_id: int) -> dict:
        self.settings.authorize(user_id)
        path = self.directory(record.manifest)
        if hashlib.sha256((path / "extension.py").read_bytes()).hexdigest() != record.digest:
            raise BotError("Код розширення змінився після перевірки. Створіть нову версію.")
        try:
            jsonschema.validate(payload, record.manifest.input_schema)
        except jsonschema.ValidationError as exc:
            raise BotError(f"Некоректні аргументи інструмента: {exc.message[:300]}") from exc
        self.store.audit(user_id, "extension.invoke", record.manifest.id)
        result = await self.sandbox.execute(path, record.manifest.model_dump(), "invoke", payload)
        try:
            jsonschema.validate(result, record.manifest.output_schema)
        except jsonschema.ValidationError as exc:
            raise BotError("Результат не відповідає output_schema інструмента.") from exc
        return result

    async def manage(
        self,
        extension_id: str,
        action: Literal["enable", "disable", "delete", "rollback"],
        user_id: int,
        version: int | None = None,
    ) -> str:
        self.settings.authorize(user_id)
        async with self.lock:
            active = self.records()
            if action in {"enable", "rollback"}:
                if version is None:
                    raise BotError("Вкажіть версію для активації або відкату.")
                record = self.get(extension_id, version)
                path = self.directory(record.manifest)
                source = (path / "extension.py").read_text()
                if hashlib.sha256(source.encode()).hexdigest() != record.digest:
                    raise BotError("Контрольна сума версії не збігається.")
                await self.sandbox.execute(path, record.manifest.model_dump(), "validate", {})
                self.check_graph([r for r in active if r.manifest.id != extension_id] + [record])
                with self.store.transaction() as db:
                    db.execute("UPDATE extensions SET active=0 WHERE id=?", (extension_id,))
                    db.execute(
                        "UPDATE extensions SET active=1 WHERE id=? AND version=?",
                        (extension_id, version),
                    )
            elif action in {"disable", "delete"}:
                if not any(r.manifest.id == extension_id for r in self.records(active_only=False)):
                    raise BotError("Розширення не знайдено.")
                self.check_graph([r for r in active if r.manifest.id != extension_id])
                with self.store.transaction() as db:
                    db.execute("UPDATE extensions SET active=0 WHERE id=?", (extension_id,))
                    if action == "delete":
                        db.execute("UPDATE extensions SET deleted=1 WHERE id=?", (extension_id,))
            else:
                raise BotError("Невідома операція з розширенням.")
            self.store.audit(user_id, "extension." + action, extension_id)
            return f"{extension_id}: {action}; застосовується з наступного повідомлення."

    async def create(
        self,
        request: str,
        kind: str,
        extension_id: str,
        parent: str,
        model: str,
        user_id: int,
        dependencies: list[str] | None = None,
    ) -> dict:
        self.settings.authorize(user_id)
        model = model or self.store.get("model:root", self.settings.default_model)
        base = Manifest(
            id=extension_id,
            kind=kind,  # type: ignore[arg-type]
            version=self.next_version(extension_id),
            description=request[:2000],
            parent=parent,
            model=model,
            entrypoint="invoke" if kind == "tool" else "build_agent",
            example_usage=request[:2000],
            dependencies=dependencies if dependencies is not None else [],
        )
        if (
            dependencies is not None
            and self.settings.allowed_dependencies
            and not set(base.dependencies) <= self.settings.allowed_dependencies
        ):
            raise BotError("Залежність відсутня в операторському ALLOWED_DEPENDENCIES.")
        template = (
            TOOL_TEMPLATE if kind == "tool" else AGENT_TEMPLATE.replace("EXTENSION_ID", base.id)
        )
        dependency_policy = (
            f"Allowed pinned packages: {sorted(self.settings.allowed_dependencies)}. "
            if self.settings.allowed_dependencies
            else "Any package is allowed; pin every dependency as package==version. "
        )
        explicit_dependencies = (
            f"Required dependencies are fixed by the caller: {base.dependencies}. "
            "Do not remove, replace or add packages; use these libraries in the source. "
            if dependencies is not None
            else "Declare every third-party import in manifest.dependencies with exact versions. "
        )
        prompt = (
            "Generate a Python extension. Return ONE JSON object with keys source and manifest. "
            "No Markdown. manifest must follow this base; you may change ONLY description, "
            "input_schema, output_schema, example, example_usage, dependencies, permissions. "
            "Tools: invoke(payload: dict) -> dict, docstring required; input_schema JSON Schema "
            "and example must be valid and safe to run during validation. "
            "Agents: build_agent(model: str, api_base: str) returns LlmAgent with LiteLlm "
            "ollama_chat/model. Only name, description, instruction, model constructor fields. "
            "No callbacks, sub_agents or tools in source; relationships are supplied by manifest. "
            "Network and access to host files outside the workspace are forbidden. "
            "All extensions have persistent workspace access by default, even with permissions=[]. "
            "The working directory is FILESYSTEM_ROOT: open('example.txt') reads a file there. "
            "os.environ['FILESYSTEM_ROOT'] provides its absolute path. Do not prefix relative "
            "file paths with workspace/; paths are relative to FILESYSTEM_ROOT itself. "
            "Use TMPDIR for temporary files. Validation runs the example: "
            "keep it safe and avoid modifying existing user files. "
            "No dependencies unless explicitly needed; "
            f"{dependency_policy}{explicit_dependencies}"
            f"Base manifest: {base.model_dump_json()}. Template:\n{template}\n"
            f"User request: {request}"
        )
        schema: dict = {
            "type": "object",
            "properties": {"source": {"type": "string"}, "manifest": Manifest.model_json_schema()},
            "required": ["source", "manifest"],
            "additionalProperties": False,
        }
        for field in ("id", "kind", "version", "parent", "model", "entrypoint"):
            schema["properties"]["manifest"]["properties"][field] = {"const": getattr(base, field)}
        if dependencies is not None:
            schema["properties"]["manifest"]["properties"]["dependencies"] = {
                "const": base.dependencies
            }
        try:
            generated = json.loads(await self.ollama.generate(model, prompt, schema))
            if dependencies is not None:
                # Caller-supplied dependencies are authoritative, including an explicit empty list.
                generated["manifest"]["dependencies"] = base.dependencies
            candidate = Manifest.model_validate(generated["manifest"])
            fixed = ("id", "kind", "version", "parent", "model", "entrypoint")
            if any(getattr(candidate, field) != getattr(base, field) for field in fixed):
                raise BotError("Генератор змінив обов’язкові поля маніфесту.")
            record = await self.install(candidate, generated["source"], user_id)
            return {
                "status": "active",
                "manifest": record.manifest.model_dump(),
                "source": generated["source"],
                "validation": "passed",
                "usage": candidate.example_usage,
            }
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise BotError(
                "Генератор повернув некоректний код або маніфест; нічого не активовано."
            ) from exc

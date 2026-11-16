import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from t_bot.tool_names import BASIC_TOOL_DESCRIPTIONS


def validate_dependency_list(values: list[str]) -> list[str]:
    """Accept at most ten pinned packages, without pip options, URLs or local paths."""
    if len(values) > 10:
        raise ValueError("At most 10 dependencies are allowed")
    for value in values:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*==[a-zA-Z0-9_.+!-]+", value):
            raise ValueError("Dependencies must use package==version without URLs or options")
    return sorted(set(values))


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    kind: Literal["tool", "agent", "subagent"]
    version: int = Field(ge=1)
    description: str = Field(min_length=1, max_length=2000)
    entrypoint: str = "invoke"
    model: str = "gemma4:12b"
    parent: str = "root"
    dependencies: list[str] = Field(default_factory=list, max_length=10)
    permissions: list[Literal["workspace"]] = Field(default=["workspace"])
    input_schema: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    )
    output_schema: dict = Field(default_factory=lambda: {"type": "object"})
    example: dict = Field(default_factory=dict)
    example_usage: str = Field(min_length=1, max_length=2000)

    @field_validator("input_schema", "output_schema")
    @classmethod
    def validate_schema_references(cls, value: dict) -> dict:
        if len(json.dumps(value)) > 16000:
            raise ValueError("Schema is too large")

        def check(node: object, depth: int = 0) -> None:
            if depth > 24:
                raise ValueError("Schema is too deeply nested")
            if isinstance(node, dict):
                for key, child in node.items():
                    if key in {"$ref", "$dynamicRef", "$recursiveRef", "$id"}:
                        raise ValueError(
                            "Schema references and external identifiers are unsupported"
                        )
                    check(child, depth + 1)
            elif isinstance(node, list):
                for child in node:
                    check(child, depth + 1)

        check(value)
        return value

    @field_validator("entrypoint", "parent")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", value):
            raise ValueError("Expected a plain Python identifier")
        return value

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, values: list[str]) -> list[str]:
        return validate_dependency_list(values)

    @model_validator(mode="after")
    def validate_contract(self) -> "Manifest":
        if self.id in BASIC_TOOL_DESCRIPTIONS or self.id in {
            "root",
            "restart_bot",
            "create_extension",
            "manage_extension",
            "list_resources",
            "set_model",
            "transfer_to_agent",
        }:
            raise ValueError("Reserved extension ID")
        if self.parent == self.id or (self.kind == "agent" and self.parent != "root"):
            raise ValueError("Invalid parent")
        if self.input_schema.get("type") != "object":
            raise ValueError("Tool arguments must be a JSON object")
        return self

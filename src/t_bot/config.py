import json
import sys
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PositiveInt = Annotated[int, Field(gt=0)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    telegram_bot_token: SecretStr = SecretStr("")
    allowed_user_ids: Annotated[set[int], NoDecode] = Field(default_factory=set)
    ollama_base_url: str = "http://127.0.0.1:11434"
    default_model: str = "gemma4:12b"
    bot_timezone: str = "Europe/Kyiv"
    data_dir: Path = Path("data")
    extensions_dir: Path = Path("data/extensions")
    filesystem_root: Path = Path("workspace")
    attachment_max_bytes: Annotated[int, Field(ge=1, le=50_000_000)] = 50_000_000
    filesystem_max_file_bytes: PositiveInt = 262144
    sqlite_database_dir: Path | None = None
    sqlite_query_timeout_seconds: PositiveInt = 5
    sqlite_max_rows: Annotated[int, Field(ge=1, le=1000)] = 100
    webpage_timeout_seconds: PositiveInt = 15
    webpage_max_bytes: Annotated[int, Field(ge=1024, le=5_242_880)] = 1_048_576
    webpage_max_chars: Annotated[int, Field(ge=256, le=48_000)] = 20_000
    web_search_timeout_seconds: PositiveInt = 15
    web_search_max_results: Annotated[int, Field(ge=1, le=10)] = 10
    console_timeout_seconds: PositiveInt = 30
    console_max_output_bytes: Annotated[int, Field(ge=256, le=65536)] = 6000
    screenshot_timeout_seconds: PositiveInt = 10
    screenshot_max_bytes: Annotated[int, Field(ge=1024, le=50_000_000)] = 20_000_000
    sandbox_memory_mb: PositiveInt = 512
    allowed_dependencies: Annotated[set[str], NoDecode] = Field(default_factory=set)
    max_parallel_tasks: PositiveInt = 2
    queue_size: PositiveInt = 32
    task_timeout_seconds: PositiveInt = 300
    model_timeout_seconds: PositiveInt = 120
    sandbox_timeout_seconds: PositiveInt = 30
    max_llm_calls: PositiveInt = 16
    extension_team_timeout_seconds: PositiveInt = 900
    extension_team_max_rounds: PositiveInt = 3
    extension_team_stage_attempts: PositiveInt = 2
    extension_team_max_calls: PositiveInt = 24
    extension_team_stagnation_limit: PositiveInt = 3
    extension_team_no_progress_limit: PositiveInt = 6
    extension_team_fallback_models: list[str] = Field(default_factory=list)
    max_delegation_depth: PositiveInt = 3
    max_context_chars: Annotated[int, Field(ge=4000)] = 48000
    max_input_chars: PositiveInt = 12000
    shutdown_timeout_seconds: PositiveInt = 20
    restart_cooldown_seconds: PositiveInt = 60

    @field_validator("ollama_base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise ValueError("OLLAMA_BASE_URL must be an HTTP URL without credentials")
        return value.rstrip("/")

    @field_validator("bot_timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                "BOT_TIMEZONE must be a valid IANA timezone, e.g. Europe/Kyiv"
            ) from exc
        return value

    @field_validator("allowed_user_ids", "allowed_dependencies", mode="before")
    @classmethod
    def parse_access_lists(cls, value: object) -> object:
        if isinstance(value, str):
            return json.loads(value) if value.strip() else []
        return value

    @field_validator("allowed_user_ids")
    @classmethod
    def validate_users(cls, value: set[int]) -> set[int]:
        if any(user_id <= 0 for user_id in value):
            raise ValueError("ALLOWED_USER_IDS must contain positive Telegram user IDs")
        return value

    @model_validator(mode="after")
    def validate_paths(self) -> "Settings":
        self.data_dir = self.data_dir.resolve()
        self.extensions_dir = self.extensions_dir.resolve()
        self.filesystem_root = self.filesystem_root.resolve()
        self.sqlite_database_dir = (
            self.sqlite_database_dir or self.data_dir / "databases"
        ).resolve()
        for runtime in (Path(sys.prefix), Path(sys.base_prefix)):
            if any(
                path.is_relative_to(runtime.resolve())
                for path in (self.data_dir, self.extensions_dir)
            ):
                raise ValueError("State and extensions must be outside the Python runtime")
        if self.extensions_dir == self.data_dir:
            raise ValueError("EXTENSIONS_DIR must be separate from DATA_DIR")
        return self

    def require_telegram(self) -> None:
        if not self.telegram_bot_token.get_secret_value():
            from t_bot.errors import BotError

            raise BotError("Заповніть TELEGRAM_BOT_TOKEN у .env.")

    def prepare_directories(self) -> None:
        for path in (self.data_dir, self.extensions_dir):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def allows(self, user_id: int) -> bool:
        return not self.allowed_user_ids or user_id in self.allowed_user_ids

    def authorize(self, user_id: int) -> None:
        from t_bot.errors import BotError

        if not self.allows(user_id):
            raise BotError("Доступ заборонено.")

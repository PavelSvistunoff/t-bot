import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = {
            "time": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key in (
            "job_id",
            "user_id",
            "chat_id",
            "session_id",
            "agent_id",
            "tool",
            "tool_call_id",
            "invocation_id",
            "error_type",
        ):
            if hasattr(record, key):
                event[key] = getattr(record, key)
        return json.dumps(event, ensure_ascii=False)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # HTTP/provider diagnostics may expose URLs containing the Telegram token or prompt bodies.
    for name in ("httpx", "httpcore", "telegram", "LiteLLM", "litellm", "google_adk", "ddgs"):
        external = logging.getLogger(name)
        external.handlers = [logging.NullHandler()]
        external.propagate = False

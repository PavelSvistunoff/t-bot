"""Read the system clock in an explicit IANA time zone, including daylight saving time."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo


def current_datetime(timezone: str) -> dict[str, str]:
    now = datetime.now(UTC).astimezone(ZoneInfo(timezone))
    iso8601 = now.isoformat(timespec="seconds")
    return {
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "timezone": timezone,
        "utc_offset": iso8601[-6:],
        "iso8601": iso8601,
    }


def format_datetime(value: dict[str, str]) -> str:
    date = datetime.fromisoformat(value["iso8601"]).strftime("%d.%m.%Y")
    return (
        f"Дата: {date}\nЧас: {value['time']}\n"
        f"Часовий пояс: {value['timezone']} (UTC{value['utc_offset']})"
    )

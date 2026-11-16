"""Weather API extension using standard-library HTTPS and explicit network permission."""

import json
from urllib.error import URLError
from urllib.request import urlopen

URL = "https://api.open-meteo.com/v1/forecast?latitude=49.44&longitude=32.61&current_weather=true"


def invoke(payload: dict) -> dict:
    """Fetch current Cherkasy weather; return an error instead of fabricated measurements."""
    try:
        with urlopen(URL, timeout=15) as response:
            current = json.load(response)["current_weather"]
        code = int(current["weathercode"])
        return {
            "temperature": float(current["temperature"]),
            "wind_speed": float(current["windspeed"]),
            "description": f"Код погоди WMO: {code}",
        }
    except (URLError, TimeoutError, OSError) as exc:
        return {"error": f"Не вдалося отримати погоду з Open-Meteo: {exc}"}
    except (KeyError, ValueError, TypeError):
        return {"error": "Open-Meteo повернув некоректні або неповні дані погоди."}

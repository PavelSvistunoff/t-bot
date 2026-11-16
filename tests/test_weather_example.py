import io
import json
import runpy
from urllib.error import URLError

import pytest


@pytest.mark.parametrize("failure", [False, True, "missing"])
def test_weather_uses_actual_windspeed_and_reports_errors(monkeypatch, failure):
    from unittest.mock import Mock

    if failure is True:
        request = Mock(side_effect=URLError("Failed to resolve host"))
    else:
        current = (
            {}
            if failure == "missing"
            else {
                "temperature": 12.5,
                "windspeed": 7.2,
                "weathercode": 3,
            }
        )
        request = Mock(return_value=io.StringIO(json.dumps({"current_weather": current})))
    monkeypatch.setattr("urllib.request.urlopen", request)
    module = runpy.run_path("examples/weather_cherkasy.py")
    result = module["invoke"]({})
    assert request.call_args.kwargs["timeout"] == 15
    if failure:
        assert "error" in result and "temperature" not in result and "wind_speed" not in result
    else:
        assert result == {
            "temperature": 12.5,
            "wind_speed": 7.2,
            "description": "Код погоди WMO: 3",
        }

import json
from unittest.mock import AsyncMock

import pytest

from t_bot.manifest import Manifest

pytestmark = pytest.mark.sandbox


@pytest.mark.parametrize("input_from", ["", "word_count"])
async def test_plot_consumes_word_count_result_without_reopening_file(
    components, settings, monkeypatch, input_from
):
    _, extensions, _, _ = components
    settings.filesystem_root.mkdir()
    original = settings.filesystem_root / "example.txt"
    original.write_text("aaa bb 123", encoding="utf-8")
    producer = Manifest(
        id="word_count",
        kind="tool",
        version=1,
        description="Count",
        example_usage="Count file",
        input_schema={
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "char_frequency": {"type": "object", "additionalProperties": {"type": "integer"}},
                "word_count": {"type": "integer"},
            },
        },
        example={"file_path": "example.txt"},
    )
    record = await extensions.install(
        producer,
        '''from collections import Counter
from pathlib import Path

def invoke(payload: dict) -> dict:
    """Count characters in a file."""
    text = Path(payload["file_path"]).read_text()
    return {"word_count": len(text.split()), "char_frequency": dict(Counter(text))}
''',
        1,
    )
    consumer = Manifest(
        id="plotter",
        kind="tool",
        version=1,
        description="Plot",
        example_usage="Plot counts",
        # Deliberately wrong generator contract: host must replace it with producer result input.
        input_schema={
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
        example={"file_path": "missing.txt"},
    )
    source = '''from pathlib import Path

def invoke(payload: dict) -> dict:
    """Draw bars from supplied letter counts, without opening the original input."""
    bars = [f'<rect width="{count * 10}" height="10"/>'
            for char, count in payload["char_frequency"].items() if char.isalpha()]
    Path("chart.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">' + ''.join(bars) + '</svg>')
    return {"path": "chart.svg", "letter_count": sum(
        n for char, n in payload["char_frequency"].items() if char.isalpha())}
'''
    generate = AsyncMock(
        return_value=json.dumps({"manifest": consumer.model_dump(), "source": source})
    )
    monkeypatch.setattr(extensions.ollama, "generate", generate)
    request = "Plot word_count output" if not input_from else "Draw a distribution chart"
    result = await extensions.create(request, "tool", "plotter", "root", "", 1, [], input_from)
    assert result["manifest"]["input_schema"]["required"] == ["char_frequency"]
    assert result["pipeline"] == {
        "producer": "word_count",
        "consumer": "plotter",
        "input_field": "char_frequency",
        "delivery_tool": "send_file",
    }
    assert "Producer output schema" in generate.call_args.args[1]
    counts = await extensions.invoke(record, {"file_path": "example.txt"}, 1)
    original.unlink()  # Consumer can no longer recount the source file.
    plotted = await extensions.invoke(extensions.get("plotter"), counts, 1)
    assert plotted == {"path": "chart.svg", "letter_count": 5}
    assert 'width="30"' in (settings.filesystem_root / "chart.svg").read_text()

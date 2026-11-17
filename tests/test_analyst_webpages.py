import json
from unittest.mock import AsyncMock

import pytest

from t_bot.errors import BotError
from t_bot.extension_team import SPEC_SCHEMA, TeamFailure

pytestmark = pytest.mark.team


@pytest.mark.parametrize("available", [True, False])
async def test_analyst_reads_unique_pages_before_revision(components, monkeypatch, available):
    _, extensions, _, _ = components
    team = extensions.team
    urls = [f"https://docs.python.org/{index}" for index in range(5)]
    monkeypatch.setattr(
        team.web,
        "search",
        lambda *args: {"results": [{"url": url, "snippet": "Unverified"} for url in urls]},
    )

    async def read(url):
        if url == urls[0] or not available:
            raise BotError("Page unavailable")
        return {"url": url, "text": "Verified documentation", "title": "Docs"}

    reader = AsyncMock(side_effect=read)
    monkeypatch.setattr(team.webpages, "read", reader)
    revisions = []

    async def ask(role, model, context, schema):
        if "queries" in schema["properties"]:
            return {"queries": ["Python", "Python docs"]}
        if role == "Analyst":
            revisions.append(json.loads(context))
            raise BotError("Stop after observing analyst input")
        return {
            name: "Objective" if name == "objective" else ["Acceptance"]
            for name in SPEC_SCHEMA["required"]
        }

    monkeypatch.setattr(team.roles, "ask", ask)
    validate = AsyncMock()
    with pytest.raises(TeamFailure) as failure:
        await team.run(
            request="Read documentation",
            extension_id="web_research",
            user_id=1,
            model="fake",
            developer_context="Implement",
            developer_schema={},
            validate=validate,
            execute_tests=AsyncMock(),
        )
    assert [call.args[0] for call in reader.await_args_list] == urls[:3]
    events = failure.value.report["events"]
    reads = [event for event in events if event["stage"] == "ReadWebpage"]
    assert len(reads) == 3
    if available:
        assert revisions
        assert [page["url"] for page in revisions[0]["web_pages"]] == urls[1:3]
        assert all(page["text"] == "Verified documentation" for page in revisions[0]["web_pages"])
    else:
        assert not revisions
        assert "read_webpage" in failure.value.report["error"]
    validate.assert_not_awaited()

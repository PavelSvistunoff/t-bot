import httpx
import pytest

from t_bot.errors import BotError
from t_bot.ollama import Ollama


async def test_missing_model_and_unavailable_service():
    client = Ollama("http://localhost:11434")
    await client.client.aclose()

    def transport(request):
        return httpx.Response(200, json={"models": [{"name": "available"}]})

    client.client = httpx.AsyncClient(
        base_url="http://localhost:11434", transport=httpx.MockTransport(transport)
    )
    try:
        with pytest.raises(BotError, match="відсутня"):
            await client.require_model("missing")
    finally:
        await client.close()

    async def offline(request):
        raise httpx.ConnectError("offline")

    client.client = httpx.AsyncClient(
        base_url="http://localhost:11434", transport=httpx.MockTransport(offline)
    )
    try:
        with pytest.raises(BotError, match="Ollama"):
            await client.models()
    finally:
        await client.close()

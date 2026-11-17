import asyncio

import httpx

from t_bot.errors import BotError


class Ollama:
    def __init__(self, base_url: str, timeout_seconds: int = 120):
        self.timeout_seconds = timeout_seconds
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds, trust_env=False)

    async def close(self) -> None:
        await self.client.aclose()

    async def request(self, method: str, path: str, **kwargs) -> dict:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError, TimeoutError) as exc:
            raise BotError(
                "Ollama не відповідає або відхилила запит. "
                "Перевірте сервіс, модель і доступну пам’ять."
            ) from exc

    async def models(self) -> list[str]:
        data = await self.request("GET", "/api/tags")
        return [model["name"] for model in data.get("models", [])]

    async def require_model(self, model: str) -> None:
        models = await self.models()
        if model not in models:
            raise BotError(f"Модель {model} відсутня. Виконайте ollama pull {model} або /models.")
        info = await self.request("POST", "/api/show", json={"model": model})
        if "completion" not in info.get("capabilities", ["completion"]):
            raise BotError("Ця модель призначена для embeddings, а не діалогу.")

    async def generate(
        self,
        model: str,
        instruction: str,
        schema: dict | None = None,
        *,
        max_output_tokens: int = 4096,
    ) -> str:
        await self.require_model(model)
        data = await self.request(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "stream": False,
                "think": False,
                "format": schema or "json",
                "messages": [{"role": "user", "content": instruction}],
                "options": {"temperature": 0.1, "num_ctx": 16384, "num_predict": max_output_tokens},
            },
        )
        content = data.get("message", {}).get("content", "")
        if not content.strip():
            raise BotError(
                "Модель не повернула код. Виберіть іншу модель або збільште ліміт генерації."
            )
        return content

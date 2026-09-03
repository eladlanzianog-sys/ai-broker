"""Claude API client wrapper.

Centralizes all LLM calls for consistent configuration, retries, and JSON parsing.
"""
import json

from anthropic import AsyncAnthropic

from src.config.settings import Settings

_client: AsyncAnthropic | None = None


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        settings = Settings()
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


async def call_claude_structured(
    system: str,
    user_content: str,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> dict:
    client = get_client()
    settings = Settings()

    json_instruction = (
        "\n\nYou MUST respond with valid JSON only. No markdown, no explanation "
        "outside the JSON object."
    )

    response = await client.messages.create(
        model=settings.claude_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system + json_instruction,
        messages=[{"role": "user", "content": user_content}],
    )

    text = response.content[0].text
    return json.loads(text)

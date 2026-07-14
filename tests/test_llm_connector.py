"""Tests for OpenAI-compatible LLM provider wiring."""

from __future__ import annotations

from pathlib import Path

from memory_agent.integrations.llm_connector import LLMConnector, PROVIDERS


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "qwen response"}}]}


class RecordingHttpClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def post(self, url: str, *, headers: dict, json: dict) -> FakeResponse:
        self.requests.append({"url": url, "headers": headers, "json": json})
        return FakeResponse()


class TestQwenCloudProvider:
    def test_provider_config_targets_dashscope_openai_compatible_mode(self):
        """qwencloud should route through DashScope's OpenAI-compatible endpoint."""
        provider = PROVIDERS["qwencloud"]

        assert provider["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode"
        assert not provider["base_url"].rstrip("/").endswith("/v1")
        assert provider["api_key_env"] == "DASHSCOPE_API_KEY"
        assert provider["default_model"].lower().startswith("qwen")

    def test_qwencloud_connector_posts_to_dashscope_chat_completions(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        """qwencloud should build a DashScope chat-completions request without network I/O."""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
        connector = LLMConnector(provider="qwencloud", db_path=tmp_path / "memory.db")
        http = RecordingHttpClient()
        original_http = connector.http
        connector.http = http

        try:
            response = connector._call_llm([{"role": "user", "content": "Hola"}])
        finally:
            original_http.close()
            connector.close()

        assert response == "qwen response"
        assert http.requests == [
            {
                "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                "headers": {
                    "Authorization": "Bearer dashscope-test-key",
                    "Content-Type": "application/json",
                },
                "json": {
                    "model": PROVIDERS["qwencloud"]["default_model"],
                    "messages": [{"role": "user", "content": "Hola"}],
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
            }
        ]

    def test_system_prompt_requires_english_responses(self):
        from memory_agent.integrations.llm_connector import SYSTEM_PROMPT

        assert "Respond in English" in SYSTEM_PROMPT
        assert "same language as the user" not in SYSTEM_PROMPT.lower()
        assert "[MEMORIES]" in SYSTEM_PROMPT

    def test_memory_context_uses_english_labels(self, tmp_path: Path):
        connector = LLMConnector(provider="qwencloud", db_path=tmp_path / "memory.db")
        original_http = connector.http
        try:
            connector.agent.store_memory(
                __import__("memory_agent.models", fromlist=["MemoryRecord"]).MemoryRecord(
                    content="The user prefers: python",
                    memory_type="preference",
                    importance=0.7,
                    confidence=0.9,
                )
            )
            context = connector._build_memory_context("python")
        finally:
            original_http.close()
            connector.close()

        assert "[MEMORIES]:" in context
        assert "[/MEMORIES]" in context
        assert "RECUERDOS" not in context

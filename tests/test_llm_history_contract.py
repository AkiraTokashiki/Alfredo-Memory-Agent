"""Contract tests for bounded LLM conversation history."""

from __future__ import annotations

from memory_agent.integrations.llm_connector import LLMConnector


class FakeAgent:
    """Minimal in-memory agent used to exercise ``LLMConnector.turn``."""

    def __init__(self) -> None:
        self.perceived: list[tuple[str, str]] = []

    def search_memories(self, user_input: str, *, top_k: int) -> dict:
        return {"results": []}

    def perceive(self, user_input: str, response: str) -> None:
        self.perceived.append((user_input, response))


def _connector(initial_messages: list[dict[str, str]]) -> LLMConnector:
    connector = LLMConnector.__new__(LLMConnector)
    connector.messages = list(initial_messages)
    connector.agent = FakeAgent()

    responses = iter(f"assistant-{turn}" for turn in range(1, 12))
    connector._call_llm = lambda messages: next(responses)
    return connector


def _run_eleven_turns(connector: LLMConnector) -> None:
    for turn in range(1, 12):
        assert connector.turn(f"user-{turn}") == f"assistant-{turn}"


def test_turn_without_system_prompt_keeps_last_twenty_conversation_messages() -> None:
    """Without a system message, trimming preserves only ten complete exchanges."""
    connector = _connector([])

    _run_eleven_turns(connector)

    assert connector.messages == [
        {"role": role, "content": f"{role}-{turn}"}
        for turn in range(2, 12)
        for role in ("user", "assistant")
    ]
    assert len(connector.messages) == 20
    assert connector.messages[0]["role"] == "user"
    assert all(
        connector.messages[index]["role"] != connector.messages[index + 1]["role"]
        for index in range(len(connector.messages) - 1)
    )


def test_turn_with_system_prompt_keeps_system_and_last_twenty_messages() -> None:
    """With a system message, trimming retains it before ten complete exchanges."""
    system = {"role": "system", "content": "You are concise."}
    connector = _connector([system])

    _run_eleven_turns(connector)

    assert connector.messages == [
        system,
        *[
            {"role": role, "content": f"{role}-{turn}"}
            for turn in range(2, 12)
            for role in ("user", "assistant")
        ],
    ]
    assert len(connector.messages) == 21
    assert connector.messages[0] == system
    assert connector.messages[1]["role"] == "user"
    assert all(
        connector.messages[index]["role"] != connector.messages[index + 1]["role"]
        for index in range(1, len(connector.messages) - 1)
    )

"""Memory extraction and decision making from user interactions."""

from __future__ import annotations

import re
from typing import Any

from memory_agent.models import MemoryRecord


# Patterns for extracting structured information from natural input
_PREFERENCE_PATTERNS: list[tuple[str, str]] = [
    # Spanish: "me gusta/encanta/prefiero X" (includes "no me gusta" as negative)
    (r"(?:(?:no\s+)?me\s+)?(?:gusta|encanta|prefiero|prefieres?|disgusta|odia?)\s+(?:el|la|los|las|que|cuando|mas|mucho(?:\s+el|\s+la)?)?\s*(.+)", "preference"),
    # Spanish: "mi [X] favorito/favorita/preferido es [Y]"
    (r"(?:mi\s+)?(?:\w+\s+)?(?:favorito|favorita|preferido|preferida)\s+(?:es|son)\s+(.+)", "preference"),
    # English: "I like/love/prefer/enjoy X"
    (r"(?:i\s+)?(?:like|love|prefer|enjoy|favorite)\s+(.+)", "preference"),
    (r"(?:i\s+)?(?:don't\s+|do\s+not\s+)?(?:like|hate|dislike)\s+(.+)", "preference"),
    # "I usually/always X"
    (r"(?:siempre|normalmente|usualmente|generalmente)\s+(.+)", "habit"),
    (r"(?:usually|always|typically|normally)\s+(.+)", "habit"),
]

_FORGET_PATTERNS: list[str] = [
    r"(?:forget|remove|delete)\s+(?:that\s+)?(.+)",
    r"(?:olvida|borra|elimina)\s+(?:que\s+)?(.+)",
]


def _topic_from_content(content: str) -> str:
    topic = content.lower().strip()
    topic = re.sub(r"^(el usuario prefiere:|al usuario no le gusta:|hecho:)\s*", "", topic)
    topic = re.sub(r"[^a-z0-9áéíóúñü\s]+", "", topic)
    return " ".join(topic.split())


def extract_forget_query(text: str) -> str | None:
    """Extract the target of an explicit forget request."""
    text_lower = text.lower().strip()
    for pattern in _FORGET_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            query = match.group(1).strip().rstrip(".,!?")
            return query or None
    return None


def extract_preferences(text: str) -> list[tuple[str, str, float]]:
    """Extract (content, memory_type, importance) from user text.

    Returns a list of candidate memories that could be extracted.
    """
    results: list[tuple[str, str, float]] = []
    text_lower = text.lower()

    for pattern, mem_type in _PREFERENCE_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            content = match.strip().rstrip(".,!?")
            if not content or len(content) < 5:
                continue

            # Detect negative preferences ("no me gusta X")
            is_negative = bool(re.search(r"no\s+me\s+(gusta|encanta|prefiero)", text_lower))

            # Higher importance for strong preferences
            importance = 0.7
            if any(word in text_lower for word in ("amo", "encanta", "love", "absolutely", "me encanta")):
                importance = 0.9
            if is_negative or any(word in text_lower for word in ("odio", "hate", "disgusta")):
                importance = 0.6

            # Build a natural-sounding memory
            if is_negative:
                memory_text = f"Al usuario no le gusta: {content}"
            else:
                memory_text = f"El usuario prefiere: {content}"
            if mem_type == "habit" and not is_negative:
                memory_text = f"El usuario usualmente: {content}"

            results.append((memory_text, mem_type, importance))

    return results


_FACT_PATTERNS: list[tuple[str, str, float]] = [
    (r"(?:yo\s+)?(?:trabajo|estudio|vivo|uso|tengo)\s+(?:en|con|como)\s+(.+)", "semantic", 0.6),
    (r"(?:i\s+)?(?:work|study|live|use|have)\s+(?:at|with|as|in)\s+(.+)", "semantic", 0.6),
    (r"(?:mi\s+)?(?:nombre es|email es|telefono es|cumpleanos es)\s+(.+)", "semantic", 0.8),
    (r"(?:my\s+)?(?:name is|email is|phone is|birthday is)\s+(.+)", "semantic", 0.8),
]


def extract_facts(text: str) -> list[tuple[str, str, float]]:
    """Extract factual information (work, study, contact details)."""
    results: list[tuple[str, str, float]] = []
    text_lower = text.lower()
    for pattern, mem_type, importance in _FACT_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            content = match.strip().rstrip(".,!?")
            if not content or len(content) < 5:
                continue
            results.append((f"Hecho: {content}", mem_type, importance))
    return results


def extract_from_input(text: str) -> list[MemoryRecord]:
    """Analyze user input and extract candidate memories.

    Returns a list of MemoryRecord objects ready to be stored.
    """
    memories: list[MemoryRecord] = []

    for content, mem_type, importance in extract_preferences(text):
        polarity = "negative" if "no le gusta" in content.lower() else "positive"
        memories.append(MemoryRecord(
            content=content,
            memory_type=mem_type,
            importance=importance,
            metadata={
                "topic": _topic_from_content(content),
                "polarity": polarity,
            },
            tags=["extracted", "preference"],
        ))

    for content, mem_type, importance in extract_facts(text):
        memories.append(MemoryRecord(
            content=content,
            memory_type=mem_type,
            importance=importance,
            metadata={"topic": _topic_from_content(content)},
            tags=["extracted", "fact"],
        ))

    return memories


def should_remember(interaction: str, response: str) -> bool:
    """Heuristic: should we store this interaction as a memory?

    Returns True if the interaction seems meaningful enough.
    """
    # Always store if we extracted anything
    if extract_from_input(interaction):
        return True

    # Store long/complex interactions
    if len(interaction.split()) > 20:
        return True

    # Store questions about the agent itself
    question_words = ("quien", "que", "como", "donde", "cuando", "who", "what", "how", "where")
    if interaction.lower().startswith(question_words):
        return True

    return False


def summarize_interaction(user_input: str, agent_response: str) -> str:
    """Create a concise episodic memory from an interaction."""
    # Simple truncation-based summary
    input_preview = user_input[:100].strip()
    response_preview = agent_response[:80].strip()
    return f"Usuario: {input_preview} | Respuesta: {response_preview}"

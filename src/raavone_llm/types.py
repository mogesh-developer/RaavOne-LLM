from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMMessage:
    role: str
    content: str

    def __post_init__(self):
        allowed_roles = {
            "system",
            "user",
            "assistant",
            "tool",
        }

        if self.role not in allowed_roles:
            raise ValueError(
                f"Invalid message role: {self.role}"
            )

        if not self.content.strip():
            raise ValueError(
                "Message content cannot be empty."
            )


@dataclass
class GenerationConfig:
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    def __post_init__(self):
        if not 0 <= self.temperature <= 2:
            raise ValueError(
                "Temperature must be between 0 and 2."
            )

        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than 0."
            )


@dataclass
class GenerationRequest:
    messages: list[LLMMessage]
    config: Optional[GenerationConfig] = None

    def __post_init__(self):
        if not self.messages:
            raise ValueError(
                "GenerationRequest requires at least one message."
            )

@dataclass
class GenerationUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GenerationResponse:
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[GenerationUsage] = None


@dataclass
class GenerationChunk:
    content: str
    model: str
    finish_reason: Optional[str] = None
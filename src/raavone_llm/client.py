from collections.abc import Iterator

from .interface import LLMProvider
from .types import (
    GenerationChunk,
    GenerationRequest,
    GenerationResponse,
    LLMMessage,
)
from .generation.service import GenerationService


class RaavOneLLM:

    def __init__(self, provider: LLMProvider):
        self.service = GenerationService(provider)

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return self.service.generate(request)

    def stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationChunk]:
        return self.service.stream(request)

    def chat(
    self,
    message: str,
    ) -> GenerationResponse:

        request = GenerationRequest(
        messages=[
            LLMMessage(
                role="user",
                content=message,
            )
        ]
    )

        return self.generate(request)
from collections.abc import AsyncIterator, Iterator
import asyncio

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
   

    async def generate_async(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:

        return await self.service.generate_async(request)

    async def stream_async(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:

        return await self.service.stream_async(request)
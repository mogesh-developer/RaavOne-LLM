from collections.abc import AsyncIterator, Iterator

from ..interface import LLMProvider
from ..types import (
    GenerationChunk,
    GenerationRequest,
    GenerationResponse,
)


class GenerationService:

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return self.provider.generate(request)

    def stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationChunk]:
        return self.provider.stream(request)

    async def generate_async(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return await self.provider.generate_async(request)

    async def stream_async(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        return self.provider.stream_async(request)
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator

from .capabilities import ProviderCapabilities
from .types import (
    GenerationChunk,
    GenerationRequest,
    GenerationResponse,
)


class LLMProvider(ABC):

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        pass

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        pass

    @abstractmethod
    def stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationChunk]:
        pass

    @abstractmethod
    async def generate_async(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        pass

    @abstractmethod
    async def stream_async(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        pass
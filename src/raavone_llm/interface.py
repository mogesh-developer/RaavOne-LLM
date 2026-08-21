from abc import ABC, abstractmethod
from collections.abc import Iterator

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
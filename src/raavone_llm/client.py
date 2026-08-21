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

        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_running_loop()

        def producer():
            try:
                for chunk in self.stream(request):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        task = loop.run_in_executor(None, producer)

        try:
            while True:
                item = await queue.get()

                if item is sentinel:
                    break

                if isinstance(item, Exception):
                    raise item

                yield item

        finally:
            # Task is a Future representing the producer running in the executor
            if not task.done():
                task.cancel()
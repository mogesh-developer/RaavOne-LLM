import os
from typing import Optional
from google import genai
from google.genai import types as genai_types

from collections.abc import AsyncIterator, Iterator
import asyncio

from ..interface import LLMProvider
from ..capabilities import ProviderCapabilities
from ..types import GenerationChunk, GenerationRequest, GenerationResponse
from ..exceptions import AuthenticationError, ProviderAPIError
from ..config import GeminiConfig


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        config = GeminiConfig(api_key=api_key)
        self.api_key = config.get_api_key()
        if not self.api_key:
            raise AuthenticationError(
                "Gemini API key is missing. Please set the GEMINI_API_KEY environment variable "
                "or pass it explicitly."
            )
        self.model = model or config.default_model
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize Gemini Client: {e}") from e

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        model = request.config.model if request.config else self.model
        temperature = request.config.temperature if request.config else 0.7
        max_tokens = request.config.max_tokens if request.config else None

        # Convert LLMMessage list to google-genai Content structure
        contents = []
        for msg in request.messages:
            contents.append(
                genai_types.Content(
                    role=msg.role,
                    parts=[genai_types.Part.from_text(text=msg.content)]
                )
            )

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            # Response might not contain text due to safety settings or other errors
            if response.text is None:
                raise ProviderAPIError("Empty or invalid response content from Gemini API (possibly safety blocked).")
            return GenerationResponse(
                content=response.text,
                model=model
            )
        except Exception as e:
            raise ProviderAPIError(f"Gemini API generation error: {e}") from e

    def stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationChunk]:
        model = request.config.model if request.config else self.model
        temperature = request.config.temperature if request.config else 0.7
        max_tokens = request.config.max_tokens if request.config else None

        # Convert LLMMessage list to google-genai Content structure
        contents = []
        for msg in request.messages:
            contents.append(
                genai_types.Content(
                    role=msg.role,
                    parts=[genai_types.Part.from_text(text=msg.content)]
                )
            )

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        try:
            response_stream = self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            for chunk in response_stream:
                if chunk.text is not None:
                    yield GenerationChunk(
                        content=chunk.text,
                        model=model
                    )
        except Exception as e:
            raise ProviderAPIError(f"Gemini API streaming error: {e}") from e

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
        )   

    async def generate_async(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return await asyncio.to_thread(self.generate, request)

    async def stream_async(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def producer():
            try:
                for chunk in self.stream(request):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, producer)

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
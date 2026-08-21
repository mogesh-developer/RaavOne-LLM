import os

from groq import Groq

from collections.abc import Iterator

from ..exceptions import AuthenticationError, ProviderAPIError
from ..interface import LLMProvider
from ..types import GenerationChunk, GenerationRequest, GenerationResponse, GenerationUsage
from ..capabilities import ProviderCapabilities

class GroqProvider(LLMProvider):

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
    ):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise AuthenticationError(
                "GROQ_API_KEY environment variable is not set."
            )

        # Use environment override if set, otherwise default
        if model == "llama-3.3-70b-versatile":
            self.model = os.getenv("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile")
        else:
            self.model = model
        self.client = Groq(api_key=api_key)

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        config = request.config
        model = config.model if config and config.model else self.model
        temperature = (
            config.temperature
            if config
            else 0.7
        )
        max_tokens = (
            config.max_tokens
            if config
            else None
        )

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                    for message in request.messages
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            usage = None
            if response.usage:
                usage = GenerationUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            return GenerationResponse(
                content=response.choices[0].message.content,
                model=model,
                finish_reason=response.choices[0].finish_reason,
                usage=usage,
            )

        except Exception as exc:
            raise ProviderAPIError(
                f"Groq generation failed: {exc}"
            ) from exc


    def stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationChunk]:
        config = request.config
        model = config.model if config and config.model else self.model
        temperature = (
            config.temperature
            if config
            else 0.7
        )
        max_tokens = (
            config.max_tokens
            if config
            else None
        )

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                    for message in request.messages
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in response:
                if (
                    chunk.choices
                    and chunk.choices[0].delta.content is not None
                ):
                    yield GenerationChunk(
                        content=chunk.choices[0].delta.content,
                        model=model,
                    )

        except Exception as exc:
            raise ProviderAPIError(
                f"Groq streaming failed: {exc}"
            ) from exc

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
        )   
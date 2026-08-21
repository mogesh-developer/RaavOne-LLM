from collections.abc import Iterator
from pathlib import Path

from llama_cpp import Llama

from ..exceptions import ConfigurationError, ProviderAPIError
from ..interface import LLMProvider
from ..capabilities import ProviderCapabilities
from ..types import (
    GenerationChunk,
    GenerationRequest,
    GenerationResponse,
)


class LocalProvider(LLMProvider):

    def __init__(
        self,
        model_path: str,
        context_size: int = 4096,
        threads: int = 4,
        gpu_layers: int = 0,
    ):
        path = Path(model_path)

        if not path.exists():
            raise ConfigurationError(
                f"Local model not found: {path}"
            )

        if path.suffix.lower() != ".gguf":
            raise ConfigurationError(
                "LocalProvider requires a GGUF model."
            )

        self.model_path = path
        self.context_size = context_size
        self.threads = threads
        self.gpu_layers = gpu_layers

        try:
            self.model = Llama(
                model_path=str(path),
                n_ctx=context_size,
                n_threads=threads,
                n_gpu_layers=gpu_layers,
                verbose=False,
            )

        except Exception as exc:
            raise ConfigurationError(
                f"Failed to load local model: {exc}"
            ) from exc

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:

        try:
            config = request.config

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

            response = self.model.create_chat_completion(
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

            content = response["choices"][0]["message"]["content"]

            return GenerationResponse(
                content=content,
                model=self.model_path.name,
                finish_reason=response["choices"][0].get(
                    "finish_reason"
                ),
            )

        except Exception as exc:
            raise ProviderAPIError(
                f"Local model generation failed: {exc}"
            ) from exc

    def stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationChunk]:

        try:
            config = request.config

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

            response = self.model.create_chat_completion(
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
                content = chunk["choices"][0]["delta"].get(
                    "content"
                )

                if content:
                    yield GenerationChunk(
                        content=content,
                        model=self.model_path.name,
                    )

        except Exception as exc:
            raise ProviderAPIError(
                f"Local model streaming failed: {exc}"
            ) from exc
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
        )
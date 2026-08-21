from dotenv import load_dotenv

# Load environment variables from .env file automatically when library is imported
load_dotenv(override=True)

from .client import RaavOneLLM
from .interface import LLMProvider
from .providers.gemini import GeminiProvider
from .providers.groq import GroqProvider
from .providers.local import LocalProvider
from .types import LLMMessage, GenerationRequest, GenerationResponse, GenerationChunk, GenerationUsage, GenerationConfig
from .config import GeminiConfig
from .exceptions import LLMError, AuthenticationError, ProviderAPIError, ConfigurationError
from .capabilities import ProviderCapabilities
from .providers import (
    GeminiProvider,
    GroqProvider,
    LocalProvider,
)


__all__ = [
    "RaavOneLLM",
    "LLMProvider",
    "GeminiProvider",
    "GroqProvider",
    "LocalProvider",
    "LLMMessage",
    "GenerationRequest",
    "GenerationResponse",
    "GenerationChunk",
    "GenerationConfig",
    "GeminiConfig",
    "LLMError",
    "AuthenticationError",
    "ProviderAPIError",
    "ConfigurationError",
    "GenerationUsage",
    "ProviderCapabilities",
]

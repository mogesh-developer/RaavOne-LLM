import os
from unittest.mock import MagicMock, patch

from raavone_llm import (
    GeminiProvider,
    GroqProvider,
    LLMProvider,
    LocalProvider,
    ProviderCapabilities,
)


@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
@patch("google.genai.Client")
def test_gemini_provider_contract(mock_client):
    provider = GeminiProvider()
    assert isinstance(provider, LLMProvider)
    assert hasattr(provider, "capabilities")
    assert isinstance(provider.capabilities, ProviderCapabilities)
    assert hasattr(provider, "generate")
    assert hasattr(provider, "stream")
    assert hasattr(provider, "generate_async")
    assert hasattr(provider, "stream_async")


@patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
@patch("groq.Groq")
def test_groq_provider_contract(mock_client):
    provider = GroqProvider()
    assert isinstance(provider, LLMProvider)
    assert hasattr(provider, "capabilities")
    assert isinstance(provider.capabilities, ProviderCapabilities)
    assert hasattr(provider, "generate")
    assert hasattr(provider, "stream")
    assert hasattr(provider, "generate_async")
    assert hasattr(provider, "stream_async")


@patch("pathlib.Path.exists", return_value=True)
@patch("raavone_llm.providers.local.Llama")
def test_local_provider_contract(mock_llama, mock_exists):
    provider = LocalProvider(model_path="dummy.gguf")
    assert isinstance(provider, LLMProvider)
    assert hasattr(provider, "capabilities")
    assert isinstance(provider.capabilities, ProviderCapabilities)
    assert hasattr(provider, "generate")
    assert hasattr(provider, "stream")
    assert hasattr(provider, "generate_async")
    assert hasattr(provider, "stream_async")

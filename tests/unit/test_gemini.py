import unittest
from unittest.mock import MagicMock, patch
import os

from raavone_llm import (
    RaavOneLLM,
    GeminiProvider,
    LLMMessage,
    GenerationRequest,
    GenerationResponse,
    AuthenticationError,
    ProviderAPIError,
)


class TestGeminiProvider(unittest.TestCase):
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"})
    @patch("google.genai.Client")
    def test_provider_initialization(self, mock_client_class):
        provider = GeminiProvider()
        self.assertEqual(provider.model, "gemini-2.5-flash")
        self.assertEqual(provider.api_key, "test-key-123")
        mock_client_class.assert_called_once_with(api_key="test-key-123")

    @patch.dict(os.environ, {}, clear=True)
    def test_provider_missing_key_raises_error(self):
        with self.assertRaises(AuthenticationError):
            GeminiProvider()

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"})
    @patch("google.genai.Client")
    def test_generation_success(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock response from client.models.generate_content
        mock_response = MagicMock()
        mock_response.text = "Hello, this is Gemini."
        mock_client.models.generate_content.return_value = mock_response

        provider = GeminiProvider()
        client = RaavOneLLM(provider)

        request = GenerationRequest(
            messages=[LLMMessage(role="user", content="Hello")]
        )
        response = client.generate(request)

        self.assertIsInstance(response, GenerationResponse)
        self.assertEqual(response.content, "Hello, this is Gemini.")
        self.assertEqual(response.model, "gemini-2.5-flash")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"})
    @patch("google.genai.Client")
    def test_generation_failure(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("API error")

        provider = GeminiProvider()
        client = RaavOneLLM(provider)

        request = GenerationRequest(
            messages=[LLMMessage(role="user", content="Hello")]
        )
        with self.assertRaises(ProviderAPIError):
            client.generate(request)


if __name__ == "__main__":
    unittest.main()

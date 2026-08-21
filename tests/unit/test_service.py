from unittest.mock import MagicMock
import pytest

from raavone_llm import GenerationRequest, LLMMessage, LLMProvider, RaavOneLLM


def test_service_generate():
    mock_provider = MagicMock(spec=LLMProvider)
    llm = RaavOneLLM(mock_provider)
    request = GenerationRequest(messages=[LLMMessage(role="user", content="Hi")])
    
    llm.generate(request)
    mock_provider.generate.assert_called_once_with(request)


def test_service_stream():
    mock_provider = MagicMock(spec=LLMProvider)
    llm = RaavOneLLM(mock_provider)
    request = GenerationRequest(messages=[LLMMessage(role="user", content="Hi")])
    
    llm.stream(request)
    mock_provider.stream.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_service_generate_async():
    mock_provider = MagicMock(spec=LLMProvider)
    
    async def mock_gen_async(req):
        return MagicMock()
    
    mock_provider.generate_async = mock_gen_async

    llm = RaavOneLLM(mock_provider)
    request = GenerationRequest(messages=[LLMMessage(role="user", content="Hi")])
    
    await llm.generate_async(request)


@pytest.mark.asyncio
async def test_service_stream_async():
    mock_provider = MagicMock(spec=LLMProvider)
    
    async def mock_str_async(req):
        yield MagicMock()
        
    mock_provider.stream_async = mock_str_async

    llm = RaavOneLLM(mock_provider)
    request = GenerationRequest(messages=[LLMMessage(role="user", content="Hi")])
    
    chunks = []
    async for chunk in llm.stream_async(request):
        chunks.append(chunk)
    assert len(chunks) == 1

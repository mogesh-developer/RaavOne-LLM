from pathlib import Path
import pytest

from raavone_llm import (
    GenerationConfig,
    GenerationRequest,
    LLMMessage,
    LocalProvider,
    RaavOneLLM,
)

MODEL_PATH = "models/Llama/Llama-3.2-3B-Instruct-uncensored.Q4_K_S.gguf"


@pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason="Local GGUF model file not found. Skipping integration test."
)
def test_local_provider_generation_integration():
    provider = LocalProvider(
        model_path=MODEL_PATH,
        context_size=1024,
    )
    llm = RaavOneLLM(provider)
    request = GenerationRequest(
        messages=[LLMMessage(role="user", content="Hello.")],
        config=GenerationConfig(max_tokens=5)
    )
    response = llm.generate(request)
    assert response.content is not None
    assert response.model == Path(MODEL_PATH).name


@pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason="Local GGUF model file not found. Skipping integration test."
)
@pytest.mark.asyncio
async def test_local_provider_async_generation_integration():
    provider = LocalProvider(
        model_path=MODEL_PATH,
        context_size=1024,
    )
    llm = RaavOneLLM(provider)
    request = GenerationRequest(
        messages=[LLMMessage(role="user", content="Hello.")],
        config=GenerationConfig(max_tokens=5)
    )
    response = await llm.generate_async(request)
    assert response.content is not None
    assert response.model == Path(MODEL_PATH).name

from raavone_llm import (
    LLMMessage,
    GenerationConfig,
    GenerationRequest,
    GenerationResponse,
    GenerationChunk,
    GenerationUsage,
)


def test_llm_message():
    msg = LLMMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_generation_config():
    config = GenerationConfig(model="test-model", temperature=0.5, max_tokens=100)
    assert config.model == "test-model"
    assert config.temperature == 0.5
    assert config.max_tokens == 100


def test_generation_request():
    msg = LLMMessage(role="user", content="Hello")
    config = GenerationConfig(model="test-model")
    req = GenerationRequest(messages=[msg], config=config)
    assert req.messages == [msg]
    assert req.config == config


def test_generation_response():
    usage = GenerationUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    resp = GenerationResponse(content="Hi", model="test-model", finish_reason="stop", usage=usage)
    assert resp.content == "Hi"
    assert resp.model == "test-model"
    assert resp.finish_reason == "stop"
    assert resp.usage == usage


def test_generation_chunk():
    chunk = GenerationChunk(content="H", model="test-model", finish_reason="stop")
    assert chunk.content == "H"
    assert chunk.model == "test-model"
    assert chunk.finish_reason == "stop"

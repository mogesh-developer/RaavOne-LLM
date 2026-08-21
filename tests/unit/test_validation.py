import pytest
from raavone_llm import LLMMessage, GenerationConfig, GenerationRequest


def test_llm_message_valid():
    msg = LLMMessage(role="user", content="Hello")
    assert msg.role == "user"


def test_llm_message_invalid_role():
    with pytest.raises(ValueError, match="Invalid message role"):
        LLMMessage(role="invalid_role", content="Hello")


def test_llm_message_empty_content():
    with pytest.raises(ValueError, match="Message content cannot be empty"):
        LLMMessage(role="user", content="   ")


def test_generation_config_valid():
    config = GenerationConfig(model="model", temperature=1.5, max_tokens=100)
    assert config.temperature == 1.5


def test_generation_config_invalid_temp():
    with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
        GenerationConfig(model="model", temperature=2.5)
    with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
        GenerationConfig(model="model", temperature=-0.5)


def test_generation_config_invalid_max_tokens():
    with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
        GenerationConfig(model="model", max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
        GenerationConfig(model="model", max_tokens=-10)


def test_generation_request_empty_messages():
    with pytest.raises(ValueError, match="GenerationRequest requires at least one message"):
        GenerationRequest(messages=[])

from raavone_llm.providers.mock import MockProvider
from raavone_llm.types import GenerationRequest, LLMMessage


provider = MockProvider()

request = GenerationRequest(
    messages=[
        LLMMessage(
            role="user",
            content="Hello RaavOne",
        )
    ]
)

response = provider.generate(request)

print(response)
print(response.content)
print(response.model)
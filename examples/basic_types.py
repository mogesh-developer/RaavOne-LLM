from raavone_llm.types import (
    LLMMessage,
    GenerationRequest,
)


request = GenerationRequest(
    messages=[
        LLMMessage(
            role="user",
            content="Hello RaavOne"
        )
    ]
)

print(request)
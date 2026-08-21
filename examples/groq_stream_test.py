from raavone_llm import (
    GenerationRequest,
    LLMMessage,
    GroqProvider,
    RaavOneLLM,
)


llm = RaavOneLLM(
    GroqProvider()
)

request = GenerationRequest(
    messages=[
        LLMMessage(
            role="user",
            content="Explain RaavOne LLM in 100 words.",
        )
    ]
)

for chunk in llm.stream(request):
    print(chunk.content, end="", flush=True)

print()
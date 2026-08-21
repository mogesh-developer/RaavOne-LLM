from raavone_llm import (
    GenerationConfig,
    GenerationRequest,
    LLMMessage,
    LocalProvider,
    RaavOneLLM,
)


provider = LocalProvider(
    model_path="models/Llama/Llama-3.2-3B-Instruct-uncensored.Q4_K_S.gguf",
    context_size=4096,
    threads=8,
    gpu_layers=0,
)

llm = RaavOneLLM(provider)

request = GenerationRequest(
    messages=[
        LLMMessage(
            role="user",
            content="Explain what a local LLM is in simple terms.",
        )
    ],
    config=GenerationConfig(
        temperature=0.7,
        max_tokens=200,
    ),
)

for chunk in llm.stream(request):
    print(chunk.content, end="", flush=True)

print()


provider = LocalProvider(
    model_path="models/Llama/Llama-3.2-3B-Instruct-uncensored.Q4_K_S.gguf"
)

print(provider.capabilities)
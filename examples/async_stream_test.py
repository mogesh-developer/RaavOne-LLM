import asyncio

from raavone_llm import (
    GenerationConfig,
    GenerationRequest,
    LLMMessage,
    LocalProvider,
    RaavOneLLM,
)


async def main():

    provider = LocalProvider(
        model_path=(
            "models/Llama/"
            "Llama-3.2-3B-Instruct-uncensored.Q4_K_S.gguf"
        ),
        context_size=4096,
        threads=8,
        gpu_layers=0,
    )

    llm = RaavOneLLM(provider)

    request = GenerationRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Explain RaavOne LLM in simple terms.",
            )
        ],
        config=GenerationConfig(
            temperature=0.7,
            max_tokens=200,
        ),
    )

    async for chunk in llm.stream_async(request):
        print(chunk.content, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())
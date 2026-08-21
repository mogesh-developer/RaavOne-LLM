from raavone_llm import (
    RaavOneLLM,
    GroqProvider,
    GenerationRequest,
    GenerationConfig,
    LLMMessage,
)


provider = GroqProvider(model="qwen/qwen3.6-27b")

llm = RaavOneLLM(provider)

# Test 1: No request config (should use provider default: qwen/qwen3.6-27b)
print("--- Running Test 1 (Using Provider Default Model) ---")
request_default = GenerationRequest(
    messages=[
        LLMMessage(
            role="user",
            content="Explain what RaavOne LLM is in one sentence.",
        )
    ]
)

try:
    response = llm.generate(request_default)
    print("Content:", response.content)
    print("Model used:", response.model)
    print("Finish Reason:", response.finish_reason)
    print("Total Tokens:", response.usage.total_tokens if response.usage else None)
except Exception as e:
    print(f"Error occurred: {e}")

# Test 2: Request config override (should override provider default to groq/compound-mini)
print("\n--- Running Test 2 (Using Request Config Override: groq/compound-mini) ---")
request_override = GenerationRequest(
    messages=[
        LLMMessage(
            role="user",
            content="Say hello in one word.",
        )
    ],
    config=GenerationConfig(
        model="groq/compound-mini",
        temperature=0.2,
        max_tokens=100,
    )
)

try:
    response = llm.generate(request_override)
    print("Content:", response.content)
    print("Model used:", response.model)
    print("Finish Reason:", response.finish_reason)
    print("Total Tokens:", response.usage.total_tokens if response.usage else None)
except Exception as e:
    print(f"Error occurred: {e}")
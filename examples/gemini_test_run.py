import os
from raavone_llm import RaavOneLLM, GeminiProvider, LLMMessage, GenerationRequest


def main():
    # Attempt to initialize with a dummy key or from environment
    api_key = os.environ.get("GEMINI_API_KEY", "dummy-key-for-import-check")
    print("Initializing GeminiProvider with model: gemini-2.5-flash...")
    try:
        provider = GeminiProvider(api_key=api_key)
        client = RaavOneLLM(provider)
        print("Success: Client and provider initialized successfully!")
        
        req = GenerationRequest(
            messages=[LLMMessage(role="user", content="Hello, Gemini!")]
        )
        print(f"Request structure: {req}")
    except Exception as e:
        print(f"Initialization/usage demo encountered error: {e}")


if __name__ == "__main__":
    main()

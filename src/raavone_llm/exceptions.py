class LLMError(Exception):
    """Base exception for all raavone_llm errors."""
    pass


class ProviderAPIError(LLMError):
    """Exception raised for errors during API calls to providers."""
    pass


class AuthenticationError(LLMError):
    """Exception raised when API key or authorization fails."""
    pass


class ConfigurationError(LLMError):
    """Exception raised for configuration issues."""
    pass

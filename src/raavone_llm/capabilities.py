from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    streaming: bool = False
    tools: bool = False
    structured_output: bool = False
    vision: bool = False

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GeminiConfig:
    api_key: Optional[str] = None
    default_model: str = "gemini-2.5-flash"

    def get_api_key(self) -> Optional[str]:
        return self.api_key or os.environ.get("GEMINI_API_KEY")

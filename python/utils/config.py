"""Configuration management for OpenRouter API."""

from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration for OpenRouter API access."""

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    BASE_URL = "https://openrouter.ai/api/v1"
    MODEL_FREE = "minimax/minimax-m2:free"

    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
                "Please set it in your .env file or environment variables."
            )
        return True

    @classmethod
    def get_client(cls) -> OpenAI:
        """Get configured OpenAI client for OpenRouter."""
        cls.validate()
        return OpenAI(
            base_url=cls.BASE_URL,
            api_key=cls.OPENROUTER_API_KEY
        )

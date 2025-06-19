import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Configuration settings for the application."""
    
    def __init__(self):
        # OpenAI Configuration
        self.openai_api_key: str = self._get_required_env("OPENAI_API_KEY")
        self.openai_model: str = self._get_env("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.temperature: float = float(self._get_env("TEMPERATURE", "0.7"))
        self.max_tokens: int = int(self._get_env("MAX_TOKENS", "4000"))
        
        # Optional: Add other configuration variables as needed
        self.debug: bool = self._get_env("DEBUG", "False").lower() == "true"
    
    def _get_required_env(self, key: str) -> str:
        """Get a required environment variable."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def _get_env(self, key: str, default: str) -> str:
        """Get an optional environment variable with a default value."""
        return os.getenv(key, default)

# Global settings instance
settings = Settings()

"""
Application configuration using Pydantic Settings.

Loads configuration from environment variables and .env file.
"""

from functools import lru_cache
from typing import NamedTuple

from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(NamedTuple):
    """Configuration for a single model."""

    name: str  # API-facing identifier
    handler_key: str  # Handler implementation selector
    model_dir: str  # Directory containing model artifacts


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Authentication
    AI_API_TOKEN: str

    # Model Registry
    # Format: model_name:handler_key:model_dir
    # Example: tabr:tabr:tabr_experiment/before_flatten
    MODEL_REGISTRY: str = ""

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # CORS settings
    # Multiple origins separated by comma, e.g. "http://localhost:3000,https://example.com"
    CORS_ORIGINS: str = "*"
    CORS_METHODS: str = "*"

    def get_cors_origins(self) -> list[str]:
        """Parse CORS_ORIGINS into list."""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    def get_cors_methods(self) -> list[str]:
        """Parse CORS_METHODS into list."""
        if self.CORS_METHODS == "*":
            return ["*"]
        return [m.strip() for m in self.CORS_METHODS.split(",") if m.strip()]

    def parse_model_registry(self) -> list[ModelConfig]:
        """
        Parse MODEL_REGISTRY string into list of ModelConfig.

        Format: model_name:handler_key:model_dir
        Multiple models can be separated by comma.

        Returns:
            List of ModelConfig namedtuples.
        """
        if not self.MODEL_REGISTRY:
            return []

        models = []
        entries = self.MODEL_REGISTRY.split(",")

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            parts = entry.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid MODEL_REGISTRY entry: '{entry}'. "
                    "Expected format: model_name:handler_key:model_dir"
                )

            name, handler_key, model_dir = [p.strip() for p in parts]
            models.append(
                ModelConfig(name=name, handler_key=handler_key, model_dir=model_dir)
            )

        return models


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()

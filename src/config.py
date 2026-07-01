"""settings via pydantic-settings"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # llms
    openai_api_key: str = Field(default="", description="openai api key")
    anthropic_api_key: str = Field(default="", description="anthropic api key")
    voyage_api_key: str = Field(default="", description="voyage rerank api key")
    cohere_api_key: str = Field(default="", description="cohere rerank api key (fallback)")

    # vector db
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="documents")

    # app
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)

    # logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

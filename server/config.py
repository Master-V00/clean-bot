"""
Server configuration and environment variables
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables (.env supported)."""

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # Google Drive API
    GOOGLE_SERVICE_ACCOUNT_JSON: str = ""
    GOOGLE_OUTPUT_FOLDER_ID: str = ""
    # OAuth tokens (set via .env only – never hardcode)
    GOOGLE_REFRESH_TOKEN: str = ""
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Paths
    PROFILES_DIR: str = "pcleaner/profiles"
    TEMP_CHAPTERS_DIR: str = "temp_chapters"

    # Cleanup
    CLEANUP_INTERVAL_SECONDS: int = 60
    EXPIRATION_TIME_MINUTES: int = 20

    # API Key for Discord Bot authentication
    DISCORD_BOT_API_KEY: str = ""
    API_KEY: str = ""  # falls back to DISCORD_BOT_API_KEY if empty

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

# Ensure API_KEY always has a value (fallback to DISCORD_BOT_API_KEY)
if not settings.API_KEY:
    settings.API_KEY = settings.DISCORD_BOT_API_KEY

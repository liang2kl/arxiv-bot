"""Configuration management for ArXiv Bot."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # AI Provider Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    default_ai_provider: str = Field("openai", env="DEFAULT_AI_PROVIDER")
    
    # Bot Configuration
    slack_bot_token: Optional[str] = Field(None, env="SLACK_BOT_TOKEN")
    slack_app_token: Optional[str] = Field(None, env="SLACK_APP_TOKEN")
    telegram_bot_token: Optional[str] = Field(None, env="TELEGRAM_BOT_TOKEN")
    
    # Database Configuration
    database_url: str = Field("sqlite:///arxiv_bot.db", env="DATABASE_URL")
    
    # Monitoring Configuration
    monitor_interval_hours: int = Field(6, env="MONITOR_INTERVAL_HOURS")
    max_papers_per_check: int = Field(5, env="MAX_PAPERS_PER_CHECK")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings() 
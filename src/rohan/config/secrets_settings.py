from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for secrets management."""

    model_config = SettingsConfigDict(secrets_dir="/run/secrets")

    openrouter_api_key: SecretStr

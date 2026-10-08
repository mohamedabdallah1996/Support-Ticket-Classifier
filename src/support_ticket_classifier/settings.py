from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GROQ_API_KEY: str
    LANGSMITH_API_KEY: str = ""  # Optional only needed if using LangSmith for tracing and debugging.

    GROQ_GENERATION_MODEL_NAME: str = "openai/gpt-oss-20b"
    GROQ_GUARD_MODEL_NAME: str = "openai/gpt-oss-20b"

    LOG_COSTS: bool = True
    MAX_CLASSIFICATION_ATTEMPTS: int = 3

    DEFAULT_REGION_PII: str = "EG"  # Default region for phone number parsing in pii detection/redaction. 

settings = Settings()
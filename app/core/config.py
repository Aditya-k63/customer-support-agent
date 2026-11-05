from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DB_NAME: str = "agent"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "password"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # LLM - Primary
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"

    # LLM - Fallback (for rate limiting)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct"

    # API
    API_KEY: str = "dev-secret-change-in-production"
    ALLOWED_HOSTS: str = "localhost,127.0.0.1,testserver"
    CORS_ORIGINS: str = "http://localhost:3000"

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # App
    ENVIRONMENT: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()
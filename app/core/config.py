from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Database
    DB_NAME: str = "agent"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_ADMIN_DB: str = "postgres"
    DB_CONNECT_RETRIES: int = 30
    DB_CONNECT_DELAY: float = 2.0

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def SYNC_DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # LLM - Primary
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"

    # LLM - Fallback (for rate limiting)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct"
    # "auto" = Groq first, OpenRouter on failure. "groq" / "openrouter" pin a
    # single provider, which is useful for cost control and for running the
    # evaluation against a provider with a known quota.
    LLM_PROVIDER: str = "auto"
    LLM_MAX_RETRIES: int = 5
    LLM_RETRY_BASE_DELAY: float = 2.0

    # API
    API_KEY: str = ""
    ADMIN_API_KEY: str = ""
    ALLOWED_HOSTS: str = "localhost,127.0.0.1,testserver"
    CORS_ORIGINS: str = "http://localhost:3000"

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # Retrieval / embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_THRESHOLD: float = 0.0
    MAX_CONTEXT_CHUNKS: int = 5
    CHUNK_SIZE: int = 200
    CHUNK_OVERLAP: int = 30
    CONVERSATION_HISTORY_LIMIT: int = 20

    # App
    ENVIRONMENT: str = "production"
    RUN_MIGRATIONS_ON_STARTUP: bool = True


settings = Settings()

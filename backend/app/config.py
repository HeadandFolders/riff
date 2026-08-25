from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="RIFF_", extra="ignore"
    )

    project_id: str
    location: str = "us-central1"

    # Verify both against the current Vertex model catalog before submission.
    reasoning_model: str = "gemini-3.5-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    firestore_database: str = "(default)"
    pdf_bucket: str = ""

    alphaxiv_mcp_url: str = "https://api.alphaxiv.org/mcp/v1"
    alphaxiv_api_key: str = ""

    # Cosine similarity above known_threshold means you hold the concept;
    # between the two means partial. Tuned per user from feedback events.
    known_threshold: float = 0.78
    partial_threshold: float = 0.62

    # Sections whose gap count exceeds this are flagged before you start.
    section_gap_warning: int = 4

    # Every Nth paper, the Examiner makes you defend your best open hypothesis.
    defence_interval: int = 10


@lru_cache
def settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

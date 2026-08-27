from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="RIFF_", extra="ignore"
    )

    project_id: str
    #: Region for deployed resources: Cloud Run, the PDF bucket, schedulers.
    location: str = "us-central1"
    #: Vertex endpoint for model calls. Gemini 3.x is only served by "global";
    #: regional endpoints answer 404 for it.
    vertex_location: str = "global"

    reasoning_model: str = "gemini-3.5-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    firestore_database: str = "(default)"
    pdf_bucket: str = ""

    alphaxiv_mcp_url: str = "https://api.alphaxiv.org/mcp/v1"
    alphaxiv_api_key: str = ""
    # Retrieval runs an agentic loop server-side, so it is slow by nature.
    alphaxiv_timeout_seconds: int = 180

    # Retrieval prefilter only: above this, the graph plausibly holds material
    # for a concept and it is worth asking the reader about it. This threshold
    # never decides whether the reader understands anything.
    presence_threshold: float = 0.62

    # Sections whose gap count exceeds this are flagged before you start.
    section_gap_warning: int = 4

    # Concepts to assess per section. Assessment costs a Gemini call each, so
    # the Cartographer ranks candidates and only the top ones get graded.
    max_assessments_per_section: int = 3

    # Every Nth paper, the Examiner makes you defend your best open hypothesis.
    defence_interval: int = 10

    queue_page_size: int = 20
    graph_node_limit: int = 600


@lru_cache
def settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

"""Runtime configuration and environment settings for DeepScholar-MCP."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


class Settings(BaseSettings):
    """DeepScholar runtime settings loaded from environment and .env file."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE if _ENV_FILE.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === L1: Deep Search Discovery APIs ===
    EXA_API_KEY: str | None = None
    TAVILY_API_KEY: str | None = None
    SERPER_API_KEY: str | None = None
    ANYSEARCH_API_URL: str | None = None
    ANYSEARCH_API_KEY: str | None = None

    # === L2: Academic Metadata & Open Access APIs ===
    S2_API_KEY: str | None = None
    OPENALEX_API_KEY: str | None = None
    OPENALEX_EMAIL: str = "researcher@example.com"
    UNPAYWALL_EMAIL: str = "researcher@example.com"
    CORE_API_KEY: str | None = None

    # === L3: Browser Rendering & Extraction Fallback ===
    CF_ACCOUNT_ID: str | None = None
    CF_API_TOKEN: str | None = None
    JINA_API_KEY: str | None = None

    # === Runtime, Cache & Timeout Settings ===
    CACHE_DB_PATH: str = ".deepscholar_cache.sqlite3"
    TOTAL_TIMEOUT_SEC: float = 35.0  # Unified end-to-end hard deadline for MCP operations
    L1_TIMEOUT_SEC: float = 12.0
    L2_TIMEOUT_SEC: float = 12.0
    L3_TIMEOUT_SEC: float = 12.0
    MAX_CONCURRENT_REQUESTS: int = 8
    MAX_FULLTEXT_CHARS: int = 35000

    @property
    def has_exa(self) -> bool:
        """Check if Exa API credentials are configured."""
        return bool(self.EXA_API_KEY)

    @property
    def has_tavily(self) -> bool:
        """Check if Tavily API credentials are configured."""
        return bool(self.TAVILY_API_KEY)

    @property
    def has_serper(self) -> bool:
        """Check if Serper API credentials are configured."""
        return bool(self.SERPER_API_KEY)

    @property
    def has_core(self) -> bool:
        """Check if CORE API credentials are configured."""
        return bool(self.CORE_API_KEY)

    @property
    def has_anysearch(self) -> bool:
        """Check if AnySearch API endpoint is configured."""
        return bool(self.ANYSEARCH_API_URL)

    @property
    def cloudflare_account_id(self) -> str | None:
        """Get Cloudflare Account ID from settings or environment."""
        import os
        return self.CF_ACCOUNT_ID or os.environ.get("CLOUDFLARE_ACCOUNT_ID")

    @property
    def cloudflare_api_token(self) -> str | None:
        """Get Cloudflare API Token from settings or environment."""
        import os
        return self.CF_API_TOKEN or os.environ.get("CLOUDFLARE_API_TOKEN")

    @property
    def has_cloudflare_browser(self) -> bool:
        """Check if Cloudflare Browser Rendering credentials are configured."""
        return bool(self.cloudflare_account_id and self.cloudflare_api_token)

"""Pydantic v2 data models for DeepScholar-MCP."""

from typing import Literal

from pydantic import BaseModel, Field


class IdentifierSet(BaseModel):
    """Canonical academic identifiers for a publication."""

    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None

    def has_any(self) -> bool:
        """Check if at least one identifier is populated."""
        return bool(
            self.doi
            or self.arxiv_id
            or self.pmid
            or self.s2_id
            or self.openalex_id
        )

    def to_s2_lookup_id(self) -> str | None:
        """Format the identifier for Semantic Scholar API lookup."""
        if self.doi:
            return f"DOI:{self.doi}"
        if self.arxiv_id:
            return f"ARXIV:{self.arxiv_id}"
        if self.pmid:
            return f"PMID:{self.pmid}"
        if self.s2_id:
            return self.s2_id
        return None

    # Alias for spec case compatibility
    to_s2_Lookup_id = to_s2_lookup_id


class RawDiscoveryHit(BaseModel):
    """Raw search result hit from L1 search engines or academic databases."""

    title: str
    url: str
    snippet: str = ""
    source_engine: str
    published_date: str | None = None
    pre_extracted_ids: IdentifierSet = Field(default_factory=IdentifierSet)
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    citation_count: int | None = None
    venue: str | None = None
    oa_pdf_url: str | None = None
    rank: int = 0


class PaperRecord(BaseModel):
    """Enriched, canonical academic paper representation."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None

    # Official author abstract from trusted academic APIs or explicit abstract sections
    abstract: str | None = None
    abstract_source: str | None = None
    tldr: str | None = None

    # Search engine snippet (distinct from official abstract)
    search_snippet: str | None = None
    search_snippet_source: str | None = None

    # Web extraction excerpt (fallback when no formal abstract section could be verified)
    extracted_excerpt: str | None = None
    extracted_excerpt_source: str | None = None

    # Relevance and provenance metadata
    relevance_score: float | None = None
    identifiers: IdentifierSet = Field(default_factory=IdentifierSet)
    url: str
    oa_pdf_url: str | None = None
    source_tier: Literal["L2_Academic_API", "L3_Browser_Fallback", "L1_Snippet_Only"]
    resolved_via: list[str] = Field(default_factory=list)


class FullTextResult(BaseModel):
    """Extracted full-text Markdown payload and metadata."""

    paper_id_or_url: str
    resolved_url: str
    title: str | None = None
    extraction_method: Literal[
        "ar5iv_html",
        "exa_contents",
        "cloudflare_browser",
        "jina_reader",
        "pymupdf4llm_pdf",
        "failed",
    ]
    available_sections: list[str] = Field(default_factory=list)
    markdown_content: str
    truncated: bool = False

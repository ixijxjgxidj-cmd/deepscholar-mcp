"""FastMCP server entry point exposing DeepScholar academic retrieval tools."""

from typing import Any, Literal

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from mcp.server.mcpserver import MCPServer as FastMCP

from deepscholar_mcp.config import Settings
from deepscholar_mcp.pipeline import DeepScholarPipeline

# Initialize FastMCP Server
mcp = FastMCP("DeepScholar-MCP")

# Global pipeline instance
_settings = Settings()
_pipeline = DeepScholarPipeline(settings=_settings)


@mcp.tool()
async def search_literature(
    query: str,
    limit: int = 10,
    year_from: int | None = None,
    force_browser_fallback_for_missing_abstracts: bool = True,
) -> list[dict[str, Any]]:
    """Search academic literature using a 4-stage concurrent waterfall architecture.

    Features:
    - Multi-engine recall across AI search engines (Exa, Tavily, Serper Google Scholar)
      and academic databases (OpenAlex, Semantic Scholar).
    - Regex identifier extraction (DOI, arXiv ID, PMID) and fuzzy entity alignment (RapidFuzz >= 85%).
    - High-throughput batch enrichment via Semantic Scholar, OpenAlex, arXiv Atom, and Unpaywall OA.
    - Automatic browser rendering fallback for newly published conference papers or preprints missing abstracts.
    - Persistent SQLite caching for fast repeated lookups.

    Args:
        query: Research topic, query keywords, or paper title to search for.
        limit: Maximum number of enriched papers to return (default: 10, max: 50).
        year_from: Optional minimum publication year filter (e.g., 2021).
        force_browser_fallback_for_missing_abstracts: When True, retrieves missing abstracts
            for preprints/conference papers via L3 browser extraction (default: True).

    Returns:
        List of paper records containing title, authors, publication year, venue,
        citation metrics, abstract/TLDR, canonical identifiers (DOI, arXiv ID, PMID),
        source URLs, and direct Open Access PDF links.
    """
    bounded_limit = max(1, min(limit, 50))
    papers = await _pipeline.search_literature(
        query=query,
        limit=bounded_limit,
        year_from=year_from,
        force_browser_fallback_for_missing_abstracts=force_browser_fallback_for_missing_abstracts,
    )
    return [p.model_dump() for p in papers]


@mcp.tool()
async def fetch_paper_fulltext(
    paper_id_or_url: str,
    sections: list[str] | None = None,
    max_chars: int = 35000,
) -> dict[str, Any]:
    """Retrieve structured Markdown full text of an academic paper.

    Multi-tier extraction waterfall:
    - Priority 1: arXiv HTML / ar5iv conversion to clean Markdown with preserved LaTeX math.
    - Priority 2: Unpaywall Open Access PDF resolution for DOIs.
    - Priority 3: Direct streaming PDF download parsed locally via PyMuPDF4LLM.
    - Priority 4: Web / SPA extraction via Exa /contents, Cloudflare Browser Rendering,
      or Jina Reader.
    - Section filtering: Slice specific sections (e.g. ['method', 'results', 'abstract']).
    - Token budget guard: Enforce max_chars limit with safe truncation.

    Args:
        paper_id_or_url: Paper identifier (e.g. '10.1038/s41586-020-2649-2', '2403.01234',
            'arXiv:2403.01234') or web landing URL / PDF URL.
        sections: Optional list of section heading keywords to filter
            (e.g., ['abstract', 'method', 'experiment', 'conclusion']). Case-insensitive substring match.
        max_chars: Maximum character budget for the returned Markdown (default: 35000).

    Returns:
        FullTextResult dictionary containing paper_id_or_url, resolved_url, extraction_method,
        available_sections, markdown_content, and truncated flag.
    """
    res = await _pipeline.fetch_paper_fulltext(
        paper_id_or_url=paper_id_or_url,
        sections=sections,
        max_chars=max_chars,
    )
    return res.model_dump()


@mcp.tool()
async def expand_citation_graph(
    paper_id: str,
    direction: Literal["citations", "references", "both"] = "both",
    limit: int = 15,
) -> dict[str, Any]:
    """Expand citation and reference graph network for a paper.

    Traverses literature graphs using Semantic Scholar Graph API, automatically
    ranking related works by influential citations and raw citation counts.
    Includes automated failover to OpenAlex if Semantic Scholar experiences rate limiting.

    Args:
        paper_id: Semantic Scholar ID, DOI (e.g. '10.1038/...'), or arXiv ID (e.g. '2403.01234').
        direction: Direction of graph traversal: 'citations', 'references', or 'both' (default: 'both').
        limit: Maximum number of related papers to return per direction (default: 15).

    Returns:
        Graph dictionary containing paper_id, direction, total count, and sorted lists of papers.
    """
    bounded_limit = max(1, min(limit, 50))
    return await _pipeline.expand_citation_graph(
        paper_id=paper_id,
        direction=direction,
        limit=bounded_limit,
    )


def main() -> None:
    """Run FastMCP server."""
    mcp.run()


if __name__ == "__main__":
    main()

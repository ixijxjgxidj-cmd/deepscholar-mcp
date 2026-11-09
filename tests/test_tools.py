"""Tests for section slicing, SQLite cache persistence/expiration, and citation graph expansion."""

import asyncio

import httpx
import pytest
import respx
from deepscholar_mcp.cache import AsyncSQLiteCache
from deepscholar_mcp.config import Settings
from deepscholar_mcp.pipeline import DeepScholarPipeline
from deepscholar_mcp.providers.l3_fallback import slice_markdown_sections


def test_slice_markdown_sections_filtering() -> None:
    sample_markdown = """# Introduction
This is the introduction discussing deep learning architectures.

## Methodology
Here we describe our Transformer with dynamic sparse attention.
### 2.1 Attention Formulation
The attention weights are computed using Gaussian kernels.

## Experiments
We evaluate our model on ImageNet and GLUE benchmarks.

# Conclusion
Our model achieves competitive performance with 40% fewer FLOPs.
"""

    available, sliced, truncated = slice_markdown_sections(
        sample_markdown, requested_sections=["Method"], max_chars=10000
    )

    assert "Introduction" in available
    assert "Methodology" in available
    assert "2.1 Attention Formulation" in available
    assert "Experiments" in available
    assert "Conclusion" in available

    assert "Methodology" in sliced
    assert "Gaussian kernels" in sliced
    assert "ImageNet and GLUE" not in sliced
    assert "competitive performance" not in sliced
    assert truncated is False


def test_slice_markdown_truncation() -> None:
    sample_text = "# Header\n" + ("A" * 500)
    _, sliced, truncated = slice_markdown_sections(
        sample_text, requested_sections=None, max_chars=100
    )
    assert truncated is True
    assert len(sliced) > 0
    assert "[... Truncated due to max_chars limit ...]" in sliced


@pytest.mark.asyncio
async def test_async_sqlite_cache_persistence_and_ttl(tmp_path) -> None:
    db_file = str(tmp_path / "test_cache_lifecycle.sqlite3")
    cache = AsyncSQLiteCache(db_path=db_file)

    # 1. Test JSON set & get
    payload = {"title": "Test Paper", "year": 2024}
    await cache.set_json("paper:123", payload, ttl=3600)

    cached_json = await cache.get_json("paper:123")
    assert cached_json == payload

    # 2. Test Text set & get
    text_content = "# Full text markdown"
    await cache.set_text("fulltext:123", text_content, ttl=3600)

    cached_text = await cache.get_text("fulltext:123")
    assert cached_text == text_content

    # 3. Test TTL expiration
    await cache.set_json("paper:expiring", {"temp": True}, ttl=1)
    # Ensure it's there
    assert await cache.get_json("paper:expiring") == {"temp": True}

    # Wait for expiration
    await asyncio.sleep(1.1)

    # Expired entry returns None on get
    expired_result = await cache.get_json("paper:expiring")
    assert expired_result is None

    # Test purge_expired
    purged_count = await cache.purge_expired()
    assert purged_count >= 0


@pytest.mark.asyncio
@respx.mock
async def test_expand_citation_graph_semantic_scholar_success(tmp_path) -> None:
    settings = Settings(CACHE_DB_PATH=str(tmp_path / "test.db"))
    pipeline = DeepScholarPipeline(settings=settings)

    paper_id = "10.1038/s41586-020-2649-2"

    # Mock Semantic Scholar Graph Citations
    respx.get(
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{paper_id}/citations"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "citingPaper": {
                            "title": "Low Impact Followup",
                            "citationCount": 5,
                            "influentialCitationCount": 0,
                            "year": 2022,
                        }
                    },
                    {
                        "citingPaper": {
                            "title": "Highly Influential Study",
                            "citationCount": 250,
                            "influentialCitationCount": 42,
                            "year": 2023,
                        }
                    },
                ]
            },
        )
    )

    # Mock References
    respx.get(
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{paper_id}/references"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "citedPaper": {
                            "title": "Foundational Physics Paper",
                            "citationCount": 1200,
                            "influentialCitationCount": 150,
                            "year": 2017,
                        }
                    }
                ]
            },
        )
    )

    result = await pipeline.expand_citation_graph(paper_id, direction="both", limit=10)
    assert result["paper_id"] == paper_id
    assert result["direction"] == "both"
    assert len(result["citations"]) == 2
    # Verify sorted by influentialCitationCount descending
    assert result["citations"][0]["title"] == "Highly Influential Study"
    assert result["citations"][1]["title"] == "Low Impact Followup"


@pytest.mark.asyncio
@respx.mock
async def test_expand_citation_graph_openalex_fallback_on_429(tmp_path) -> None:
    settings = Settings(CACHE_DB_PATH=str(tmp_path / "test.db"))
    pipeline = DeepScholarPipeline(settings=settings)

    paper_id = "10.1038/s41586-020-2649-2"

    # Mock Semantic Scholar returning 429 Too Many Requests
    respx.get(
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{paper_id}/citations"
    ).mock(return_value=httpx.Response(429, text="Rate limit exceeded"))

    # Mock OpenAlex work lookup
    respx.get(
        f"https://api.openalex.org/works/https://doi.org/{paper_id}"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"id": "https://openalex.org/W2964032478"},
        )
    )

    # Mock OpenAlex works cites filter
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "display_name": "OpenAlex Citing Paper A",
                        "publication_year": 2024,
                        "cited_by_count": 88,
                        "doi": "https://doi.org/10.1000/182",
                    }
                ]
            },
        )
    )

    fallback_result = await pipeline.expand_citation_graph(
        paper_id, direction="citations", limit=5
    )
    assert fallback_result["direction"] == "citations"
    assert len(fallback_result["results"]) >= 1
    assert fallback_result["results"][0]["title"] == "OpenAlex Citing Paper A"
    assert fallback_result["results"][0]["citationCount"] == 88

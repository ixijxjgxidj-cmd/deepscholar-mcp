"""Mocked end-to-end waterfall pipeline and L1->L2->L3 fallback integration tests."""

import httpx
import pymupdf
import pytest
import respx
from deepscholar_mcp.cache import AsyncSQLiteCache
from deepscholar_mcp.config import Settings
from deepscholar_mcp.pipeline import DeepScholarPipeline


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    db_path = str(tmp_path / "test_cache.sqlite3")
    return Settings(
        EXA_API_KEY="test-exa-key",
        CF_ACCOUNT_ID="test-cf-account",
        CF_API_TOKEN="test-cf-token",
        CACHE_DB_PATH=db_path,
        L1_TIMEOUT_SEC=5.0,
        L2_TIMEOUT_SEC=5.0,
        L3_TIMEOUT_SEC=5.0,
    )


@pytest.fixture
def pipeline(test_settings) -> DeepScholarPipeline:
    cache = AsyncSQLiteCache(db_path=test_settings.CACHE_DB_PATH)
    return DeepScholarPipeline(settings=test_settings, cache=cache)


@pytest.mark.asyncio
@respx.mock
async def test_scenario_a_l1_to_l2_batch_hit(pipeline: DeepScholarPipeline) -> None:
    """Scenario A: Exa search returns hit with DOI -> Semantic Scholar Batch API enriches metadata."""
    doi = "10.1038/s41586-020-2649-2"
    # Mock OpenAlex Native and S2 Native empty responses
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    # Mock Exa Search returning hit with DOI in text
    respx.post("https://api.exa.ai/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "AlphaFold: High accuracy protein structure prediction",
                        "url": f"https://nature.com/articles/{doi}",
                        "text": f"Accurate structure prediction with deep learning. DOI: {doi}",
                        "publishedDate": "2020-07-15",
                    }
                ]
            },
        )
    )

    # Mock Semantic Scholar Batch API
    respx.post("https://api.semanticscholar.org/graph/v1/paper/batch").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "paperId": "s2_alphafold_123",
                    "title": "Highly accurate protein structure prediction with AlphaFold",
                    "authors": [{"name": "John Jumper"}, {"name": "Demis Hassabis"}],
                    "year": 2021,
                    "venue": "Nature",
                    "citationCount": 15420,
                    "influentialCitationCount": 2100,
                    "abstract": "Proteins are essential to life, and understanding their structure can shed light on their function. Here we present AlphaFold.",
                    "tldr": {"text": "AlphaFold can predict protein structures with atomic accuracy."},
                    "externalIds": {"DOI": doi},
                    "openAccessPdf": {"url": "https://nature.com/articles/alphafold.pdf"},
                    "url": f"https://doi.org/{doi}",
                }
            ],
        )
    )

    papers = await pipeline.search_literature("AlphaFold protein structure", limit=5)
    assert len(papers) >= 1
    target = papers[0]
    assert target.identifiers.doi == doi
    assert target.source_tier == "L2_Academic_API"
    assert "semantic_scholar_batch" in target.resolved_via
    assert target.citation_count == 15420
    assert target.influential_citation_count == 2100
    assert target.year == 2021
    assert "Demis Hassabis" in target.authors


@pytest.mark.asyncio
@respx.mock
async def test_scenario_b_l2_miss_to_l3_cloudflare_browser(
    pipeline: DeepScholarPipeline,
) -> None:
    """Scenario B: OpenReview hit missing abstract in L2 -> falls back to Cloudflare Browser Rendering."""
    paper_url = "https://openreview.net/forum?id=test_iclr_2026"

    # Mock OpenAlex Native and S2 Native empty
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    # Mock Exa returning the OpenReview paper
    respx.post("https://api.exa.ai/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Quantum Attention Networks for Molecule Generation",
                        "url": paper_url,
                        "text": "OpenReview forum submission discussion.",
                        "publishedDate": "2026-02-01",
                    }
                ]
            },
        )
    )

    # Mock Exa /contents returning 404 or empty
    respx.post("https://api.exa.ai/contents").mock(
        return_value=httpx.Response(404, text="Not found")
    )

    # Mock Cloudflare Browser Rendering markdown endpoint
    cf_url = (
        f"https://api.cloudflare.com/client/v4/accounts/{pipeline.settings.CF_ACCOUNT_ID}"
        "/browser-rendering/markdown"
    )
    rendered_markdown = (
        "# Quantum Attention Networks for Molecule Generation\n\n"
        "## Abstract\n\n"
        "We introduce Quantum Attention Networks (QAN), a novel deep learning framework "
        "leveraging parameterized quantum circuits to model molecular conformational spaces. "
        "Our experiments demonstrate state-of-the-art chemical property prediction across QM9 and PCQM4Mv2."
    )
    respx.post(cf_url).mock(
        return_value=httpx.Response(
            200,
            json={"success": True, "result": rendered_markdown},
        )
    )

    papers = await pipeline.search_literature(
        "Quantum Attention Networks OpenReview",
        limit=5,
        force_browser_fallback_for_missing_abstracts=True,
    )

    assert len(papers) >= 1
    target = papers[0]
    assert target.source_tier == "L3_Browser_Fallback"
    assert "l3_browser_fallback" in target.resolved_via
    assert target.abstract is not None
    assert "We introduce Quantum Attention Networks" in target.abstract


@pytest.mark.asyncio
@respx.mock
async def test_scenario_c_closed_access_unpaywall_to_pymupdf(
    pipeline: DeepScholarPipeline,
) -> None:
    """Scenario C: Closed-access DOI resolves via Unpaywall to OA PDF, parsed with PyMuPDF4LLM."""
    doi = "10.1016/j.cell.2024.01.001"
    pdf_oa_url = "https://cell.example.com/openaccess/cell-2024-01-001.pdf"

    # Mock Unpaywall API
    respx.get(f"https://api.unpaywall.org/v2/{doi}").mock(
        return_value=httpx.Response(
            200,
            json={
                "doi": doi,
                "is_oa": True,
                "best_oa_location": {
                    "url_for_pdf": pdf_oa_url,
                    "version": "publishedVersion",
                },
            },
        )
    )

    # Generate in-memory valid 1-page PDF using PyMuPDF
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        pymupdf.Point(50, 72),
        "# Molecular Mechanisms of Cellular Reprogramming\n\n"
        "## Abstract\n\n"
        "Cellular reprogramming is governed by stochastic transcription factor binding.\n\n"
        "## Methodology\n\n"
        "We performed single-cell RNA sequencing on 50,000 murine embryonic fibroblasts.",
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    # Mock direct PDF download
    respx.get(pdf_oa_url).mock(
        return_value=httpx.Response(
            200,
            content=pdf_bytes,
            headers={"Content-Type": "application/pdf"},
        )
    )

    # Execute fetch_paper_fulltext with requested section
    result = await pipeline.fetch_paper_fulltext(
        paper_id_or_url=doi,
        sections=["Methodology"],
        max_chars=20000,
    )

    assert result.extraction_method == "pymupdf4llm_pdf"
    assert result.resolved_url == pdf_oa_url
    assert "Methodology" in result.markdown_content
    assert "single-cell RNA sequencing" in result.markdown_content
    assert result.truncated is False

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


@pytest.mark.asyncio
@respx.mock
async def test_provenance_separation_snippets_and_excerpts(
    pipeline: DeepScholarPipeline,
) -> None:
    """Verify strict provenance: search snippets and unverified web text are never mistaken for author abstracts."""
    paper_url = "https://forum.openreview.net/forum?id=unstructured_discussion"

    # Mock empty academic API results
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    # Mock Exa returning hit with search snippet
    respx.post("https://api.exa.ai/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Discussion on Hyperparameter Tuning in RL",
                        "url": paper_url,
                        "text": "User123 commented: What learning rate did you use for PPO? Reviewer 2 replied...",
                        "publishedDate": "2026-03-01",
                    }
                ]
            },
        )
    )

    # Mock Exa contents returning 404
    respx.post("https://api.exa.ai/contents").mock(
        return_value=httpx.Response(404)
    )

    # Mock Cloudflare returning generic discussion text WITHOUT any Abstract heading
    cf_url = (
        f"https://api.cloudflare.com/client/v4/accounts/{pipeline.settings.CF_ACCOUNT_ID}"
        "/browser-rendering/markdown"
    )
    unstructured_web_markdown = (
        "# Forum Discussion: Tuning PPO in Complex Robotics Tasks\n\n"
        "Navigation: Home > Forums > Topics > Reinforcement Learning Algorithms\n\n"
        "Comments and Feedback from Community:\n"
        "- Alice: I observed policy collapse at step 500k when scaling up batch sizes.\n"
        "- Bob: Try decreasing the entropy coefficient and tuning the GAE lambda parameter.\n"
        "- Charlie: Clipping ratio between 0.1 and 0.2 worked best in our benchmark trials."
    )
    respx.post(cf_url).mock(
        return_value=httpx.Response(
            200,
            json={"success": True, "result": unstructured_web_markdown},
        )
    )

    papers = await pipeline.search_literature(
        "Hyperparameter Tuning in RL",
        limit=5,
        force_browser_fallback_for_missing_abstracts=True,
    )

    assert len(papers) >= 1
    target = papers[0]
    # The official abstract MUST remain None because no trusted academic API or formal abstract section was found!
    assert target.abstract is None
    assert target.abstract_source is None

    # The raw search snippet and web excerpt must be cleanly recorded under their separate provenance fields
    assert target.search_snippet is not None
    assert "User123 commented" in target.search_snippet
    assert target.search_snippet_source == "exa"

    assert target.extracted_excerpt is not None
    assert "Navigation: Home" in target.extracted_excerpt
    assert target.extracted_excerpt_source == "cloudflare_browser"


@pytest.mark.asyncio
@respx.mock
async def test_l1_discovery_timeout_preserves_completed_results(
    pipeline: DeepScholarPipeline,
) -> None:
    """Verify that a slow provider timing out does NOT cause all L1 results to be lost."""
    import asyncio

    from deepscholar_mcp.providers.l1_discovery import run_l1_discovery

    # Mock OpenAlex returning immediately
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "display_name": "Fast Responding Paper",
                        "doi": "https://doi.org/10.1000/fast",
                        "abstract_inverted_index": {"Fast": [0], "results.": [1]},
                    }
                ]
            },
        )
    )

    # Mock S2 returning empty immediately
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    # Mock Exa with a slow delay that will exceed a tight timeout
    async def _delayed_exa(request):
        await asyncio.sleep(2.0)
        return httpx.Response(200, json={"results": []})

    respx.post("https://api.exa.ai/search").mock(side_effect=_delayed_exa)

    async with httpx.AsyncClient() as client:
        provider_statuses: dict[str, str] = {}
        hits = await run_l1_discovery(
            client=client,
            query="Fast Responding Paper",
            limit=5,
            settings=pipeline.settings,
            timeout=0.3,  # Tight timeout to trigger Exa timeout while OpenAlex finishes
            provider_statuses=provider_statuses,
        )

        # Completed OpenAlex hits MUST be preserved and not dropped!
        assert len(hits) >= 1
        assert hits[0].title == "Fast Responding Paper"

        # Check provider status tracking
        assert provider_statuses.get("openalex_native") == "success"
        assert provider_statuses.get("exa") == "timeout"


@pytest.mark.asyncio
@respx.mock
async def test_relevance_ranking_rrf_and_recency_over_raw_citations(
    pipeline: DeepScholarPipeline,
) -> None:
    """Verify multi-factor ranking: new relevant paper with multi-engine recall ranks above high-citation irrelevant paper."""
    query = "Equivariant Diffusion Models for Molecule Generation"

    # Mock OpenAlex returning two papers:
    # 1. New 2026 exact title match with 0 citations
    # 2. Old 2005 paper with 20,000 citations on unrelated topic
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "display_name": "Survey of Classical Data Structures in C++",
                        "publication_year": 2005,
                        "cited_by_count": 20000,
                        "doi": "https://doi.org/10.1000/old_cpp",
                        "abstract_inverted_index": {"Old": [0]},
                    },
                    {
                        "display_name": "Equivariant Diffusion Models for Molecule Generation",
                        "publication_year": 2026,
                        "cited_by_count": 0,
                        "doi": "https://doi.org/10.1000/new_molecules",
                        "abstract_inverted_index": {"Molecules": [0]},
                    },
                ]
            },
        )
    )

    # Mock Semantic Scholar also finding the 2026 paper (yielding multi-engine RRF consensus!)
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "s2_new_mol",
                        "title": "Equivariant Diffusion Models for Molecule Generation",
                        "year": 2026,
                        "citationCount": 0,
                        "externalIds": {"DOI": "10.1000/new_molecules"},
                        "abstract": "We present equivariant diffusion models for 3D molecular graph generation.",
                    }
                ]
            },
        )
    )

    respx.post("https://api.exa.ai/search").mock(
        return_value=httpx.Response(200, json={"results": []})
    )

    papers = await pipeline.search_literature(query, limit=5)

    assert len(papers) >= 2
    top_paper = papers[0]
    # The new relevant paper MUST rank first despite having 0 citations compared to 20,000!
    assert "Equivariant Diffusion Models" in top_paper.title
    assert top_paper.year == 2026
    assert top_paper.relevance_score is not None
    assert top_paper.relevance_score > papers[1].relevance_score


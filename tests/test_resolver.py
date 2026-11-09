"""Unit tests for identifier extraction, title cleaning, OpenAlex abstract reconstruction, and entity alignment."""

import httpx
import pytest
import respx
from deepscholar_mcp.resolver import (
    align_title_via_openalex,
    clean_academic_title,
    extract_identifiers,
    reconstruct_openalex_abstract,
)


def test_extract_identifiers_complex_doi() -> None:
    # 1. Complex Wiley DOI with parentheses and angle brackets
    text = (
        "Relevant paper: 10.1002/(SICI)1097-0142(19960715)78:2<285::AID-CNCR14>3.0.CO;2-1. "
        "Also see (10.1038/s41586-020-2649-2/pdf)."
    )
    ids = extract_identifiers(text)
    assert ids.doi is not None
    assert "10.1002/(sici)1097-0142(19960715)78:2<285::aid-cncr14>3.0.co;2-1" in ids.doi

    # 2. DOI URL with /pdf suffix and trailing parenthesis
    ids_url = extract_identifiers(
        "", url="https://nature.com/articles/10.1038/s41586-020-2649-2/pdf"
    )
    assert ids_url.doi == "10.1038/s41586-020-2649-2"


def test_extract_identifiers_arxiv_modern_and_legacy() -> None:
    # Modern arXiv URL with version and trailing slash/text
    url_modern = "https://arxiv.org/abs/2403.09876v2"
    ids = extract_identifiers("Check out this paper", url=url_modern)
    assert ids.arxiv_id == "2403.09876"

    # ar5iv HTML URL
    url_ar5iv = "https://ar5iv.labs.arxiv.org/html/2310.01234"
    ids_ar5iv = extract_identifiers("", url=url_ar5iv)
    assert ids_ar5iv.arxiv_id == "2310.01234"

    # Legacy arXiv ID
    text_legacy = "Reference [1]: arxiv:math/0102034v1"
    ids_legacy = extract_identifiers(text_legacy)
    assert ids_legacy.arxiv_id == "math/0102034"


def test_extract_identifiers_pubmed() -> None:
    url_pmid = "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    ids = extract_identifiers("Medical trial results", url=url_pmid)
    assert ids.pmid == "12345678"

    text_pmid = "Paper published on NCBI PMID: 98765432."
    ids_text = extract_identifiers(text_pmid)
    assert ids_text.pmid == "98765432"


def test_clean_academic_title() -> None:
    raw_1 = "[PDF] Attention Is All You Need | IEEE Xplore"
    assert clean_academic_title(raw_1) == "Attention Is All You Need"

    raw_2 = "[HTML] Deep Residual Learning for Image Recognition - ScienceDirect"
    assert clean_academic_title(raw_2) == "Deep Residual Learning for Image Recognition"

    raw_3 = "Language Models are Few-Shot Learners — arXiv.org"
    assert clean_academic_title(raw_3) == "Language Models are Few-Shot Learners"

    raw_4 = "  [BOOK] Foundations of Machine Learning – SpringerLink   "
    assert clean_academic_title(raw_4) == "Foundations of Machine Learning"


def test_reconstruct_openalex_abstract() -> None:
    inverted_index = {
        "Deep": [0],
        "learning": [1],
        "has": [2],
        "revolutionized": [3],
        "natural": [4],
        "language": [5],
        "processing.": [6],
    }
    abstract = reconstruct_openalex_abstract(inverted_index)
    assert abstract == "Deep learning has revolutionized natural language processing."

    assert reconstruct_openalex_abstract(None) is None
    assert reconstruct_openalex_abstract({}) is None


@pytest.mark.asyncio
@respx.mock
async def test_align_title_via_openalex() -> None:
    async with httpx.AsyncClient() as client:
        # Mock high similarity match (>= 85%)
        respx.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "https://openalex.org/W2964032478",
                            "title": "Attention Is All You Need",
                            "doi": "https://doi.org/10.48550/arxiv.1706.03762",
                            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/31234567"},
                            "locations": [
                                {
                                    "landing_page_url": "https://arxiv.org/abs/1706.03762",
                                    "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
                                }
                            ],
                        }
                    ]
                },
            )
        )

        matched_ids = await align_title_via_openalex(
            client,
            raw_title="Attention is All you Need (NeurIPS 2017) | arXiv",
            email="test@example.com",
        )
        assert matched_ids.doi == "10.48550/arxiv.1706.03762"
        assert matched_ids.arxiv_id == "1706.03762"
        assert matched_ids.pmid == "31234567"
        assert matched_ids.openalex_id == "https://openalex.org/W2964032478"


@pytest.mark.asyncio
@respx.mock
async def test_align_title_via_openalex_low_similarity_rejected() -> None:
    async with httpx.AsyncClient() as client:
        # Mock low similarity (< 85%)
        respx.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "https://openalex.org/W111111",
                            "title": "Quantum Computing Algorithms in Cryptography",
                            "doi": "https://doi.org/10.1000/182",
                        }
                    ]
                },
            )
        )

        rejected_ids = await align_title_via_openalex(
            client,
            raw_title="Attention Is All You Need",
            email="test@example.com",
        )
        assert rejected_ids.has_any() is False

"""Identifier extraction, title normalization, OpenAlex inverted index reconstruction, and fuzzy entity alignment."""

import logging
import re
import urllib.parse
from typing import Any

import httpx
from rapidfuzz import fuzz

from deepscholar_mcp.models import IdentifierSet

logger = logging.getLogger(__name__)

# DOI regex matching prefix 10.xxxx with typical DOI characters
DOI_PATTERN = re.compile(
    r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9<>]+)",
    re.IGNORECASE,
)

# Modern arXiv pattern (YYMM.NNNN or YYMM.NNNNN)
ARXIV_MODERN_PATTERN = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/|ar5iv\.labs\.arxiv\.org/html/|arxiv\s*:\s*|^)(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)

# Legacy arXiv pattern (archive/YYMMNNN)
ARXIV_LEGACY_PATTERN = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv\s*:\s*|^)([a-z\-]+/\d{7})(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)

# PubMed PMID pattern
PMID_PATTERN = re.compile(
    r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|ncbi\.nlm\.nih\.gov/pubmed/|PMID\s*:\s*)(\d{6,9})",
    re.IGNORECASE,
)

# Leading academic tags like [PDF], [HTML], etc.
LEADING_TAGS_PATTERN = re.compile(
    r"^\s*\[(PDF|HTML|BOOK|DOC|CITATION)\]\s*",
    re.IGNORECASE,
)

# Trailing venue/repository suffixes
TRAILING_SUFFIX_PATTERN = re.compile(
    r"\s*[\-–—|]\s*(?:IEEE Xplore|ScienceDirect|SpringerLink|Springer|ACM Digital Library|arXiv\.org|arXiv|OpenReview|ACL Anthology|PubMed|NCBI|Nature|ResearchGate|Semantic Scholar).*$",
    re.IGNORECASE,
)


def _clean_extracted_doi(candidate: str) -> str:
    """Strip trailing punctuation, unbalanced brackets, and path suffixes from DOI."""
    doi = candidate.strip()
    trailing_chars = ".,;:]}>'\""
    while doi and doi[-1] in trailing_chars:
        doi = doi[:-1]

    # Handle unbalanced closing parenthesis (e.g. from '(see 10.1000/182)')
    if doi.endswith(")"):
        open_count = doi.count("(")
        close_count = doi.count(")")
        if close_count > open_count:
            doi = doi[:-1]

    # Strip URL sub-path endings often attached to DOI URLs
    suffixes = ("/pdf", "/full", "/meta", "/epdf", "/abstract")
    for sfx in suffixes:
        if doi.lower().endswith(sfx):
            doi = doi[: -len(sfx)]
            break

    return doi.lower()


def extract_identifiers(text: str, url: str = "") -> IdentifierSet:
    """Extract DOI, arXiv ID, and PMID from arbitrary text and URL."""
    decoded_url = urllib.parse.unquote(url) if url else ""
    decoded_text = urllib.parse.unquote(text) if text else ""
    combined = f"{decoded_url} {decoded_text}".strip()

    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None

    # 1. DOI Extraction
    doi_match = DOI_PATTERN.search(combined)
    if doi_match:
        doi = _clean_extracted_doi(doi_match.group(1))

    # 2. arXiv ID Extraction
    modern_match = ARXIV_MODERN_PATTERN.search(combined)
    if modern_match:
        arxiv_id = modern_match.group(1)
    else:
        legacy_match = ARXIV_LEGACY_PATTERN.search(combined)
        if legacy_match:
            arxiv_id = legacy_match.group(1)

    # 3. PMID Extraction
    pmid_match = PMID_PATTERN.search(combined)
    if pmid_match:
        pmid = pmid_match.group(1)

    return IdentifierSet(doi=doi, arxiv_id=arxiv_id, pmid=pmid)


def clean_academic_title(raw_title: str) -> str:
    """Normalize academic paper title by stripping site prefixes/suffixes and excessive whitespace."""
    if not raw_title:
        return ""
    cleaned = LEADING_TAGS_PATTERN.sub("", raw_title)
    cleaned = TRAILING_SUFFIX_PATTERN.sub("", cleaned)
    # Strip common conference/preprint annotations in parentheses or brackets e.g. (NeurIPS 2017)
    cleaned = re.sub(
        r"\s*[\(\[](?:NeurIPS|ICLR|ICML|CVPR|ECCV|ICCV|ACL|EMNLP|NAACL|AAAI|IJCAI|arXiv)[^\)\]]*[\)\]]",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def reconstruct_openalex_abstract(
    inverted_index: dict[str, list[int]] | None,
) -> str | None:
    """Reconstruct linear abstract text from OpenAlex inverted index dictionary."""
    if not inverted_index:
        return None

    pos_word: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pos_word.append((pos, word))

    if not pos_word:
        return None

    pos_word.sort(key=lambda item: item[0])
    return " ".join(word for _, word in pos_word)


async def align_title_via_openalex(
    client: httpx.AsyncClient,
    raw_title: str,
    email: str,
) -> IdentifierSet:
    """Search OpenAlex by title and align with RapidFuzz token sorting (>= 85% similarity)."""
    cleaned_title = clean_academic_title(raw_title)
    if len(cleaned_title) < 12:
        return IdentifierSet()

    url = "https://api.openalex.org/works"
    params = {
        "search": cleaned_title,
        "per-page": 3,
        "mailto": email,
    }

    try:
        resp = await client.get(url, params=params, timeout=8.0)
        if resp.status_code != 200:
            logger.warning(
                "OpenAlex title alignment failed with status %d for title: %s",
                resp.status_code,
                cleaned_title,
            )
            return IdentifierSet()

        data: dict[str, Any] = resp.json()
        results: list[dict[str, Any]] = data.get("results", [])
        if not results:
            return IdentifierSet()

        best_score = 0.0
        best_work: dict[str, Any] | None = None
        target_title_lower = cleaned_title.lower()

        for work in results:
            candidate_title = (
                work.get("title") or work.get("display_name") or ""
            ).lower()
            if not candidate_title:
                continue
            score = fuzz.token_sort_ratio(target_title_lower, candidate_title)
            if score > best_score:
                best_score = score
                best_work = work

        if best_score < 85.0 or best_work is None:
            return IdentifierSet()

        # Extract identifiers from matched work
        raw_doi = best_work.get("doi")
        doi = None
        if raw_doi:
            doi = raw_doi.replace("https://doi.org/", "").strip().lower()

        ids = best_work.get("ids", {})
        pmid = None
        if "pmid" in ids and ids["pmid"]:
            pmid = str(ids["pmid"]).replace("https://pubmed.ncbi.nlm.nih.gov/", "").strip()

        openalex_id = best_work.get("id")

        arxiv_id = None
        # Check locations or ids for arXiv
        locations = best_work.get("locations", [])
        for loc in locations:
            landing = loc.get("landing_page_url") or ""
            pdf_url = loc.get("pdf_url") or ""
            arxiv_match = ARXIV_MODERN_PATTERN.search(f"{landing} {pdf_url}")
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
                break

        return IdentifierSet(
            doi=doi,
            arxiv_id=arxiv_id,
            pmid=pmid,
            openalex_id=openalex_id,
        )
    except Exception as exc:
        logger.warning(
            "Exception during OpenAlex title alignment for '%s': %s",
            cleaned_title,
            exc,
        )
        return IdentifierSet()

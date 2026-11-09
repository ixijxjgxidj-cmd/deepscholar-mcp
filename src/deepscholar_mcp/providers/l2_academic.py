"""Stage 3 — L2 Academic Batch Enrichment & OA Resolution providers."""

import logging
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

from deepscholar_mcp.config import Settings
from deepscholar_mcp.models import IdentifierSet, PaperRecord
from deepscholar_mcp.resolver import (
    clean_academic_title,
    extract_identifiers,
    reconstruct_openalex_abstract,
)

logger = logging.getLogger(__name__)


async def batch_enrich_s2(
    client: httpx.AsyncClient,
    s2_lookup_ids: list[str],
    settings: Settings,
) -> dict[str, PaperRecord]:
    """Batch enrich academic metadata using Semantic Scholar Graph Batch API."""
    unique_ids = list(dict.fromkeys(s2_lookup_ids))
    if not unique_ids:
        return {}

    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    params = {
        "fields": "paperId,title,authors,year,venue,citationCount,influentialCitationCount,abstract,tldr,externalIds,openAccessPdf,url"
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.S2_API_KEY:
        headers["x-api-key"] = settings.S2_API_KEY

    enriched: dict[str, PaperRecord] = {}

    # Semantic Scholar allows up to 500 IDs per batch; chunk to 100 for safety
    chunk_size = 100
    for i in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[i : i + chunk_size]
        try:
            resp = await client.post(
                url,
                params=params,
                headers=headers,
                json={"ids": chunk},
                timeout=12.0,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Semantic Scholar batch enrichment returned status %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                continue

            items: list[dict[str, Any] | None] = resp.json()
            for idx, item in enumerate(items):
                if not item:
                    continue

                paper_id = item.get("paperId")
                title = clean_academic_title(item.get("title") or "")
                authors = [
                    a.get("name")
                    for a in item.get("authors", [])
                    if a.get("name")
                ]
                year = item.get("year")
                venue = item.get("venue")
                citation_count = item.get("citationCount")
                influential_count = item.get("influentialCitationCount")
                abstract = item.get("abstract")

                tldr_obj = item.get("tldr")
                tldr = (
                    tldr_obj.get("text")
                    if isinstance(tldr_obj, dict)
                    else None
                )

                ext_ids = item.get("externalIds") or {}
                raw_doi = ext_ids.get("DOI")
                doi = raw_doi.strip().lower() if raw_doi else None
                arxiv_id = ext_ids.get("ArXiv")
                pmid = ext_ids.get("PubMed")

                oa_pdf = item.get("openAccessPdf") or {}
                oa_pdf_url = (
                    oa_pdf.get("url") if isinstance(oa_pdf, dict) else None
                )
                target_url = item.get("url") or (
                    f"https://www.semanticscholar.org/paper/{paper_id}"
                    if paper_id
                    else ""
                )

                record = PaperRecord(
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue,
                    citation_count=citation_count,
                    influential_citation_count=influential_count,
                    abstract=abstract,
                    abstract_source="semantic_scholar_batch" if abstract else None,
                    tldr=tldr,
                    identifiers=IdentifierSet(
                        doi=doi,
                        arxiv_id=arxiv_id,
                        pmid=pmid,
                        s2_id=paper_id,
                    ),
                    url=target_url,
                    oa_pdf_url=oa_pdf_url,
                    source_tier="L2_Academic_API",
                    resolved_via=["semantic_scholar_batch"],
                )

                # Index record by multiple identifier variants
                req_id = chunk[idx]
                enriched[req_id] = record
                enriched[req_id.lower()] = record
                if paper_id:
                    enriched[paper_id] = record
                if doi:
                    enriched[doi] = record
                    enriched[f"doi:{doi}"] = record
                    enriched[f"DOI:{doi}"] = record
                if arxiv_id:
                    enriched[arxiv_id] = record
                    enriched[f"arxiv:{arxiv_id}"] = record
                    enriched[f"ARXIV:{arxiv_id}"] = record
                if pmid:
                    enriched[pmid] = record
                    enriched[f"pmid:{pmid}"] = record
        except Exception as exc:
            logger.warning("Error in Semantic Scholar batch enrichment: %s", exc)

    return enriched


async def batch_enrich_openalex(
    client: httpx.AsyncClient,
    dois: list[str],
    settings: Settings,
) -> dict[str, PaperRecord]:
    """Batch enrich academic metadata using OpenAlex filter API."""
    cleaned_dois = list(
        dict.fromkeys(
            doi.replace("https://doi.org/", "").strip().lower()
            for doi in dois
            if doi
        )
    )
    if not cleaned_dois:
        return {}

    url = "https://api.openalex.org/works"
    enriched: dict[str, PaperRecord] = {}

    # OpenAlex filter supports pipe-separated DOI chunks (up to 40)
    chunk_size = 40
    for i in range(0, len(cleaned_dois), chunk_size):
        chunk = cleaned_dois[i : i + chunk_size]
        doi_filter = "|".join(chunk)
        params = {
            "filter": f"doi:{doi_filter}",
            "per-page": len(chunk),
            "mailto": settings.OPENALEX_EMAIL,
        }
        if settings.OPENALEX_API_KEY:
            params["api_key"] = settings.OPENALEX_API_KEY

        try:
            resp = await client.get(url, params=params, timeout=12.0)
            if resp.status_code != 200:
                logger.warning(
                    "OpenAlex batch enrichment returned status %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                continue

            data: dict[str, Any] = resp.json()
            for work in data.get("results", []):
                raw_doi = work.get("doi")
                doi = (
                    raw_doi.replace("https://doi.org/", "").strip().lower()
                    if raw_doi
                    else None
                )
                title = clean_academic_title(
                    work.get("display_name") or work.get("title") or ""
                )
                abstract = reconstruct_openalex_abstract(
                    work.get("abstract_inverted_index")
                )
                authors = [
                    a.get("author", {}).get("display_name")
                    for a in work.get("authorships", [])
                    if a.get("author", {}).get("display_name")
                ]
                year = work.get("publication_year")
                citation_count = work.get("cited_by_count")

                venue = None
                primary_loc = work.get("primary_location") or {}
                source = primary_loc.get("source") or {}
                if source.get("display_name"):
                    venue = source.get("display_name")

                oa_pdf_url = None
                oa_info = work.get("open_access") or {}
                if oa_info.get("oa_url"):
                    oa_pdf_url = oa_info.get("oa_url")
                elif work.get("best_oa_location") and work["best_oa_location"].get("pdf_url"):
                    oa_pdf_url = work["best_oa_location"].get("pdf_url")

                ids = extract_identifiers(abstract or "", work.get("id") or "")
                ids.doi = doi
                ids.openalex_id = work.get("id")

                record = PaperRecord(
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue,
                    citation_count=citation_count,
                    abstract=abstract,
                    abstract_source="openalex_batch" if abstract else None,
                    identifiers=ids,
                    url=work.get("doi") or work.get("id") or "",
                    oa_pdf_url=oa_pdf_url,
                    source_tier="L2_Academic_API",
                    resolved_via=["openalex_batch"],
                )

                if doi:
                    enriched[doi] = record
                    enriched[f"doi:{doi}"] = record
                if work.get("id"):
                    enriched[work["id"]] = record
        except Exception as exc:
            logger.warning("Error in OpenAlex batch enrichment: %s", exc)

    return enriched


async def fetch_arxiv_atom_batch(
    client: httpx.AsyncClient,
    arxiv_ids: list[str],
) -> dict[str, PaperRecord]:
    """Fetch preprint metadata from arXiv Atom API."""
    cleaned_ids = list(
        dict.fromkeys(
            re.sub(r"v\d+$", "", aid.strip())
            for aid in arxiv_ids
            if aid
        )
    )
    if not cleaned_ids:
        return {}

    url = "https://export.arxiv.org/api/query"
    id_list_str = ",".join(cleaned_ids)
    params = {
        "id_list": id_list_str,
        "max_results": len(cleaned_ids),
    }

    enriched: dict[str, PaperRecord] = {}
    try:
        resp = await client.get(url, params=params, timeout=12.0)
        if resp.status_code != 200:
            logger.warning("arXiv Atom API returned status %d", resp.status_code)
            return {}

        soup = BeautifulSoup(resp.text, "xml")
        entries = soup.find_all("entry")
        for entry in entries:
            id_tag = entry.find("id")
            if not id_tag:
                continue

            entry_id_text = id_tag.text.strip()
            # Extract raw canonical ID from entry ID URL (e.g. http://arxiv.org/abs/2403.01234v1)
            raw_match = re.search(r"arxiv\.org/abs/([^/]+)$", entry_id_text)
            matched_id = raw_match.group(1) if raw_match else entry_id_text
            canonical_id = re.sub(r"v\d+$", "", matched_id)

            title_tag = entry.find("title")
            raw_title = title_tag.text if title_tag else ""
            title = clean_academic_title(re.sub(r"\s+", " ", raw_title))

            summary_tag = entry.find("summary")
            summary = (
                re.sub(r"\s+", " ", summary_tag.text).strip()
                if summary_tag
                else None
            )

            published_tag = entry.find("published")
            year = None
            if published_tag and len(published_tag.text) >= 4:
                year_match = re.match(r"^(\d{4})", published_tag.text)
                if year_match:
                    year = int(year_match.group(1))

            authors = [
                author_tag.find("name").text.strip()
                for author_tag in entry.find_all("author")
                if author_tag.find("name")
            ]

            doi = None
            doi_tag = entry.find("doi")
            if doi_tag:
                doi = doi_tag.text.strip().lower()

            record = PaperRecord(
                title=title,
                authors=authors,
                year=year,
                venue="arXiv",
                abstract=summary,
                abstract_source="arxiv_atom" if summary else None,
                identifiers=IdentifierSet(doi=doi, arxiv_id=canonical_id),
                url=f"https://arxiv.org/abs/{canonical_id}",
                oa_pdf_url=f"https://arxiv.org/pdf/{canonical_id}.pdf",
                source_tier="L2_Academic_API",
                resolved_via=["arxiv_atom"],
            )

            enriched[canonical_id] = record
            enriched[f"arxiv:{canonical_id}"] = record
            enriched[f"ARXIV:{canonical_id}"] = record
            if doi:
                enriched[doi] = record
                enriched[f"doi:{doi}"] = record
    except Exception as exc:
        logger.warning("Error fetching arXiv Atom batch: %s", exc)

    return enriched


async def resolve_unpaywall_oa(
    client: httpx.AsyncClient,
    doi: str,
    settings: Settings,
) -> str | None:
    """Resolve direct Open Access PDF URL using Unpaywall API."""
    cleaned_doi = doi.replace("https://doi.org/", "").strip()
    if not cleaned_doi:
        return None

    url = f"https://api.unpaywall.org/v2/{cleaned_doi}"
    params = {"email": settings.UNPAYWALL_EMAIL}

    try:
        resp = await client.get(url, params=params, timeout=8.0)
        if resp.status_code != 200:
            return None

        data: dict[str, Any] = resp.json()
        best_oa = data.get("best_oa_location")
        if isinstance(best_oa, dict):
            return best_oa.get("url_for_pdf") or best_oa.get("url")

        oa_locations = data.get("oa_locations", [])
        for loc in oa_locations:
            if isinstance(loc, dict) and loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
        for loc in oa_locations:
            if isinstance(loc, dict) and loc.get("url"):
                return loc["url"]
        return None
    except Exception as exc:
        logger.warning("Unpaywall resolution failed for %s: %s", cleaned_doi, exc)
        return None

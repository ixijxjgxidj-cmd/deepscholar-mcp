"""Stage 1 — L1 Deep Search Discovery providers."""

import asyncio
import logging
import re
from typing import Any

import httpx

from deepscholar_mcp.config import Settings
from deepscholar_mcp.models import IdentifierSet, RawDiscoveryHit
from deepscholar_mcp.resolver import (
    extract_identifiers,
    reconstruct_openalex_abstract,
)

logger = logging.getLogger(__name__)


async def search_exa(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    settings: Settings,
) -> list[RawDiscoveryHit]:
    """Query Exa AI neural search engine for academic papers."""
    if not settings.has_exa:
        return []

    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": settings.EXA_API_KEY or "",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "type": "auto",
        "category": "research paper",
        "numResults": limit,
        "contents": {"text": {"maxCharacters": 1000}},
    }

    try:
        resp = await client.post(url, headers=headers, json=payload, timeout=8.0)
        if resp.status_code == 429:
            logger.warning("Exa search rate-limited (HTTP 429)")
            return []
        if resp.status_code != 200:
            logger.warning("Exa search error status %d: %s", resp.status_code, resp.text[:200])
            return []

        data: dict[str, Any] = resp.json()
        hits: list[RawDiscoveryHit] = []
        for idx, item in enumerate(data.get("results", [])):
            title = item.get("title") or ""
            target_url = item.get("url") or ""
            text = item.get("text") or ""
            published_date = item.get("publishedDate")
            pre_ids = extract_identifiers(text, target_url)

            hits.append(
                RawDiscoveryHit(
                    title=title,
                    url=target_url,
                    snippet=text,
                    source_engine="exa",
                    published_date=published_date,
                    pre_extracted_ids=pre_ids,
                    rank=idx,
                )
            )
        return hits
    except Exception as exc:
        logger.warning("Exa search failed: %s", exc)
        return []


async def search_tavily(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    settings: Settings,
) -> list[RawDiscoveryHit]:
    """Query Tavily search API with academic domain filters."""
    if not settings.has_tavily:
        return []

    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": settings.TAVILY_API_KEY,
        "query": query,
        "max_results": limit,
        "include_domains": [
            "arxiv.org",
            "openreview.net",
            "aclanthology.org",
            "ieeexplore.ieee.org",
            "dl.acm.org",
            "ncbi.nlm.nih.gov",
            "nature.com",
            "springer.com",
            "sciencedirect.com",
            "neurips.cc",
            "mlr.press",
        ],
    }

    try:
        resp = await client.post(url, headers=headers, json=payload, timeout=8.0)
        if resp.status_code == 429:
            logger.warning("Tavily search rate-limited (HTTP 429)")
            return []
        if resp.status_code != 200:
            logger.warning("Tavily search error status %d: %s", resp.status_code, resp.text[:200])
            return []

        data: dict[str, Any] = resp.json()
        hits: list[RawDiscoveryHit] = []
        for idx, item in enumerate(data.get("results", [])):
            title = item.get("title") or ""
            target_url = item.get("url") or ""
            snippet = item.get("content") or ""
            pre_ids = extract_identifiers(snippet, target_url)

            hits.append(
                RawDiscoveryHit(
                    title=title,
                    url=target_url,
                    snippet=snippet,
                    source_engine="tavily",
                    pre_extracted_ids=pre_ids,
                    rank=idx,
                )
            )
        return hits
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return []


async def search_serper_scholar(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    settings: Settings,
) -> list[RawDiscoveryHit]:
    """Query Google Scholar via Serper API."""
    if not settings.has_serper:
        return []

    url = "https://google.serper.dev/scholar"
    headers = {
        "X-API-KEY": settings.SERPER_API_KEY or "",
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": limit}

    try:
        resp = await client.post(url, headers=headers, json=payload, timeout=8.0)
        if resp.status_code == 429:
            logger.warning("Serper Scholar rate-limited (HTTP 429)")
            return []
        if resp.status_code != 200:
            logger.warning("Serper Scholar error status %d: %s", resp.status_code, resp.text[:200])
            return []

        data: dict[str, Any] = resp.json()
        hits: list[RawDiscoveryHit] = []
        for idx, item in enumerate(data.get("organic", [])):
            title = item.get("title") or ""
            link = item.get("link") or ""
            snippet = item.get("snippet") or ""

            year: int | None = None
            raw_year = item.get("year")
            if isinstance(raw_year, int):
                year = raw_year
            elif isinstance(raw_year, str) and raw_year.isdigit():
                year = int(raw_year)
            elif item.get("publicationInfo"):
                match = re.search(r"\b(19\d\d|20\d\d)\b", item["publicationInfo"])
                if match:
                    year = int(match.group(1))

            venue = item.get("publicationInfo")
            citation_count: int | None = None
            inline_links = item.get("inlineLinks", {})
            if "citedBy" in inline_links and isinstance(inline_links["citedBy"], dict):
                citation_count = inline_links["citedBy"].get("total")

            pre_ids = extract_identifiers(snippet, link)
            hits.append(
                RawDiscoveryHit(
                    title=title,
                    url=link,
                    snippet=snippet,
                    source_engine="serper_scholar",
                    year=year,
                    citation_count=citation_count,
                    venue=venue,
                    pre_extracted_ids=pre_ids,
                    rank=idx,
                )
            )
        return hits
    except Exception as exc:
        logger.warning("Serper Scholar search failed: %s", exc)
        return []


async def search_anysearch(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    settings: Settings,
) -> list[RawDiscoveryHit]:
    """Query custom AnySearch or SearXNG instance."""
    if not settings.has_anysearch or not settings.ANYSEARCH_API_URL:
        return []

    headers = {}
    if settings.ANYSEARCH_API_KEY:
        headers["Authorization"] = f"Bearer {settings.ANYSEARCH_API_KEY}"

    hits: list[RawDiscoveryHit] = []
    try:
        # First attempt GET query
        params = {"q": query, "format": "json", "limit": limit}
        resp = await client.get(
            settings.ANYSEARCH_API_URL,
            headers=headers,
            params=params,
            timeout=8.0,
        )

        # Fallback to POST on 405 Method Not Allowed
        if resp.status_code == 405:
            resp = await client.post(
                settings.ANYSEARCH_API_URL,
                headers=headers,
                json={"query": query, "limit": limit},
                timeout=8.0,
            )

        if resp.status_code == 429:
            logger.warning("AnySearch rate-limited (HTTP 429)")
            return []
        if resp.status_code != 200:
            logger.warning("AnySearch error status %d", resp.status_code)
            return []

        data: dict[str, Any] = resp.json()
        raw_items = (
            data.get("results")
            or data.get("data")
            or data.get("hits")
            or []
        )
        for idx, item in enumerate(raw_items):
            title = item.get("title") or ""
            target_url = item.get("url") or item.get("link") or ""
            snippet = item.get("content") or item.get("snippet") or ""
            pre_ids = extract_identifiers(snippet, target_url)

            hits.append(
                RawDiscoveryHit(
                    title=title,
                    url=target_url,
                    snippet=snippet,
                    source_engine="anysearch",
                    pre_extracted_ids=pre_ids,
                    rank=idx,
                )
            )
        return hits
    except Exception as exc:
        logger.warning("AnySearch failed: %s", exc)
        return []


async def search_openalex_native(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    settings: Settings,
) -> list[RawDiscoveryHit]:
    """Query OpenAlex works API (Always-on zero-key baseline)."""
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": limit,
        "mailto": settings.OPENALEX_EMAIL,
    }

    try:
        resp = await client.get(url, params=params, timeout=10.0)
        if resp.status_code == 429:
            logger.warning("OpenAlex native search rate-limited (HTTP 429)")
            return []
        if resp.status_code != 200:
            logger.warning("OpenAlex search returned status %d", resp.status_code)
            return []

        data: dict[str, Any] = resp.json()
        hits: list[RawDiscoveryHit] = []
        for idx, item in enumerate(data.get("results", [])):
            title = item.get("display_name") or item.get("title") or ""
            target_url = item.get("doi") or item.get("id") or ""
            snippet = reconstruct_openalex_abstract(item.get("abstract_inverted_index")) or ""

            authors = [
                a.get("author", {}).get("display_name")
                for a in item.get("authorships", [])
                if a.get("author", {}).get("display_name")
            ]
            year = item.get("publication_year")
            citation_count = item.get("cited_by_count")

            venue = None
            primary_loc = item.get("primary_location") or {}
            source = primary_loc.get("source") or {}
            if source.get("display_name"):
                venue = source.get("display_name")

            oa_pdf_url = None
            oa_info = item.get("open_access") or {}
            if oa_info.get("oa_url"):
                oa_pdf_url = oa_info.get("oa_url")
            elif item.get("best_oa_location") and item["best_oa_location"].get("pdf_url"):
                oa_pdf_url = item["best_oa_location"].get("pdf_url")

            pre_ids = extract_identifiers(snippet, target_url)
            if item.get("doi"):
                pre_ids.doi = (
                    item.get("doi").replace("https://doi.org/", "").strip().lower()
                )
            if item.get("id"):
                pre_ids.openalex_id = item.get("id")

            hits.append(
                RawDiscoveryHit(
                    title=title,
                    url=target_url,
                    snippet=snippet,
                    source_engine="openalex_native",
                    authors=authors,
                    year=year,
                    citation_count=citation_count,
                    venue=venue,
                    oa_pdf_url=oa_pdf_url,
                    pre_extracted_ids=pre_ids,
                    rank=idx,
                )
            )
        return hits
    except Exception as exc:
        logger.warning("OpenAlex native search failed: %s", exc)
        return []


async def search_s2_native(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    settings: Settings,
) -> list[RawDiscoveryHit]:
    """Query Semantic Scholar Graph paper search (Always-on zero-key baseline)."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,authors,year,abstract,citationCount,externalIds,openAccessPdf,url,venue",
    }
    headers = {}
    if settings.S2_API_KEY:
        headers["x-api-key"] = settings.S2_API_KEY

    try:
        resp = await client.get(url, params=params, headers=headers, timeout=10.0)
        if resp.status_code == 429:
            logger.warning("Semantic Scholar native search rate-limited (HTTP 429)")
            return []
        if resp.status_code != 200:
            logger.warning("Semantic Scholar native search returned status %d", resp.status_code)
            return []

        data: dict[str, Any] = resp.json()
        hits: list[RawDiscoveryHit] = []
        for idx, item in enumerate(data.get("data", [])):
            paper_id = item.get("paperId")
            title = item.get("title") or ""
            target_url = item.get("url") or (
                f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""
            )
            snippet = item.get("abstract") or ""

            authors = [
                a.get("name")
                for a in item.get("authors", [])
                if a.get("name")
            ]
            year = item.get("year")
            citation_count = item.get("citationCount")
            venue = item.get("venue")

            oa_pdf = item.get("openAccessPdf") or {}
            oa_pdf_url = oa_pdf.get("url")

            ext_ids = item.get("externalIds") or {}
            raw_doi = ext_ids.get("DOI")
            doi = raw_doi.strip().lower() if raw_doi else None
            arxiv_id = ext_ids.get("ArXiv")
            pmid = ext_ids.get("PubMed")

            pre_ids = IdentifierSet(
                doi=doi,
                arxiv_id=arxiv_id,
                pmid=pmid,
                s2_id=paper_id,
            )

            hits.append(
                RawDiscoveryHit(
                    title=title,
                    url=target_url,
                    snippet=snippet,
                    source_engine="semantic_scholar_native",
                    authors=authors,
                    year=year,
                    citation_count=citation_count,
                    venue=venue,
                    oa_pdf_url=oa_pdf_url,
                    pre_extracted_ids=pre_ids,
                    rank=idx,
                )
            )
        return hits
    except Exception as exc:
        logger.warning("Semantic Scholar native search failed: %s", exc)
        return []


async def run_l1_discovery(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    settings: Settings,
    timeout: float | None = None,
    provider_statuses: dict[str, str] | None = None,
) -> list[RawDiscoveryHit]:
    """Execute concurrent recall across all configured L1 discovery providers with graceful partial degradation."""
    task_map: dict[asyncio.Task[list[RawDiscoveryHit]], str] = {
        asyncio.create_task(
            search_openalex_native(client, query, limit, settings)
        ): "openalex_native",
        asyncio.create_task(
            search_s2_native(client, query, limit, settings)
        ): "semantic_scholar_native",
    }

    if settings.has_exa:
        task_map[
            asyncio.create_task(search_exa(client, query, limit, settings))
        ] = "exa"
    if settings.has_tavily:
        task_map[
            asyncio.create_task(search_tavily(client, query, limit, settings))
        ] = "tavily"
    if settings.has_serper:
        task_map[
            asyncio.create_task(
                search_serper_scholar(client, query, limit, settings)
            )
        ] = "serper_scholar"
    if settings.has_anysearch:
        task_map[
            asyncio.create_task(search_anysearch(client, query, limit, settings))
        ] = "anysearch"

    stage_timeout = timeout if timeout is not None else settings.L1_TIMEOUT_SEC
    done, pending = await asyncio.wait(
        task_map.keys(),
        timeout=stage_timeout,
    )

    combined_hits: list[RawDiscoveryHit] = []
    statuses: dict[str, str] = {}

    # 1. Harvest hits from completed providers
    for task in done:
        provider_name = task_map[task]
        try:
            if task.exception() is not None:
                exc = task.exception()
                logger.warning(
                    "Provider '%s' failed with exception: %s",
                    provider_name,
                    exc,
                )
                statuses[provider_name] = "failed"
            else:
                hits = task.result()
                statuses[provider_name] = "success"
                combined_hits.extend(hits)
        except Exception as err:
            logger.warning(
                "Error processing result for '%s': %s", provider_name, err
            )
            statuses[provider_name] = "failed"

    # 2. Handle timed-out providers by cancelling and recording status
    for task in pending:
        provider_name = task_map[task]
        statuses[provider_name] = "timeout"
        logger.warning(
            "Provider '%s' timed out after %.1f seconds; cancelling task.",
            provider_name,
            stage_timeout,
        )
        task.cancel()

    if pending:
        # Await cancelled tasks to suppress UnhandledTaskCancellation warnings
        await asyncio.gather(*pending, return_exceptions=True)

    logger.info(
        "L1 discovery finished. Statuses: %s. Preserved hits count: %d",
        statuses,
        len(combined_hits),
    )

    if provider_statuses is not None:
        provider_statuses.update(statuses)

    return combined_hits

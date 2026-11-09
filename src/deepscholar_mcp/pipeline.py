"""4-Stage async waterfall orchestrator and deduplication engine."""

import asyncio
import logging
import math
import re
import time
from typing import Any, Literal

import httpx
from rapidfuzz import fuzz

from deepscholar_mcp.cache import AsyncSQLiteCache
from deepscholar_mcp.config import Settings
from deepscholar_mcp.models import FullTextResult, PaperRecord, RawDiscoveryHit
from deepscholar_mcp.providers.l1_discovery import run_l1_discovery
from deepscholar_mcp.providers.l2_academic import (
    batch_enrich_openalex,
    batch_enrich_s2,
    fetch_arxiv_atom_batch,
    resolve_unpaywall_oa,
)
from deepscholar_mcp.providers.l3_fallback import (
    extract_abstract_section,
    fetch_and_parse_pdf,
    fetch_ar5iv_html_markdown,
    fetch_via_cloudflare_browser,
    fetch_via_exa_contents,
    fetch_via_jina_reader,
    slice_markdown_sections,
)
from deepscholar_mcp.resolver import (
    align_title_via_openalex,
    clean_academic_title,
    extract_identifiers,
)

logger = logging.getLogger(__name__)


class TimeBudget:
    """Manages end-to-end deadline and dynamically budgets stage timeouts."""

    def __init__(self, total_seconds: float = 35.0) -> None:
        self.start_time = time.monotonic()
        self.total_seconds = total_seconds
        self.deadline = self.start_time + total_seconds

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def stage_timeout(self, requested: float) -> float:
        """Allocate stage timeout ensuring at least a buffer for subsequent stages."""
        rem = self.remaining
        if rem <= 1.0:
            return 0.5
        return max(0.5, min(requested, rem - 0.5))

    @property
    def is_exhausted(self) -> bool:
        return self.remaining <= 1.5


def compute_rrf_for_paper(
    paper: PaperRecord,
    engine_ranks: dict[str, list[str]],
) -> float:
    """Compute Reciprocal Rank Fusion (RRF) score across search engines."""
    paper_keys: set[str] = set()
    if paper.identifiers.doi:
        paper_keys.add(f"doi:{paper.identifiers.doi.lower()}")
    if paper.identifiers.arxiv_id:
        paper_keys.add(f"arxiv:{paper.identifiers.arxiv_id.lower()}")
    title_norm = clean_academic_title(paper.title).lower()
    if title_norm:
        paper_keys.add(f"title:{title_norm}")

    raw_rrf = 0.0
    k_constant = 60.0

    for _engine, ranked_keys in engine_ranks.items():
        matched_rank: int | None = None
        for rank_idx, key in enumerate(ranked_keys):
            if key in paper_keys:
                matched_rank = rank_idx + 1  # 1-based rank
                break
            if key.startswith("title:") and title_norm:
                engine_title = key[6:]
                if fuzz.ratio(title_norm, engine_title) >= 90:
                    matched_rank = rank_idx + 1
                    break
        if matched_rank is not None:
            raw_rrf += 1.0 / (k_constant + matched_rank)

    # Scale RRF so a single rank-1 hit gives ~1.0
    return raw_rrf * 60.0


def compute_paper_relevance(
    paper: PaperRecord,
    query: str,
    rrf_score: float,
) -> float:
    """Compute multi-factor composite relevance score.

    Combines:
    - Title lexical similarity (35% weight)
    - Multi-engine RRF score (30% weight)
    - Source quality / verified identifier (15% weight)
    - Recency / publication freshness (10% weight)
    - Logarithmic citations as mild auxiliary signal (10% weight)
    """
    # 1. Title Similarity (0.0 - 1.0)
    norm_title = clean_academic_title(paper.title).lower()
    norm_query = query.lower().strip()
    title_sim = fuzz.token_set_ratio(norm_query, norm_title) / 100.0

    # 2. Recency Signal (0.0 - 1.0)
    current_year = 2026
    if paper.year is not None:
        recency = max(0.0, min(1.0, 1.0 - (current_year - paper.year) / 20.0))
    else:
        recency = 0.5

    # 3. Source Quality / Verification Signal (0.0 - 1.0)
    quality = 0.5
    if paper.identifiers.doi or paper.identifiers.arxiv_id:
        quality = 1.0
    elif paper.source_tier == "L2_Academic_API":
        quality = 0.85
    elif paper.venue and len(paper.venue.strip()) > 3:
        quality = 0.70

    # 4. Logarithmic Citation Signal (0.0 - 1.0) - strictly avoiding starvation of new papers
    cite_score = 0.0
    if paper.citation_count is not None and paper.citation_count > 0:
        cite_score = min(1.0, math.log1p(paper.citation_count) / math.log(10001))

    # Composite weighted formula
    composite = (
        0.35 * title_sim
        + 0.30 * min(1.0, rrf_score)
        + 0.15 * quality
        + 0.10 * recency
        + 0.10 * cite_score
    )
    return round(composite, 4)


class DeepScholarPipeline:
    """Production orchestrator executing the 4-stage waterfall literature pipeline."""

    def __init__(
        self,
        settings: Settings | None = None,
        cache: AsyncSQLiteCache | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.cache = cache or AsyncSQLiteCache(db_path=self.settings.CACHE_DB_PATH)

    async def search_literature(
        self,
        query: str,
        limit: int = 10,
        year_from: int | None = None,
        force_browser_fallback_for_missing_abstracts: bool = True,
    ) -> list[PaperRecord]:
        """Execute 4-stage literature search: Discovery -> Alignment -> Enrichment -> Fallback."""
        budget = TimeBudget(total_seconds=self.settings.TOTAL_TIMEOUT_SEC)

        async with httpx.AsyncClient(
            headers={"User-Agent": "DeepScholar-MCP/1.0"},
            follow_redirects=True,
        ) as client:
            # 1. Stage 1: L1 Deep Discovery with individual task tracking and deadline
            l1_timeout = budget.stage_timeout(self.settings.L1_TIMEOUT_SEC)
            provider_statuses: dict[str, str] = {}
            raw_hits: list[RawDiscoveryHit] = await run_l1_discovery(
                client,
                query,
                limit * 2,
                self.settings,
                timeout=l1_timeout,
                provider_statuses=provider_statuses,
            )

            # Build engine rank index for RRF
            raw_hits_by_engine: dict[str, list[str]] = {}
            for hit in raw_hits:
                eng = hit.source_engine
                if eng not in raw_hits_by_engine:
                    raw_hits_by_engine[eng] = []
                if hit.pre_extracted_ids.doi:
                    k = f"doi:{hit.pre_extracted_ids.doi.lower()}"
                elif hit.pre_extracted_ids.arxiv_id:
                    k = f"arxiv:{hit.pre_extracted_ids.arxiv_id.lower()}"
                else:
                    k = f"title:{clean_academic_title(hit.title).lower()}"
                raw_hits_by_engine[eng].append(k)

            # 2. Stage 2: Identifier Extraction
            for hit in raw_hits:
                extracted = extract_identifiers(hit.snippet, hit.url)
                if not hit.pre_extracted_ids.doi and extracted.doi:
                    hit.pre_extracted_ids.doi = extracted.doi
                if not hit.pre_extracted_ids.arxiv_id and extracted.arxiv_id:
                    hit.pre_extracted_ids.arxiv_id = extracted.arxiv_id
                if not hit.pre_extracted_ids.pmid and extracted.pmid:
                    hit.pre_extracted_ids.pmid = extracted.pmid

            # Align top unresolved hits (missing both DOI and arXiv ID) via OpenAlex title matching
            unresolved = [
                h
                for h in raw_hits
                if not h.pre_extracted_ids.doi
                and not h.pre_extracted_ids.arxiv_id
            ][:8]

            sem = asyncio.Semaphore(self.settings.MAX_CONCURRENT_REQUESTS)

            async def _align_hit(hit_item: RawDiscoveryHit) -> None:
                async with sem:
                    aligned_ids = await align_title_via_openalex(
                        client, hit_item.title, self.settings.OPENALEX_EMAIL
                    )
                    if aligned_ids.has_any():
                        if aligned_ids.doi:
                            hit_item.pre_extracted_ids.doi = aligned_ids.doi
                        if aligned_ids.arxiv_id:
                            hit_item.pre_extracted_ids.arxiv_id = (
                                aligned_ids.arxiv_id
                            )
                        if aligned_ids.pmid:
                            hit_item.pre_extracted_ids.pmid = aligned_ids.pmid
                        if aligned_ids.openalex_id:
                            hit_item.pre_extracted_ids.openalex_id = (
                                aligned_ids.openalex_id
                            )

            if unresolved and not budget.is_exhausted:
                align_timeout = budget.stage_timeout(6.0)
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *[_align_hit(h) for h in unresolved],
                            return_exceptions=True,
                        ),
                        timeout=align_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Title alignment stage timed out")

            # 3. Stage 3: Cache lookup and L2 Academic Batch Enrichment
            s2_lookups: list[str] = []
            openalex_dois: list[str] = []
            arxiv_ids: list[str] = []
            cached_records: dict[str, PaperRecord] = {}

            for hit in raw_hits:
                doi = hit.pre_extracted_ids.doi
                aid = hit.pre_extracted_ids.arxiv_id

                cache_key = None
                if doi:
                    cache_key = f"paper:doi:{doi}"
                elif aid:
                    cache_key = f"paper:arxiv:{aid}"

                if cache_key:
                    cached = await self.cache.get_json(cache_key)
                    if cached:
                        rec = PaperRecord.model_validate(cached)
                        cached_records[cache_key] = rec
                        continue

                # Prepare uncached IDs for batch enrichment
                lookup_id = hit.pre_extracted_ids.to_s2_lookup_id()
                if lookup_id:
                    s2_lookups.append(lookup_id)
                if doi:
                    openalex_dois.append(doi)
                if aid:
                    arxiv_ids.append(aid)

            # Concurrently execute L2 batch enrichments with dynamic budget
            l2_timeout = budget.stage_timeout(self.settings.L2_TIMEOUT_SEC)
            enrich_tasks = []
            if s2_lookups:
                enrich_tasks.append(
                    batch_enrich_s2(client, s2_lookups, self.settings)
                )
            else:
                enrich_tasks.append(asyncio.sleep(0, result={}))

            if openalex_dois:
                enrich_tasks.append(
                    batch_enrich_openalex(
                        client, openalex_dois, self.settings
                    )
                )
            else:
                enrich_tasks.append(asyncio.sleep(0, result={}))

            if arxiv_ids:
                enrich_tasks.append(
                    fetch_arxiv_atom_batch(client, arxiv_ids)
                )
            else:
                enrich_tasks.append(asyncio.sleep(0, result={}))

            try:
                l2_results = await asyncio.wait_for(
                    asyncio.gather(*enrich_tasks, return_exceptions=True),
                    timeout=l2_timeout,
                )
                s2_map: dict[str, PaperRecord] = (
                    l2_results[0] if isinstance(l2_results[0], dict) else {}
                )
                openalex_map: dict[str, PaperRecord] = (
                    l2_results[1] if isinstance(l2_results[1], dict) else {}
                )
                arxiv_map: dict[str, PaperRecord] = (
                    l2_results[2] if isinstance(l2_results[2], dict) else {}
                )
            except asyncio.TimeoutError:
                logger.warning("L2 enrichment stage timed out")
                s2_map, openalex_map, arxiv_map = {}, {}, {}

            # Synthesize hits into candidate PaperRecords with strict provenance separation
            consolidated: list[PaperRecord] = []
            for hit in raw_hits:
                doi = hit.pre_extracted_ids.doi
                aid = hit.pre_extracted_ids.arxiv_id

                # Check cached
                if doi and f"paper:doi:{doi}" in cached_records:
                    consolidated.append(cached_records[f"paper:doi:{doi}"])
                    continue
                if aid and f"paper:arxiv:{aid}" in cached_records:
                    consolidated.append(cached_records[f"paper:arxiv:{aid}"])
                    continue

                # Find enrichment matches
                matched_rec: PaperRecord | None = None
                lookup_id = hit.pre_extracted_ids.to_s2_lookup_id()
                if lookup_id and lookup_id in s2_map:
                    matched_rec = s2_map[lookup_id]
                elif doi and doi in s2_map:
                    matched_rec = s2_map[doi]
                elif aid and aid in s2_map:
                    matched_rec = s2_map[aid]
                elif doi and doi in openalex_map:
                    matched_rec = openalex_map[doi]
                elif aid and aid in arxiv_map:
                    matched_rec = arxiv_map[aid]

                if matched_rec:
                    # Complement with OpenAlex if abstract missing
                    if (
                        (not matched_rec.abstract or len(matched_rec.abstract) < 80)
                        and doi
                        and doi in openalex_map
                    ):
                        oa_rec = openalex_map[doi]
                        if oa_rec.abstract:
                            matched_rec.abstract = oa_rec.abstract
                            matched_rec.abstract_source = "openalex_batch"
                            matched_rec.resolved_via.extend(["openalex_batch"])
                    if not matched_rec.oa_pdf_url and doi and doi in openalex_map:
                        matched_rec.oa_pdf_url = openalex_map[doi].oa_pdf_url
                    consolidated.append(matched_rec)
                else:
                    # Fall back to L1 hit - carefully separating real abstract from search snippet
                    clean_title = clean_academic_title(hit.title)
                    is_academic_native = hit.source_engine in (
                        "openalex_native",
                        "semantic_scholar_native",
                    )

                    official_abstract = None
                    official_abstract_source = None
                    search_snip = None
                    search_snip_source = None

                    if is_academic_native and len(hit.snippet.strip()) >= 40:
                        official_abstract = hit.snippet.strip()
                        official_abstract_source = hit.source_engine
                    elif len(hit.snippet.strip()) > 0:
                        search_snip = hit.snippet.strip()
                        search_snip_source = hit.source_engine

                    consolidated.append(
                        PaperRecord(
                            title=clean_title,
                            authors=hit.authors,
                            year=hit.year,
                            venue=hit.venue,
                            citation_count=hit.citation_count,
                            abstract=official_abstract,
                            abstract_source=official_abstract_source,
                            search_snippet=search_snip,
                            search_snippet_source=search_snip_source,
                            identifiers=hit.pre_extracted_ids,
                            url=hit.url,
                            oa_pdf_url=hit.oa_pdf_url,
                            source_tier="L1_Snippet_Only",
                            resolved_via=[f"l1_{hit.source_engine}"],
                        )
                    )

            # Unpaywall Open Access resolution for DOIs missing OA PDF
            missing_oa = [
                p for p in consolidated if p.identifiers.doi and not p.oa_pdf_url
            ]
            if missing_oa and not budget.is_exhausted:
                async def _resolve_oa(p: PaperRecord) -> None:
                    async with sem:
                        if p.identifiers.doi:
                            oa_url = await resolve_unpaywall_oa(
                                client, p.identifiers.doi, self.settings
                            )
                            if oa_url:
                                p.oa_pdf_url = oa_url
                                p.resolved_via.append("unpaywall_oa")

                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *[_resolve_oa(p) for p in missing_oa],
                            return_exceptions=True,
                        ),
                        timeout=budget.stage_timeout(5.0),
                    )
                except asyncio.TimeoutError:
                    logger.warning("OA resolution stage timed out")

            # Stage 4: L3 Fallback for missing abstracts if requested and budget permits
            if force_browser_fallback_for_missing_abstracts and not budget.is_exhausted:
                missing_abstracts = [
                    p
                    for p in consolidated
                    if (not p.abstract or len(p.abstract.strip()) < 80)
                    and p.url
                    and p.url.startswith("http")
                ][:4]

                l3_timeout = budget.stage_timeout(self.settings.L3_TIMEOUT_SEC)

                async def _fetch_abstract_fallback(p: PaperRecord) -> None:
                    async with sem:
                        # Priority 1: Cloudflare Browser Rendering
                        method = "cloudflare_browser"
                        md = await fetch_via_cloudflare_browser(
                            client, p.url, self.settings
                        )
                        # Priority 2: Exa Headless Browser Page Scraping Fallback
                        if not md:
                            method = "exa_contents"
                            md = await fetch_via_exa_contents(client, p.url, self.settings)
                        # Priority 3: Jina Reader
                        if not md:
                            method = "jina_reader"
                            md = await fetch_via_jina_reader(
                                client, p.url, self.settings
                            )

                        if md:
                            # Verify if a formal abstract section can be identified
                            formal_abstract = extract_abstract_section(md)
                            if formal_abstract:
                                p.abstract = formal_abstract
                                p.abstract_source = f"{method}_abstract_section"
                            else:
                                # Safe separation: record as extracted excerpt instead of posing as official abstract
                                cleaned_md = re.sub(r"\s+", " ", md).strip()
                                p.extracted_excerpt = cleaned_md[:1500]
                                p.extracted_excerpt_source = method

                            p.source_tier = "L3_Browser_Fallback"
                            p.resolved_via.append("l3_browser_fallback")

                if missing_abstracts:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(
                                *[_fetch_abstract_fallback(p) for p in missing_abstracts],
                                return_exceptions=True,
                            ),
                            timeout=l3_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("L3 browser fallback stage timed out")

            # Calculate composite relevance scores using RRF, title match, recency, quality, citations
            for p in consolidated:
                rrf_val = compute_rrf_for_paper(p, raw_hits_by_engine)
                p.relevance_score = compute_paper_relevance(p, query, rrf_val)

            # Sort candidate papers by composite relevance score descending BEFORE deduplication
            consolidated.sort(
                key=lambda item: (item.relevance_score or 0.0), reverse=True
            )

            # Deduplication: retain the highest relevance score instance for each unique entity
            seen_dois: set[str] = set()
            seen_arxiv: set[str] = set()
            seen_titles: list[tuple[str, PaperRecord]] = []
            deduped: list[PaperRecord] = []

            for p in consolidated:
                doi = p.identifiers.doi
                aid = p.identifiers.arxiv_id
                norm_title = clean_academic_title(p.title).lower()

                if doi:
                    if doi in seen_dois:
                        continue
                    seen_dois.add(doi)
                elif aid:
                    if aid in seen_arxiv:
                        continue
                    seen_arxiv.add(aid)
                else:
                    is_dup = False
                    for existing_title, _ in seen_titles:
                        if fuzz.ratio(norm_title, existing_title) >= 90:
                            is_dup = True
                            break
                    if is_dup:
                        continue

                seen_titles.append((norm_title, p))
                deduped.append(p)

            # Apply optional minimum publication year filter
            if year_from is not None:
                deduped = [
                    p for p in deduped if p.year is None or p.year >= year_from
                ]

            # Cache enriched records
            for p in deduped:
                if p.identifiers.doi:
                    await self.cache.set_json(
                        f"paper:doi:{p.identifiers.doi}",
                        p.model_dump(),
                        ttl=self.cache.METADATA_TTL,
                    )
                if p.identifiers.arxiv_id:
                    await self.cache.set_json(
                        f"paper:arxiv:{p.identifiers.arxiv_id}",
                        p.model_dump(),
                        ttl=self.cache.METADATA_TTL,
                    )

            return deduped[:limit]

    async def fetch_paper_fulltext(
        self,
        paper_id_or_url: str,
        sections: list[str] | None = None,
        max_chars: int = 35000,
    ) -> FullTextResult:
        """Retrieve complete academic Markdown text via 4-tier fallback extraction within end-to-end deadline."""
        cache_key = f"fulltext:{paper_id_or_url.strip()}"
        cached_md = await self.cache.get_text(cache_key)

        extraction_method: Literal[
            "ar5iv_html",
            "exa_contents",
            "cloudflare_browser",
            "jina_reader",
            "pymupdf4llm_pdf",
            "failed",
        ] = "failed"
        resolved_url = paper_id_or_url
        raw_markdown: str | None = None

        if cached_md:
            raw_markdown = cached_md
            extraction_method = "jina_reader"  # general marker for cached content
        else:
            budget = TimeBudget(total_seconds=self.settings.TOTAL_TIMEOUT_SEC)
            ids = extract_identifiers(paper_id_or_url, paper_id_or_url)

            async def _execute_waterfall(client: httpx.AsyncClient) -> tuple[str | None, str, str]:
                target_url = paper_id_or_url
                if not target_url.startswith("http"):
                    if ids.arxiv_id:
                        target_url = f"https://arxiv.org/abs/{ids.arxiv_id}"
                    elif ids.doi:
                        target_url = f"https://doi.org/{ids.doi}"

                md_res = None
                method_res = "failed"
                url_res = target_url

                # Priority 1: arXiv HTML rewriting & parsing
                if ids.arxiv_id:
                    md = await fetch_ar5iv_html_markdown(client, ids.arxiv_id)
                    if md:
                        return md, "ar5iv_html", f"https://ar5iv.labs.arxiv.org/html/{ids.arxiv_id}"
                    else:
                        ar5iv_url = f"https://ar5iv.labs.arxiv.org/html/{ids.arxiv_id}"
                        # Priority 1b: Cloudflare browser rendering first
                        md = await fetch_via_cloudflare_browser(client, ar5iv_url, self.settings)
                        if md:
                            return md, "cloudflare_browser", ar5iv_url
                        # Priority 1c: Exa headless browser page scraping fallback
                        md = await fetch_via_exa_contents(client, ar5iv_url, self.settings)
                        if md:
                            return md, "exa_contents", ar5iv_url
                        # Priority 1d: Jina reader
                        md = await fetch_via_jina_reader(client, ar5iv_url, self.settings)
                        if md:
                            return md, "jina_reader", ar5iv_url

                # Priority 2: Unpaywall Open Access resolution for DOIs
                if ids.doi and not paper_id_or_url.startswith("http"):
                    oa_url = await resolve_unpaywall_oa(client, ids.doi, self.settings)
                    if oa_url:
                        target_url = oa_url
                        url_res = target_url

                # Priority 3: Direct PDF parsing
                if (
                    target_url.lower().endswith(".pdf")
                    or "arxiv.org/pdf" in target_url.lower()
                ):
                    md = await fetch_and_parse_pdf(client, target_url)
                    if md:
                        return md, "pymupdf4llm_pdf", target_url

                # Priority 4: Web / SPA Waterfall (Cloudflare Browser -> Exa Browser -> Jina -> PDF)
                md = await fetch_via_cloudflare_browser(client, target_url, self.settings)
                if md:
                    return md, "cloudflare_browser", target_url

                md = await fetch_via_exa_contents(client, target_url, self.settings)
                if md:
                    return md, "exa_contents", target_url

                md = await fetch_via_jina_reader(client, target_url, self.settings)
                if md:
                    return md, "jina_reader", target_url

                md = await fetch_and_parse_pdf(client, target_url)
                if md:
                    return md, "pymupdf4llm_pdf", target_url

                return md_res, method_res, url_res

            async with httpx.AsyncClient(
                headers={"User-Agent": "DeepScholar-MCP/1.0"},
                follow_redirects=True,
            ) as client:
                try:
                    raw_markdown, extraction_method, resolved_url = await asyncio.wait_for(
                        _execute_waterfall(client),
                        timeout=budget.remaining,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Full-text extraction timed out after %.1f seconds", self.settings.TOTAL_TIMEOUT_SEC)
                    raw_markdown = None
                    extraction_method = "failed"

            if raw_markdown:
                await self.cache.set_text(
                    cache_key, raw_markdown, ttl=self.cache.FULLTEXT_TTL
                )
            else:
                raw_markdown = (
                    f"# Full-Text Retrieval Failed\n\nCould not extract full text for: {paper_id_or_url}"
                )
                extraction_method = "failed"

        # Apply section slicing and length constraints
        available_sections, sliced_content, truncated = slice_markdown_sections(
            raw_markdown, sections, max_chars=max_chars
        )

        return FullTextResult(
            paper_id_or_url=paper_id_or_url,
            resolved_url=resolved_url,
            extraction_method=extraction_method,
            available_sections=available_sections,
            markdown_content=sliced_content,
            truncated=truncated,
        )

    async def expand_citation_graph(
        self,
        paper_id: str,
        direction: Literal["citations", "references", "both"] = "both",
        limit: int = 15,
    ) -> dict[str, Any]:
        """Expand citation and reference graph via Semantic Scholar Graph API with OpenAlex fallback."""
        clean_id = paper_id.strip()
        # Canonicalize lookup prefix for Semantic Scholar
        s2_paper_id = clean_id
        if clean_id.startswith("10.") and "/" in clean_id:
            s2_paper_id = f"DOI:{clean_id}"
        elif re.match(r"^\d{4}\.\d{4,5}$", clean_id):
            s2_paper_id = f"ARXIV:{clean_id}"

        async with httpx.AsyncClient(
            headers={"User-Agent": "DeepScholar-MCP/1.0"},
            follow_redirects=True,
        ) as client:
            headers: dict[str, str] = {}
            if self.settings.S2_API_KEY:
                headers["x-api-key"] = self.settings.S2_API_KEY

            fields = (
                "title,authors,year,venue,citationCount,influentialCitationCount,"
                "externalIds,openAccessPdf,url,abstract"
            )

            citations_list: list[dict[str, Any]] = []
            references_list: list[dict[str, Any]] = []
            s2_failed = False

            try:
                if direction in ("citations", "both"):
                    cite_url = (
                        f"https://api.semanticscholar.org/graph/v1/paper/{s2_paper_id}/citations"
                    )
                    resp = await client.get(
                        cite_url,
                        headers=headers,
                        params={"fields": fields, "limit": limit * 2},
                        timeout=12.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("data", []):
                            cp = item.get("citingPaper")
                            if cp and cp.get("title"):
                                citations_list.append(cp)
                    else:
                        logger.warning(
                            "Semantic Scholar citations failed with status %d", resp.status_code
                        )
                        s2_failed = True

                if direction in ("references", "both") and not s2_failed:
                    ref_url = (
                        f"https://api.semanticscholar.org/graph/v1/paper/{s2_paper_id}/references"
                    )
                    resp = await client.get(
                        ref_url,
                        headers=headers,
                        params={"fields": fields, "limit": limit * 2},
                        timeout=12.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("data", []):
                            cp = item.get("citedPaper")
                            if cp and cp.get("title"):
                                references_list.append(cp)
                    else:
                        logger.warning(
                            "Semantic Scholar references failed with status %d", resp.status_code
                        )
                        s2_failed = True
            except Exception as exc:
                logger.warning("Semantic Scholar citation query error: %s", exc)
                s2_failed = True

            # If Semantic Scholar fails or rate-limits, execute OpenAlex fallback
            if s2_failed:
                logger.info("Executing OpenAlex fallback for citation graph of %s", clean_id)
                citations_list = []
                references_list = []

                # Find OpenAlex work ID
                openalex_work_id = None
                raw_doi = clean_id.replace("DOI:", "").strip() if "DOI:" in clean_id else None
                if not raw_doi and clean_id.startswith("10."):
                    raw_doi = clean_id

                if raw_doi:
                    oa_work_url = f"https://api.openalex.org/works/https://doi.org/{raw_doi}"
                    try:
                        work_resp = await client.get(
                            oa_work_url,
                            params={"mailto": self.settings.OPENALEX_EMAIL},
                            timeout=8.0,
                        )
                        if work_resp.status_code == 200:
                            openalex_work_id = work_resp.json().get("id")
                    except Exception as exc:
                        logger.warning(
                            "Could not resolve OpenAlex work ID for DOI %s: %s",
                            raw_doi,
                            exc,
                        )

                if not openalex_work_id:
                    # Query by ID directly if already OpenAlex ID or search
                    openalex_work_id = clean_id

                # Query OpenAlex cites / cited_by
                if openalex_work_id:
                    try:
                        if direction in ("citations", "both"):
                            cites_resp = await client.get(
                                "https://api.openalex.org/works",
                                params={
                                    "filter": f"cites:{openalex_work_id}",
                                    "per-page": limit * 2,
                                    "mailto": self.settings.OPENALEX_EMAIL,
                                },
                                timeout=10.0,
                            )
                            if cites_resp.status_code == 200:
                                for w in cites_resp.json().get("results", []):
                                    citations_list.append({
                                        "title": w.get("display_name") or w.get("title"),
                                        "year": w.get("publication_year"),
                                        "citationCount": w.get("cited_by_count"),
                                        "influentialCitationCount": 0,
                                        "url": w.get("doi") or w.get("id"),
                                    })

                        if direction in ("references", "both"):
                            refs_resp = await client.get(
                                "https://api.openalex.org/works",
                                params={
                                    "filter": f"cited_by:{openalex_work_id}",
                                    "per-page": limit * 2,
                                    "mailto": self.settings.OPENALEX_EMAIL,
                                },
                                timeout=10.0,
                            )
                            if refs_resp.status_code == 200:
                                for w in refs_resp.json().get("results", []):
                                    references_list.append({
                                        "title": w.get("display_name") or w.get("title"),
                                        "year": w.get("publication_year"),
                                        "citationCount": w.get("cited_by_count"),
                                        "influentialCitationCount": 0,
                                        "url": w.get("doi") or w.get("id"),
                                    })
                    except Exception as oa_exc:
                        logger.warning("OpenAlex citation fallback failed: %s", oa_exc)

            # Sort papers by (influentialCitationCount desc, citationCount desc)
            def _sort_key(item: dict[str, Any]) -> tuple[int, int]:
                return (
                    item.get("influentialCitationCount") or 0,
                    item.get("citationCount") or 0,
                )

            citations_list.sort(key=_sort_key, reverse=True)
            references_list.sort(key=_sort_key, reverse=True)

            citations_list = citations_list[:limit]
            references_list = references_list[:limit]

            if direction == "citations":
                return {
                    "paper_id": clean_id,
                    "direction": "citations",
                    "count": len(citations_list),
                    "results": citations_list,
                }
            elif direction == "references":
                return {
                    "paper_id": clean_id,
                    "direction": "references",
                    "count": len(references_list),
                    "results": references_list,
                }
            else:
                combined_results = citations_list + references_list
                combined_results.sort(key=_sort_key, reverse=True)
                return {
                    "paper_id": clean_id,
                    "direction": "both",
                    "citations": citations_list,
                    "references": references_list,
                    "count": len(combined_results),
                    "results": combined_results[:limit],
                }

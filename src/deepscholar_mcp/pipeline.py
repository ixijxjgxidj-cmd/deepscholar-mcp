"""4-Stage async waterfall orchestrator and deduplication engine."""

import asyncio
import logging
import re
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
        async with httpx.AsyncClient(
            headers={"User-Agent": "DeepScholar-MCP/1.0"},
            follow_redirects=True,
        ) as client:
            # 1. Stage 1: L1 Deep Discovery
            raw_hits: list[RawDiscoveryHit] = await run_l1_discovery(
                client, query, limit * 2, self.settings
            )

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
                h for h in raw_hits
                if not h.pre_extracted_ids.doi and not h.pre_extracted_ids.arxiv_id
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
                            hit_item.pre_extracted_ids.arxiv_id = aligned_ids.arxiv_id
                        if aligned_ids.pmid:
                            hit_item.pre_extracted_ids.pmid = aligned_ids.pmid
                        if aligned_ids.openalex_id:
                            hit_item.pre_extracted_ids.openalex_id = aligned_ids.openalex_id

            if unresolved:
                await asyncio.gather(*[_align_hit(h) for h in unresolved], return_exceptions=True)

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

            # Concurrently execute L2 batch enrichments
            enrich_tasks = []
            if s2_lookups:
                enrich_tasks.append(batch_enrich_s2(client, s2_lookups, self.settings))
            else:
                enrich_tasks.append(asyncio.sleep(0, result={}))

            if openalex_dois:
                enrich_tasks.append(batch_enrich_openalex(client, openalex_dois, self.settings))
            else:
                enrich_tasks.append(asyncio.sleep(0, result={}))

            if arxiv_ids:
                enrich_tasks.append(fetch_arxiv_atom_batch(client, arxiv_ids))
            else:
                enrich_tasks.append(asyncio.sleep(0, result={}))

            try:
                l2_results = await asyncio.wait_for(
                    asyncio.gather(*enrich_tasks, return_exceptions=True),
                    timeout=self.settings.L2_TIMEOUT_SEC,
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

            # Synthesize hits into candidate PaperRecords
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
                    # Merge any extra metadata from OpenAlex or arXiv if complementary
                    if (not matched_rec.abstract or len(matched_rec.abstract) < 80) and doi and doi in openalex_map:
                        oa_rec = openalex_map[doi]
                        if oa_rec.abstract:
                            matched_rec.abstract = oa_rec.abstract
                            matched_rec.resolved_via.extend(["openalex_batch"])
                    if not matched_rec.oa_pdf_url and doi and doi in openalex_map:
                        matched_rec.oa_pdf_url = openalex_map[doi].oa_pdf_url
                    consolidated.append(matched_rec)
                else:
                    # Fall back to L1 snippet
                    clean_title = clean_academic_title(hit.title)
                    consolidated.append(
                        PaperRecord(
                            title=clean_title,
                            authors=hit.authors,
                            year=hit.year,
                            venue=hit.venue,
                            citation_count=hit.citation_count,
                            abstract=hit.snippet if len(hit.snippet) >= 40 else None,
                            identifiers=hit.pre_extracted_ids,
                            url=hit.url,
                            oa_pdf_url=hit.oa_pdf_url,
                            source_tier="L1_Snippet_Only",
                            resolved_via=[f"l1_{hit.source_engine}"],
                        )
                    )

            # Unpaywall Open Access resolution for DOIs missing OA PDF
            missing_oa = [p for p in consolidated if p.identifiers.doi and not p.oa_pdf_url]
            if missing_oa:
                async def _resolve_oa(p: PaperRecord) -> None:
                    async with sem:
                        if p.identifiers.doi:
                            oa_url = await resolve_unpaywall_oa(client, p.identifiers.doi, self.settings)
                            if oa_url:
                                p.oa_pdf_url = oa_url
                                p.resolved_via.append("unpaywall_oa")

                await asyncio.gather(*[_resolve_oa(p) for p in missing_oa], return_exceptions=True)

            # Stage 4: L3 Fallback for missing abstracts if requested
            if force_browser_fallback_for_missing_abstracts:
                missing_abstracts = [
                    p for p in consolidated
                    if (not p.abstract or len(p.abstract.strip()) < 80)
                    and p.url
                    and p.url.startswith("http")
                ][:4]

                async def _fetch_abstract_fallback(p: PaperRecord) -> None:
                    async with sem:
                        md = await fetch_via_exa_contents(client, p.url, self.settings)
                        if not md:
                            md = await fetch_via_cloudflare_browser(client, p.url, self.settings)
                        if not md:
                            md = await fetch_via_jina_reader(client, p.url, self.settings)

                        if md:
                            # Clean and take first 1500 chars as abstract
                            cleaned_md = re.sub(r"\s+", " ", md).strip()
                            p.abstract = cleaned_md[:1500]
                            p.source_tier = "L3_Browser_Fallback"
                            p.resolved_via.append("l3_browser_fallback")

                if missing_abstracts:
                    await asyncio.gather(
                        *[_fetch_abstract_fallback(p) for p in missing_abstracts],
                        return_exceptions=True,
                    )

            # 8. Deduplication by canonical DOI -> arXiv ID -> Normalized Title
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
                    # Check fuzzy title similarity against already accepted records
                    is_dup = False
                    for existing_title, _ in seen_titles:
                        if fuzz.ratio(norm_title, existing_title) >= 90:
                            is_dup = True
                            break
                    if is_dup:
                        continue

                seen_titles.append((norm_title, p))
                deduped.append(p)

            # 9. Apply year_from filter
            if year_from is not None:
                deduped = [
                    p for p in deduped
                    if p.year is None or p.year >= year_from
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
        """Retrieve complete academic Markdown text via 4-tier fallback extraction."""
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
            ids = extract_identifiers(paper_id_or_url, paper_id_or_url)
            async with httpx.AsyncClient(
                headers={"User-Agent": "DeepScholar-MCP/1.0"},
                follow_redirects=True,
            ) as client:
                target_url = paper_id_or_url
                if not target_url.startswith("http"):
                    if ids.arxiv_id:
                        target_url = f"https://arxiv.org/abs/{ids.arxiv_id}"
                    elif ids.doi:
                        target_url = f"https://doi.org/{ids.doi}"

                # Priority 1: arXiv HTML rewriting & parsing
                if ids.arxiv_id:
                    md = await fetch_ar5iv_html_markdown(client, ids.arxiv_id)
                    if md:
                        raw_markdown = md
                        extraction_method = "ar5iv_html"
                        resolved_url = f"https://ar5iv.labs.arxiv.org/html/{ids.arxiv_id}"
                    else:
                        ar5iv_url = f"https://ar5iv.labs.arxiv.org/html/{ids.arxiv_id}"
                        md = await fetch_via_cloudflare_browser(client, ar5iv_url, self.settings)
                        if md:
                            raw_markdown = md
                            extraction_method = "cloudflare_browser"
                            resolved_url = ar5iv_url
                        else:
                            md = await fetch_via_jina_reader(client, ar5iv_url, self.settings)
                            if md:
                                raw_markdown = md
                                extraction_method = "jina_reader"
                                resolved_url = ar5iv_url

                # Priority 2: Unpaywall Open Access resolution for DOIs
                if not raw_markdown and ids.doi and not paper_id_or_url.startswith("http"):
                    oa_url = await resolve_unpaywall_oa(client, ids.doi, self.settings)
                    if oa_url:
                        target_url = oa_url

                # Priority 3: Direct PDF parsing
                if not raw_markdown and (
                    target_url.lower().endswith(".pdf")
                    or "arxiv.org/pdf" in target_url.lower()
                ):
                    md = await fetch_and_parse_pdf(client, target_url)
                    if md:
                        raw_markdown = md
                        extraction_method = "pymupdf4llm_pdf"
                        resolved_url = target_url

                # Priority 4: Web / SPA Waterfall (Exa -> Cloudflare -> Jina -> PDF)
                if not raw_markdown:
                    md = await fetch_via_exa_contents(client, target_url, self.settings)
                    if md:
                        raw_markdown = md
                        extraction_method = "exa_contents"
                        resolved_url = target_url
                    else:
                        md = await fetch_via_cloudflare_browser(client, target_url, self.settings)
                        if md:
                            raw_markdown = md
                            extraction_method = "cloudflare_browser"
                            resolved_url = target_url
                        else:
                            md = await fetch_via_jina_reader(client, target_url, self.settings)
                            if md:
                                raw_markdown = md
                                extraction_method = "jina_reader"
                                resolved_url = target_url
                            else:
                                # Final attempt: try parsing as PDF stream
                                md = await fetch_and_parse_pdf(client, target_url)
                                if md:
                                    raw_markdown = md
                                    extraction_method = "pymupdf4llm_pdf"
                                    resolved_url = target_url

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

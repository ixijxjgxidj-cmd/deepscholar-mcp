"""Live End-to-End Verification Script for DeepScholar-MCP Document Retrieval Workflow.

Tests all key fetching/scraping tiers and integrated discovery -> fulltext pipeline.
"""

import asyncio
import json
import logging
import sqlite3
import sys
import time
from typing import Any

# Ensure UTF-8 stdout encoding on Windows and unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from deepscholar_mcp.config import Settings
from deepscholar_mcp.pipeline import DeepScholarPipeline
from deepscholar_mcp.providers.l3_fallback import fetch_via_exa_contents
import httpx


async def run_tier_1_ar5iv(pipeline: DeepScholarPipeline) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print(">>> [Tier 1] ar5iv / arXiv HTML parsing & Section Slicing")
    print("=" * 60)
    
    paper_id = "1706.03762"
    sections_filter = ["abstract", "introduction"]
    
    t0 = time.perf_counter()
    res = await pipeline.fetch_paper_fulltext(
        paper_id_or_url=paper_id,
        sections=sections_filter,
        max_chars=20000,
    )
    latency = time.perf_counter() - t0
    
    has_latex = "$" in res.markdown_content
    passed = (
        res.extraction_method == "ar5iv_html"
        and len(res.markdown_content) > 300
        and has_latex
    )
    
    print(f"Paper ID: {paper_id}")
    print(f"Extraction Method: {res.extraction_method}")
    print(f"Resolved URL: {res.resolved_url}")
    print(f"Latency: {latency:.2f}s")
    print(f"Character Count: {len(res.markdown_content)}")
    print(f"LaTeX Preservation (contains $): {has_latex}")
    print(f"Available Sections Discovered ({len(res.available_sections)}): {res.available_sections[:8]}")
    print(f"Snippet Preview:\n{res.markdown_content[:300]}...")
    print(f"Tier 1 Status: {'PASS' if passed else 'FAIL'}")
    
    return {
        "tier": "Tier 1: ar5iv / arXiv HTML Parsing",
        "passed": passed,
        "extraction_method": res.extraction_method,
        "expected_method": "ar5iv_html",
        "latency_sec": round(latency, 2),
        "char_count": len(res.markdown_content),
        "has_latex": has_latex,
        "available_sections_count": len(res.available_sections),
        "sections_sample": res.available_sections[:8],
        "snippet": res.markdown_content[:250],
    }


async def run_tier_2_cloudflare(pipeline: DeepScholarPipeline) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print(">>> [Tier 2] Cloudflare Browser Rendering API (/browser-rendering/markdown)")
    print("=" * 60)
    
    target_url = f"https://example.com?t=nocache_cf_{int(time.time())}"
    
    t0 = time.perf_counter()
    res = await pipeline.fetch_paper_fulltext(
        paper_id_or_url=target_url,
        max_chars=20000,
    )
    latency = time.perf_counter() - t0
    
    passed = (
        res.extraction_method == "cloudflare_browser"
        and len(res.markdown_content) > 50
    )
    
    print(f"Target URL: {target_url}")
    print(f"Extraction Method: {res.extraction_method}")
    print(f"Latency: {latency:.2f}s")
    print(f"Character Count: {len(res.markdown_content)}")
    print(f"Snippet Preview:\n{res.markdown_content[:300]}...")
    print(f"Tier 2 Status: {'PASS' if passed else 'FAIL'}")
    
    return {
        "tier": "Tier 2: Cloudflare Browser Rendering",
        "passed": passed,
        "extraction_method": res.extraction_method,
        "expected_method": "cloudflare_browser",
        "latency_sec": round(latency, 2),
        "char_count": len(res.markdown_content),
        "snippet": res.markdown_content[:250],
    }


async def run_tier_3_exa(base_settings: Settings) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print(">>> [Tier 3] Exa Livecrawl (/contents endpoint)")
    print("=" * 60)
    
    # Configure pipeline settings with Cloudflare disabled to test Exa fallback in waterfall
    exa_settings = Settings()
    exa_settings.CF_API_TOKEN = None
    exa_settings.CF_ACCOUNT_ID = None
    exa_pipeline = DeepScholarPipeline(exa_settings)
    
    target_url = f"https://example.com?t=nocache_exa_{int(time.time())}"
    
    t0 = time.perf_counter()
    res = await exa_pipeline.fetch_paper_fulltext(
        paper_id_or_url=target_url,
        max_chars=20000,
    )
    latency = time.perf_counter() - t0
    
    passed = (
        res.extraction_method == "exa_contents"
        and len(res.markdown_content) > 50
    )
    
    print(f"Target URL: {target_url}")
    print(f"Extraction Method: {res.extraction_method}")
    print(f"Latency: {latency:.2f}s")
    print(f"Character Count: {len(res.markdown_content)}")
    print(f"Snippet Preview:\n{res.markdown_content[:300]}...")
    print(f"Tier 3 Status: {'PASS' if passed else 'FAIL'}")
    
    return {
        "tier": "Tier 3: Exa Livecrawl",
        "passed": passed,
        "extraction_method": res.extraction_method,
        "expected_method": "exa_contents",
        "latency_sec": round(latency, 2),
        "char_count": len(res.markdown_content),
        "snippet": res.markdown_content[:250],
    }


async def run_tier_4_pymupdf(pipeline: DeepScholarPipeline) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print(">>> [Tier 4] PyMuPDF4LLM Local Streaming PDF Parsing")
    print("=" * 60)
    
    pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
    
    t0 = time.perf_counter()
    res = await pipeline.fetch_paper_fulltext(
        paper_id_or_url=pdf_url,
        max_chars=35000,
    )
    latency = time.perf_counter() - t0
    
    passed = (
        res.extraction_method == "pymupdf4llm_pdf"
        and len(res.markdown_content) > 1000
    )
    
    print(f"PDF URL: {pdf_url}")
    print(f"Extraction Method: {res.extraction_method}")
    print(f"Latency: {latency:.2f}s")
    print(f"Character Count: {len(res.markdown_content)}")
    print(f"Available Sections Discovered ({len(res.available_sections)}): {res.available_sections[:8]}")
    print(f"Snippet Preview:\n{res.markdown_content[:300]}...")
    print(f"Tier 4 Status: {'PASS' if passed else 'FAIL'}")
    
    return {
        "tier": "Tier 4: PyMuPDF4LLM Local PDF Parsing",
        "passed": passed,
        "extraction_method": res.extraction_method,
        "expected_method": "pymupdf4llm_pdf",
        "latency_sec": round(latency, 2),
        "char_count": len(res.markdown_content),
        "available_sections_count": len(res.available_sections),
        "sections_sample": res.available_sections[:8],
        "snippet": res.markdown_content[:250],
    }


async def run_integrated_discovery_and_graph(pipeline: DeepScholarPipeline) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print(">>> [Integrated Discovery -> Fulltext -> Citation Graph]")
    print("=" * 60)
    
    query = "Transformer self-attention"
    print(f"1. Searching literature for query: '{query}'...")
    t0 = time.perf_counter()
    papers = await pipeline.search_literature(query=query, limit=5)
    t_search = time.perf_counter() - t0
    
    if not papers:
        print("FAIL: No papers discovered")
        return {"tier": "Integrated Workflow", "passed": False, "error": "No papers discovered"}
    
    top_paper = papers[0]
    print(f"Discovered {len(papers)} papers in {t_search:.2f}s.")
    print(f"Top Paper: '{top_paper.title}' ({top_paper.year})")
    print(f"  Authors: {', '.join(top_paper.authors[:3])} et al.")
    print(f"  Venue: {top_paper.venue}")
    print(f"  Citations: {top_paper.citation_count}")
    print(f"  Identifiers: DOI={top_paper.identifiers.doi}, arXiv={top_paper.identifiers.arxiv_id}")
    
    # Target paper id for fulltext
    target_fetch_id = (
        top_paper.identifiers.arxiv_id
        or top_paper.identifiers.doi
        or (top_paper.urls[0] if top_paper.urls else "1706.03762")
    )
    print(f"\n2. Fetching structured fulltext for: '{target_fetch_id}'...")
    t1 = time.perf_counter()
    ft_res = await pipeline.fetch_paper_fulltext(
        paper_id_or_url=target_fetch_id,
        sections=["abstract", "introduction", "architecture", "attention"],
        max_chars=25000,
    )
    t_fulltext = time.perf_counter() - t1
    print(f"Fulltext retrieved via: {ft_res.extraction_method} in {t_fulltext:.2f}s ({len(ft_res.markdown_content)} chars)")
    print(f"Available sections: {ft_res.available_sections[:6]}")
    
    # Citation Graph Expansion
    target_graph_id = (
        top_paper.identifiers.doi
        or top_paper.identifiers.arxiv_id
        or "1706.03762"
    )
    print(f"\n3. Expanding citation graph for: '{target_graph_id}'...")
    t2 = time.perf_counter()
    cg_res = await pipeline.expand_citation_graph(
        paper_id=target_graph_id,
        direction="both",
        limit=5,
    )
    t_graph = time.perf_counter() - t2
    
    cites = cg_res.get("citations", [])
    refs = cg_res.get("references", [])
    print(f"Citation Graph in {t_graph:.2f}s: {len(cites)} citations, {len(refs)} references")
    if cites:
        print(f"  Sample Citation: {cites[0].get('title')} ({cites[0].get('year')})")
    if refs:
        print(f"  Sample Reference: {refs[0].get('title')} ({refs[0].get('year')})")
        
    passed = (
        len(papers) > 0
        and len(ft_res.markdown_content) > 300
        and (len(cites) > 0 or len(refs) > 0)
    )
    print(f"Integrated Workflow Status: {'PASS' if passed else 'FAIL'}")
    
    return {
        "tier": "Integrated Discovery -> Fulltext -> Citation Graph",
        "passed": passed,
        "search_latency_sec": round(t_search, 2),
        "fulltext_latency_sec": round(t_fulltext, 2),
        "graph_latency_sec": round(t_graph, 2),
        "total_latency_sec": round(t_search + t_fulltext + t_graph, 2),
        "discovered_papers_count": len(papers),
        "top_paper": {
            "title": top_paper.title,
            "year": top_paper.year,
            "venue": top_paper.venue,
            "citation_count": top_paper.citation_count,
            "doi": top_paper.identifiers.doi,
            "arxiv_id": top_paper.identifiers.arxiv_id,
        },
        "fulltext_extraction_method": ft_res.extraction_method,
        "fulltext_chars": len(ft_res.markdown_content),
        "citations_count": len(cites),
        "references_count": len(refs),
    }


async def main() -> None:
    settings = Settings()
    pipeline = DeepScholarPipeline(settings)
    
    # Clear test-related keys from cache for clean benchmark
    try:
        with sqlite3.connect(settings.CACHE_DB_PATH) as conn:
            conn.execute("DELETE FROM cache_entries WHERE cache_key LIKE 'fulltext:%' OR cache_key LIKE 'query:%'")
            conn.commit()
        print("SQLite cache cleared for live test run.", flush=True)
    except Exception as exc:
        print(f"Cache clear skipped: {exc}", flush=True)
    
    report: dict[str, Any] = {"tiers": []}
    
    # 1. Tier 1: ar5iv
    tier1 = await run_tier_1_ar5iv(pipeline)
    report["tiers"].append(tier1)
    
    # 2. Tier 2: Cloudflare Browser Rendering
    tier2 = await run_tier_2_cloudflare(pipeline)
    report["tiers"].append(tier2)
    
    # 3. Tier 3: Exa Livecrawl
    tier3 = await run_tier_3_exa(settings)
    report["tiers"].append(tier3)
    
    # 4. Tier 4: PyMuPDF4LLM local PDF parsing
    tier4 = await run_tier_4_pymupdf(pipeline)
    report["tiers"].append(tier4)
    
    # 5. Integrated Workflow
    integrated = await run_integrated_discovery_and_graph(pipeline)
    report["tiers"].append(integrated)
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY REPORT")
    print("=" * 60)
    all_passed = all(t["passed"] for t in report["tiers"])
    print(f"Overall Result: {'ALL TESTS PASSED (5/5)' if all_passed else 'SOME TESTS FAILED'}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Save report to json file
    with open("retrieval_verification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nDetailed JSON saved to retrieval_verification_report.json")


if __name__ == "__main__":
    asyncio.run(main())

# DeepScholar-MCP

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Protocol](https://img.shields.io/badge/MCP-1.2.0+-green.svg)](https://modelcontextprotocol.io/)

Production-grade Model Context Protocol (MCP) server for deep academic literature retrieval, metadata enrichment, citation graph navigation, and full-text Markdown extraction.

---

## 1. Architecture Overview

DeepScholar-MCP implements a **4-Stage Concurrent Waterfall Architecture** designed for speed, resilience, and high recall:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Stage 1: L1 Deep Discovery                      │
│   Concurrent multi-engine recall across AI search & native databases:  │
│   • Exa AI • Tavily • Serper Scholar • AnySearch / SearXNG             │
│   • OpenAlex (Native) • Semantic Scholar (Native)                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              Stage 2: Identifier Extraction & Fuzzy Alignment          │
│   • Regex extraction for DOIs, canonical arXiv IDs, and PMIDs          │
│   • RapidFuzz token sorting (>= 85% match) via OpenAlex title lookup   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              Stage 3: L2 Academic Batch Enrichment & OA                │
│   • Semantic Scholar Batch API (externalIds, TLDR, citations)          │
│   • OpenAlex Filter API batching (abstract reconstruction, venues)     │
│   • arXiv Atom XML batch queries (preprints, direct PDF links)         │
│   • Unpaywall Open Access PDF resolution                               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Stage 4: L3 Browser & Full-Text Waterfall                │
│   • Priority 1: ar5iv HTML to clean Markdown rewriting                 │
│   • Priority 2: Unpaywall / OpenAlex Open Access PDF discovery         │
│   • Priority 3: Local PyMuPDF4LLM streaming PDF parsing                │
│   • Priority 4: Cloudflare Browser Rendering & Jina Reader fallback    │
│   • Markdown section filtering & token budget slicing                  │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Principles
- **Zero-Key Cold-Start Baseline:** Out of the box with zero API keys configured, DeepScholar-MCP provides full academic search and full-text retrieval using **OpenAlex + Semantic Scholar Public + arXiv API + Unpaywall + ar5iv HTML + Jina Reader + PyMuPDF4LLM**.
- **Abstract Credibility & Strict Provenance:** Scientifically separates official `abstract` (from verified academic APIs or explicit `## Abstract` sections) from `search_snippet` (search engine snippets) and `extracted_excerpt` (raw web scrape excerpts), preventing AI agents from confusing forum text or navigation bars with author abstracts.
- **Resilient Multi-Source Timeout & Result Preservation:** Each discovery provider runs as an independent task. If a slow provider times out, all completed providers' results are fully preserved, the slow task is cancelled, and provider statuses (`success`, `timeout`, `rate_limited`, `failed`) are recorded.
- **Unified 35-Second End-to-End Budget (`TOTAL_TIMEOUT_SEC`):** Enforces a global deadline across the entire waterfall pipeline, dynamically distributing remaining time budgets to L1, L2, and L3 stages.
- **Multi-Factor Relevance Ranking with RRF:** Eliminates engine-order bias by computing Reciprocal Rank Fusion (RRF) across engines, combined with RapidFuzz title similarity, publication recency, and verified identifier quality. Logarithmic citations serve as an auxiliary signal, ensuring fresh papers are never starved.
- **Persistent Caching:** Built-in `AsyncSQLiteCache` caches metadata for 7 days and full text for 30 days.

---

## 2. MCP Tools

### `search_literature`
Searches academic literature across all configured search engines and academic databases, extracts canonical identifiers, enriches metadata, and resolves open-access links.

**Parameters:**
- `query` (str): Search keywords, title query, research topic, or research question.
- `limit` (int, default: 10): Maximum number of enriched paper records to return (1-50).
- `year_from` (int | None, default: None): Optional minimum publication year filter.
- `force_browser_fallback_for_missing_abstracts` (bool, default: True): When True, retrieves missing abstracts for new preprints/conference papers via L3 browser extraction.

**Returns:**
A list of enriched `PaperRecord` objects containing title, authors, year, venue, citation counts, abstract, TLDR, identifiers (DOI, arXiv ID, PMID, S2 ID), URLs, open-access PDF links, and resolution tier.

---

### `fetch_paper_fulltext`
Extracts structured Markdown full text for an academic paper given its DOI, arXiv ID, or web URL.

**Parameters:**
- `paper_id_or_url` (str): Paper identifier (e.g. `10.1038/s41586-020-2649-2`, `2403.01234`, `arXiv:2403.01234`) or target URL.
- `sections` (list[str] | None, default: None): Specific sections to extract (e.g. `["abstract", "method", "results"]`). Case-insensitive substring matching against Markdown headings.
- `max_chars` (int, default: 35000): Maximum character budget for the returned Markdown.

**Returns:**
A `FullTextResult` object containing resolved URL, extraction method (`ar5iv_html`, `cloudflare_browser`, `jina_reader`, `pymupdf4llm_pdf`, `exa_contents`), detected sections, and Markdown content.

---

### `expand_citation_graph`
Expands the citation and reference network for a paper using Semantic Scholar Graph API with automatic fallback to OpenAlex.

**Parameters:**
- `paper_id` (str): Semantic Scholar ID, DOI (`10.xxxx/...` or `DOI:10.xxxx/...`), or arXiv ID (`2403.xxxxx` or `ARXIV:2403.xxxxx`).
- `direction` ("citations" | "references" | "both", default: "both"): Direction of citation traversal.
- `limit` (int, default: 15): Maximum papers to return per direction.

**Returns:**
Graph dictionary sorted by influential citation count and total citations.

---

## 3. Configuration & Environment Variables

Copy `.env.example` to `.env` and set any keys you have:

```bash
cp .env.example .env
```

| Variable | Tier | Description |
| :--- | :---: | :--- |
| `EXA_API_KEY` | L1 | Exa AI neural search API key |
| `TAVILY_API_KEY` | L1 | Tavily search API key |
| `SERPER_API_KEY` | L1 | Google Serper API key for Google Scholar |
| `ANYSEARCH_API_URL` | L1 | Custom AnySearch or SearXNG instance endpoint |
| `ANYSEARCH_API_KEY` | L1 | AnySearch API authentication token |
| `S2_API_KEY` | L2 | Semantic Scholar API key (higher rate limits) |
| `OPENALEX_EMAIL` | L2 | Email for OpenAlex Polite Pool (higher throughput) |
| `UNPAYWALL_EMAIL` | L2 | Email for Unpaywall OA PDF lookup |
| `CF_ACCOUNT_ID` | L3 | Cloudflare Account ID for Browser Rendering |
| `CF_API_TOKEN` | L3 | Cloudflare API Token for Browser Rendering |
| `JINA_API_KEY` | L3 | Jina Reader API key (optional for higher rate limits) |
| `CACHE_DB_PATH` | Cache | Path to SQLite cache file (default: `.deepscholar_cache.sqlite3`) |
| `TOTAL_TIMEOUT_SEC` | Timeout | Unified end-to-end hard deadline for MCP operations (default: 35.0) |
| `L1_TIMEOUT_SEC` | Timeout | L1 search discovery timeout (default: 12.0) |
| `L2_TIMEOUT_SEC` | Timeout | L2 academic metadata enrichment timeout (default: 12.0) |
| `L3_TIMEOUT_SEC` | Timeout | L3 browser fallback extraction timeout (default: 12.0) |
| `MAX_CONCURRENT_REQUESTS` | Runtime | Maximum concurrent asynchronous requests (default: 8) |
| `MAX_FULLTEXT_CHARS` | Runtime | Truncation limit for full-text extraction (default: 35000) |

---

## 4. Client Integration Examples

### Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "deepscholar": {
      "command": "python",
      "args": ["-m", "deepscholar_mcp.server"],
      "env": {
        "OPENALEX_EMAIL": "your_email@domain.com",
        "UNPAYWALL_EMAIL": "your_email@domain.com",
        "S2_API_KEY": "your_s2_key_optional",
        "EXA_API_KEY": "your_exa_key_optional"
      }
    }
  }
}
```

### Cursor (`.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "deepscholar": {
      "command": "deepscholar-mcp",
      "env": {
        "OPENALEX_EMAIL": "your_email@domain.com",
        "UNPAYWALL_EMAIL": "your_email@domain.com"
      }
    }
  }
}
```

### Codex CLI / Antigravity MCP Config
```json
{
  "mcpServers": {
    "deepscholar": {
      "command": "uv",
      "args": ["run", "--with", "deepscholar-mcp", "deepscholar-mcp"]
    }
  }
}
```

---

## 5. Development & Testing

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1

# Install package in editable mode with development dependencies
pip install -e ".[dev]"

# Run offline mocked test suite
pytest -v
```

---

## 6. License
MIT License.

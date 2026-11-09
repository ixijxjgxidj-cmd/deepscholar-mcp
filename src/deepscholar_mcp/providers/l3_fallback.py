"""Stage 4 — L3 Browser & Full-Text Extraction Fallback providers."""

import asyncio
import logging
import re
from typing import Any

import httpx
import pymupdf
import pymupdf4llm
from bs4 import BeautifulSoup

from deepscholar_mcp.config import Settings
from deepscholar_mcp.resolver import extract_identifiers

logger = logging.getLogger(__name__)

HEADING_REGEX = re.compile(r"^(#{1,4})\s+(.+)$")


def rewrite_to_ar5iv_url(url_or_id: str, arxiv_id: str | None = None) -> str | None:
    """Rewrite arXiv ID or URL to ar5iv HTML URL."""
    target_id = arxiv_id
    if not target_id:
        target_id = extract_identifiers(url_or_id).arxiv_id

    if target_id:
        canonical = re.sub(r"v\d+$", "", target_id.strip())
        return f"https://ar5iv.labs.arxiv.org/html/{canonical}"
    return None


def _html_to_markdown(html_content: str) -> str:
    """Convert HTML document into clean structured Markdown."""
    soup = BeautifulSoup(html_content, "lxml")

    # Decompose unwanted elements
    for tag in soup.find_all(
        ["nav", "header", "footer", "script", "style", "noscript", "svg"]
    ):
        tag.decompose()

    # Replace MathML with readable LaTeX alttext
    for math_tag in soup.find_all("math"):
        alttext = math_tag.get("alttext")
        if alttext:
            math_tag.replace_with(f" ${alttext.strip()}$ ")
        else:
            math_tag.replace_with(f" ${math_tag.get_text().strip()}$ ")

    # Process headings
    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        level = int(h.name[1])
        h_text = h.get_text().strip()
        h.replace_with(f"\n\n{'#' * level} {h_text}\n\n")

    # Process paragraphs
    for p in soup.find_all("p"):
        p_text = p.get_text().strip()
        p.replace_with(f"\n\n{p_text}\n\n")

    # Process list items
    for li in soup.find_all("li"):
        li_text = li.get_text().strip()
        li.replace_with(f"\n- {li_text}")

    raw_text = soup.get_text()
    # Normalize multiple line breaks
    cleaned = re.sub(r"\n{3,}", "\n\n", raw_text)
    return cleaned.strip()


async def fetch_ar5iv_html_markdown(
    client: httpx.AsyncClient,
    arxiv_id: str,
) -> str | None:
    """Fetch and parse paper full text from ar5iv HTML (or arxiv.org HTML fallback)."""
    canonical_id = re.sub(r"v\d+$", "", arxiv_id.strip())
    urls = [
        f"https://ar5iv.labs.arxiv.org/html/{canonical_id}",
        f"https://arxiv.org/html/{canonical_id}",
    ]

    for target_url in urls:
        try:
            resp = await client.get(target_url, timeout=15.0, follow_redirects=True)
            if resp.status_code == 200 and len(resp.text) > 500:
                md = _html_to_markdown(resp.text)
                if len(md) >= 400:
                    return md
        except Exception as exc:
            logger.warning("Failed to fetch ar5iv from %s: %s", target_url, exc)

    return None


async def fetch_via_exa_contents(
    client: httpx.AsyncClient,
    url: str,
    settings: Settings,
) -> str | None:
    """Extract clean text content via Exa /contents endpoint."""
    if not settings.has_exa:
        return None

    api_url = "https://api.exa.ai/contents"
    headers = {
        "x-api-key": settings.EXA_API_KEY or "",
        "Content-Type": "application/json",
    }
    payload = {
        "ids": [url],
        "text": {
            "maxCharacters": settings.MAX_FULLTEXT_CHARS,
            "includeHtmlTags": False,
        },
    }

    try:
        resp = await client.post(api_url, headers=headers, json=payload, timeout=15.0)
        if resp.status_code != 200:
            logger.warning("Exa contents returned status %d", resp.status_code)
            return None

        data: dict[str, Any] = resp.json()
        results = data.get("results", [])
        if results and isinstance(results[0], dict) and results[0].get("text"):
            text = results[0]["text"].strip()
            if len(text) >= 200:
                return text
        return None
    except Exception as exc:
        logger.warning("Exa contents extraction failed for %s: %s", url, exc)
        return None


async def fetch_via_cloudflare_browser(
    client: httpx.AsyncClient,
    url: str,
    settings: Settings,
) -> str | None:
    """Render SPA and extract Markdown via Cloudflare Browser Rendering API."""
    if not settings.has_cloudflare_browser:
        return None

    api_url = (
        f"https://api.cloudflare.com/client/v4/accounts/{settings.CF_ACCOUNT_ID}"
        "/browser-rendering/markdown"
    )
    headers = {
        "Authorization": f"Bearer {settings.CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {
        "url": url,
        "gotoOptions": {"waitUntil": "networkidle2", "timeout": 15000},
    }

    try:
        resp = await client.post(api_url, headers=headers, json=body, timeout=20.0)
        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            if data.get("success") is True and data.get("result"):
                result_text = str(data["result"]).strip()
                if len(result_text) >= 200:
                    return result_text
        else:
            logger.warning("Cloudflare browser error %d: %s", resp.status_code, resp.text[:200])
        return None
    except Exception as exc:
        logger.warning("Cloudflare browser extraction failed for %s: %s", url, exc)
        return None


async def fetch_via_jina_reader(
    client: httpx.AsyncClient,
    url: str,
    settings: Settings,
) -> str | None:
    """Extract clean Markdown via Jina Reader (r.jina.ai)."""
    clean_url = re.sub(r"^https?://r\.jina\.ai/", "", url).strip()
    reader_url = f"https://r.jina.ai/{clean_url}"
    headers = {
        "Accept": "text/plain",
        "X-Return-Format": "markdown",
    }
    if settings.JINA_API_KEY:
        headers["Authorization"] = f"Bearer {settings.JINA_API_KEY}"

    try:
        resp = await client.get(reader_url, headers=headers, timeout=18.0)
        if resp.status_code == 200:
            text = resp.text.strip()
            if len(text) >= 200:
                return text
        return None
    except Exception as exc:
        logger.warning("Jina Reader extraction failed for %s: %s", url, exc)
        return None


async def fetch_and_parse_pdf(
    client: httpx.AsyncClient,
    pdf_url: str,
) -> str | None:
    """Download PDF stream and parse to structured Markdown using PyMuPDF4LLM."""
    try:
        resp = await client.get(pdf_url, follow_redirects=True, timeout=25.0)
        if resp.status_code != 200:
            logger.warning("PDF download returned status %d for %s", resp.status_code, pdf_url)
            return None

        content_type = resp.headers.get("content-type", "").lower()
        is_pdf = (
            "application/pdf" in content_type
            or pdf_url.lower().endswith(".pdf")
            or resp.content.startswith(b"%PDF-")
        )

        if not is_pdf:
            logger.warning("Downloaded payload from %s is not PDF", pdf_url)
            return None

        def _parse_pdf_sync(data_bytes: bytes) -> str:
            doc = pymupdf.open(stream=data_bytes, filetype="pdf")
            try:
                return pymupdf4llm.to_markdown(doc)
            finally:
                doc.close()

        markdown_text = await asyncio.to_thread(_parse_pdf_sync, resp.content)
        if markdown_text and len(markdown_text.strip()) >= 100:
            return markdown_text.strip()
        return None
    except Exception as exc:
        logger.warning("PyMuPDF4LLM extraction failed for %s: %s", pdf_url, exc)
        return None


def slice_markdown_sections(
    markdown_text: str,
    requested_sections: list[str] | None,
    max_chars: int,
) -> tuple[list[str], str, bool]:
    """Extract section titles, filter requested sections, and enforce token budget."""
    lines = markdown_text.splitlines()
    available_sections: list[str] = []

    # First pass: collect available section headings
    for line in lines:
        match = HEADING_REGEX.match(line)
        if match:
            available_sections.append(match.group(2).strip())

    # If no specific sections requested, retain full text
    if not requested_sections:
        result_text = markdown_text
    else:
        # Build sections by slicing text based on headings
        norm_requested = [sec.lower().strip() for sec in requested_sections if sec.strip()]
        matched_blocks: list[str] = []
        capturing = False
        captured_level = 0
        current_block: list[str] = []

        for line in lines:
            h_match = HEADING_REGEX.match(line)
            if h_match:
                level = len(h_match.group(1))
                h_title = h_match.group(2).strip().lower()

                # Check if this heading matches any requested section
                is_target = any(req in h_title for req in norm_requested)

                if is_target:
                    if capturing and current_block:
                        matched_blocks.append("\n".join(current_block))
                        current_block = []
                    capturing = True
                    captured_level = level
                    current_block.append(line)
                elif capturing:
                    # If we encounter an equal or higher-level heading, stop capturing current section
                    if level <= captured_level:
                        matched_blocks.append("\n".join(current_block))
                        current_block = []
                        capturing = False
                    else:
                        current_block.append(line)
            elif capturing:
                current_block.append(line)

        if capturing and current_block:
            matched_blocks.append("\n".join(current_block))

        if matched_blocks:
            result_text = "\n\n".join(matched_blocks)
        else:
            # Fall back to full text if no requested section matched
            result_text = markdown_text

    truncated = False
    if len(result_text) > max_chars:
        result_text = result_text[:max_chars].rstrip() + "\n\n[... Truncated due to max_chars limit ...]"
        truncated = True

    return available_sections, result_text, truncated

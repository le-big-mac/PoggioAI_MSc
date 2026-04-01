"""
CitationSearchTool - Search and format academic citations for LaTeX papers.

This tool provides comprehensive academic search capabilities including:
- arXiv paper search and metadata extraction
- Semantic Scholar integration for broader academic coverage
- BibTeX citation generation for LaTeX
- Citation formatting and validation
- Related work discovery and recommendation

Combines functionality from fetch_arxiv_papers and web search for academic sources.
"""

from __future__ import annotations
import json
import os
import re
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

from ...workflow_utils import safe_int_env as _safe_int_env, safe_float_env as _safe_float_env


class CitationSearchTool:
    def __init__(self, **kwargs: Any):
        self.arxiv_base_url = "http://export.arxiv.org/api/query?"
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.semantic_scholar_max_retries = self._safe_int_env("CONSORTIUM_SS_MAX_RETRIES", 3)
        self.semantic_scholar_base_delay = self._safe_float_env("CONSORTIUM_SS_BASE_DELAY_SEC", 2.0)
        self.semantic_scholar_cooldown = self._safe_float_env("CONSORTIUM_SS_COOLDOWN_SEC", 60.0)
        self.semantic_scholar_timeout = self._safe_int_env("CONSORTIUM_SS_TIMEOUT_SEC", 30)
        self._semantic_scholar_cooldown_until = 0.0
        self._cache_ttl_seconds = 1800
        self._cache_max_entries = 256
        self._cache: Dict[str, Dict[str, Any]] = {}

    # Use module-level _safe_int_env / _safe_float_env from workflow_utils
    _safe_int_env = staticmethod(_safe_int_env)
    _safe_float_env = staticmethod(_safe_float_env)

    @staticmethod
    def _extract_arxiv_id(text: str) -> Optional[str]:
        if not text:
            return None
        # Supports patterns like:
        # - arXiv:1706.03762
        # - 1706.03762
        # - 1706.03762v3
        match = re.search(r"(?:arxiv:)?\s*(\d{4}\.\d{4,5}(?:v\d+)?)", text, flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(1)

    def _cache_get(self, key: str) -> Optional[str]:
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry["created_at"] > self._cache_ttl_seconds:
            self._cache.pop(key, None)
            return None
        return entry["value"]

    def _cache_put(self, key: str, value: str) -> None:
        if self._cache_max_entries <= 0:
            return
        if len(self._cache) >= self._cache_max_entries and self._cache:
            oldest_key = min(self._cache.items(), key=lambda item: item[1]["created_at"])[0]
            self._cache.pop(oldest_key, None)
        self._cache[key] = {"value": value, "created_at": time.time()}

    def run(self, search_query: str, max_results: Optional[int] = None, search_source: Optional[str] = None) -> str:
        """
        Search for academic papers and generate citations.

        Args:
            search_query: Search query for papers
            max_results: Maximum number of papers to return
            search_source: Source to search ('arxiv', 'semantic_scholar', or 'both')

        Returns:
            JSON string containing structured citations
        """
        if max_results is None:
            max_results = 10
        if search_source is None:
            search_source = "both"
        try:
            cache_key = f"{search_source}|{max_results}|{search_query.strip().lower()}"
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

            citations = []
            arxiv_id = self._extract_arxiv_id(search_query)
            # Deterministic mode for explicit arXiv IDs: prioritize exact arXiv lookup
            # and skip Semantic Scholar when source='both' to avoid retry loops/noisy matches.
            deterministic_arxiv_lookup = arxiv_id is not None and search_source in ["arxiv", "both"]

            if search_source in ["arxiv", "both"]:
                arxiv_results = self._search_arxiv(
                    arxiv_id if deterministic_arxiv_lookup else search_query,
                    max(1, max_results // 2) if search_source == "both" else max_results,
                )
                citations.extend(arxiv_results)

            use_semantic_scholar = search_source in ["semantic_scholar", "both"]
            if deterministic_arxiv_lookup and search_source == "both":
                use_semantic_scholar = False

            if use_semantic_scholar:
                ss_results = self._search_semantic_scholar(
                    search_query,
                    max(1, max_results // 2) if search_source == "both" else max_results,
                )
                citations.extend(ss_results)

            # Remove duplicates based on title similarity
            citations = self._deduplicate_citations(citations)

            # Limit to max_results
            citations = citations[:max_results]

            # Generate BibTeX entries
            bibtex_entries = []
            for citation in citations:
                bibtex = self._generate_bibtex(citation)
                if bibtex:
                    bibtex_entries.append(bibtex)

            result = {
                "search_query": search_query,
                "search_source": search_source,
                "total_results": len(citations),
                "citations": citations,
                "bibtex_entries": bibtex_entries,
                "usage_instructions": {
                    "latex_integration": "Copy BibTeX entries to your .bib file and use \\cite{key} in LaTeX",
                    "citation_keys": [self._extract_citation_key(bibtex) for bibtex in bibtex_entries if bibtex]
                }
            }

            result_json = json.dumps(result, indent=2)
            self._cache_put(cache_key, result_json)
            return result_json

        except Exception as e:
            error_result = {
                "error": f"Citation search failed: {str(e)}",
                "search_query": search_query,
                "citations": [],
                "bibtex_entries": []
            }
            return json.dumps(error_result, indent=2)


    def _search_arxiv(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search arXiv for papers."""
        try:
            # Add small delay to be respectful to arXiv API
            time.sleep(1)
            arxiv_id = self._extract_arxiv_id(query)
            if arxiv_id:
                url = f"{self.arxiv_base_url}id_list={arxiv_id}"
            else:
                url = (
                    f"{self.arxiv_base_url}search_query=all:{query}"
                    f"&start=0&max_results={max_results}"
                    "&sortBy=submittedDate&sortOrder=descending"
                )

            headers = {
                "User-Agent": "Academic-Citation-Tool/1.0 (research-tool)"
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            return self._parse_arxiv_response(response.text)

        except Exception as e:
            print(f"Warning: arXiv search failed: {e}")
            return []

    def _search_semantic_scholar(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for papers with capped retries and cooldown."""
        try:
            now = time.time()
            if now < self._semantic_scholar_cooldown_until:
                remaining = int(self._semantic_scholar_cooldown_until - now)
                print(f"Warning: Semantic Scholar in cooldown, skipping request ({remaining}s remaining).")
                return []

            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,year,abstract,citationCount,venue,externalIds,url"
            }

            headers = {
                "User-Agent": "Academic-Citation-Tool/1.0 (research-tool; contact@example.com)"
            }

            # Add small delay between requests to reduce burstiness.
            time.sleep(self.semantic_scholar_base_delay)

            for attempt in range(self.semantic_scholar_max_retries + 1):
                response = requests.get(
                    self.semantic_scholar_base_url,
                    params=params,
                    headers=headers,
                    timeout=self.semantic_scholar_timeout
                )
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_semantic_scholar_response(data)

                retryable = response.status_code in {429, 500, 502, 503, 504}
                if retryable and attempt < self.semantic_scholar_max_retries:
                    wait_seconds = self.semantic_scholar_base_delay * (2 ** attempt)
                    print(
                        "Warning: Semantic Scholar request failed "
                        f"(status {response.status_code}), retrying in {wait_seconds:.1f}s "
                        f"[attempt {attempt + 1}/{self.semantic_scholar_max_retries}]"
                    )
                    time.sleep(wait_seconds)
                    continue

                if response.status_code == 429:
                    self._semantic_scholar_cooldown_until = time.time() + self.semantic_scholar_cooldown
                    print(
                        "Warning: Semantic Scholar rate limit exceeded. "
                        f"Entering cooldown for {int(self.semantic_scholar_cooldown)}s."
                    )
                    return []

                response.raise_for_status()
                return []

            return []

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                self._semantic_scholar_cooldown_until = time.time() + self.semantic_scholar_cooldown
                print(f"Warning: Semantic Scholar rate limit exceeded. Try again later.")
                return []
            else:
                print(f"Warning: Semantic Scholar HTTP error: {e}")
                return []
        except Exception as e:
            print(f"Warning: Semantic Scholar search failed: {e}")
            return []

    def _parse_arxiv_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse arXiv API response XML."""
        try:
            root = ET.fromstring(response_text)
            papers = []

            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                paper = {
                    "source": "arxiv",
                    "title": "",
                    "authors": [],
                    "year": "",
                    "abstract": "",
                    "url": "",
                    "arxiv_id": "",
                    "citation_count": 0,
                    "venue": "arXiv preprint"
                }

                # Title
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                if title_elem is not None:
                    paper["title"] = title_elem.text.strip()

                # Authors
                for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
                    name_elem = author.find("{http://www.w3.org/2005/Atom}name")
                    if name_elem is not None:
                        paper["authors"].append(name_elem.text.strip())

                # Published date
                published_elem = entry.find("{http://www.w3.org/2005/Atom}published")
                if published_elem is not None:
                    paper["year"] = published_elem.text[:4]

                # Abstract
                summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                if summary_elem is not None:
                    paper["abstract"] = summary_elem.text.strip()

                # URL and arXiv ID
                id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
                if id_elem is not None:
                    paper["url"] = id_elem.text
                    paper["arxiv_id"] = id_elem.text.split("/")[-1]

                if paper["title"]:
                    papers.append(paper)

            return papers

        except Exception as e:
            print(f"Warning: Failed to parse arXiv response: {e}")
            return []

    def _parse_semantic_scholar_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Semantic Scholar API response."""
        try:
            papers = []

            for item in data.get("data", []):
                paper = {
                    "source": "semantic_scholar",
                    "title": item.get("title", ""),
                    "authors": [author.get("name", "") for author in item.get("authors", [])],
                    "year": str(item.get("year", "")),
                    "abstract": item.get("abstract", ""),
                    "url": item.get("url", ""),
                    "arxiv_id": "",
                    "citation_count": item.get("citationCount", 0),
                    "venue": item.get("venue", "")
                }

                # Extract arXiv ID if present
                external_ids = item.get("externalIds", {})
                if external_ids and "ArXiv" in external_ids:
                    paper["arxiv_id"] = external_ids["ArXiv"]

                if paper["title"]:
                    papers.append(paper)

            return papers

        except Exception as e:
            print(f"Warning: Failed to parse Semantic Scholar response: {e}")
            return []

    def _deduplicate_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate citations based on title similarity."""
        if not citations:
            return citations

        unique_citations = []
        seen_titles = set()

        for citation in citations:
            title = citation.get("title", "").lower().strip()
            # Simple deduplication - normalize title and check if similar exists
            normalized_title = re.sub(r'[^\w\s]', '', title).replace(' ', '')

            if normalized_title not in seen_titles and len(normalized_title) > 10:
                seen_titles.add(normalized_title)
                unique_citations.append(citation)

        return unique_citations

    def _generate_bibtex(self, citation: Dict[str, Any]) -> Optional[str]:
        """Generate BibTeX entry for a citation."""
        try:
            title = citation.get("title", "").strip()
            if not title:
                return None

            # Generate citation key
            first_author = citation.get("authors", ["Unknown"])[0] if citation.get("authors") else "Unknown"
            first_author_last = first_author.split()[-1] if " " in first_author else first_author
            year = citation.get("year", "")

            # Clean author name for key
            clean_author = re.sub(r'[^\w]', '', first_author_last.lower())
            citation_key = f"{clean_author}{year}"

            # Choose entry type
            if citation.get("arxiv_id"):
                entry_type = "misc"
                note_field = f"arXiv preprint arXiv:{citation['arxiv_id']}"
            elif citation.get("venue") and "conference" in citation.get("venue", "").lower():
                entry_type = "inproceedings"
                note_field = ""
            elif citation.get("venue") and "journal" in citation.get("venue", "").lower():
                entry_type = "article"
                note_field = ""
            else:
                entry_type = "misc"
                note_field = ""

            # Format authors
            authors = citation.get("authors", [])
            if authors:
                author_str = " and ".join(authors)
            else:
                author_str = "Unknown"

            # Build BibTeX entry
            bibtex_lines = [f"@{entry_type}{{{citation_key},"]
            bibtex_lines.append(f'  title = {{{title}}},')
            bibtex_lines.append(f'  author = {{{author_str}}},')

            if year:
                bibtex_lines.append(f'  year = {{{year}}},')

            if citation.get("venue"):
                if entry_type == "inproceedings":
                    bibtex_lines.append(f'  booktitle = {{{citation["venue"]}}},')
                elif entry_type == "article":
                    bibtex_lines.append(f'  journal = {{{citation["venue"]}}},')

            if note_field:
                bibtex_lines.append(f'  note = {{{note_field}}},')

            if citation.get("url"):
                bibtex_lines.append(f'  url = {{{citation["url"]}}},')

            bibtex_lines.append("}")

            return "\n".join(bibtex_lines)

        except Exception as e:
            print(f"Warning: Failed to generate BibTeX for citation: {e}")
            return None

    def _extract_citation_key(self, bibtex: str) -> Optional[str]:
        """Extract citation key from BibTeX entry."""
        try:
            match = re.search(r'@\w+\{([^,]+),', bibtex)
            return match.group(1) if match else None
        except Exception:
            return None

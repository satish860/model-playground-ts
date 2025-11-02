"""Grep-style search tool with regex and exact phrase matching."""

import json
import re
from .data_loader import load_docs_data


def chunk_grep(pattern: str, case_sensitive: bool = False, search_in: str = "all", max_results: int = 10) -> str:
    """Grep-style search with regex and exact phrase matching.

    Args:
        pattern: Search pattern (supports regex, exact phrases in quotes)
        case_sensitive: Whether search is case sensitive
        search_in: Where to search - "all", "heading", "text"
        max_results: Maximum number of results to return

    Returns:
        JSON string with search results
    """
    print(f"  [TOOL EXECUTED] chunk_grep(pattern='{pattern}', case_sensitive={case_sensitive}, search_in='{search_in}', max={max_results})")

    docs = load_docs_data()
    results = []

    # Prepare regex pattern
    regex_flags = 0 if case_sensitive else re.IGNORECASE

    try:
        # Check if pattern is a quoted phrase for exact matching
        if pattern.startswith('"') and pattern.endswith('"'):
            # Exact phrase match
            search_pattern = re.escape(pattern.strip('"'))
        else:
            # Treat as regex pattern
            search_pattern = pattern

        compiled_pattern = re.compile(search_pattern, regex_flags)
    except re.error:
        # Invalid regex, fall back to literal search
        compiled_pattern = re.compile(re.escape(pattern), regex_flags)

    for idx, chunk in enumerate(docs):
        heading = chunk.get('chunk_heading', '')
        text = chunk.get('text', '')
        link = chunk.get('chunk_link', '')

        # Determine what to search
        if search_in == "heading":
            search_text = heading
        elif search_in == "text":
            search_text = text
        else:  # "all"
            search_text = heading + "\n" + text

        # Find all matches
        matches = list(compiled_pattern.finditer(search_text))

        if matches:
            match_count = len(matches)

            # Calculate score: phrase matches score higher
            # Exact phrase match in heading = highest score
            heading_match = compiled_pattern.search(heading)

            # Check if this is an exact phrase search (quoted pattern)
            is_exact_phrase = pattern.startswith('"') and pattern.endswith('"')

            if heading_match:
                match_score = 10.0  # Match in heading
            elif is_exact_phrase and match_count > 0:
                match_score = 8.0  # Exact phrase match in text
            else:
                # Score based on match density
                match_score = min(10.0, (match_count * 100.0) / max(1, len(search_text)))

            # Extract snippet around first match
            first_match = matches[0]
            snippet_start = max(0, first_match.start() - 50)
            snippet_end = min(len(text), first_match.end() + 150)
            snippet = "..." + text[snippet_start:snippet_end] + "..."

            results.append({
                "chunk_id": idx,
                "chunk_link": link,
                "chunk_heading": heading,
                "snippet": snippet,
                "match_count": match_count,
                "match_score": round(match_score, 3)
            })

    # Sort by match score (descending) and limit results
    results.sort(key=lambda x: x['match_score'], reverse=True)
    results = results[:max_results]

    result = json.dumps(results, indent=2)
    print(f"  [TOOL RESULT] Found {len(results)} chunks")
    return result

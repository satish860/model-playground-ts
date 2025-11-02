"""Keyword search tool for documentation chunks."""

import json
from .data_loader import load_docs_data


def chunk_search(query: str, case_sensitive: bool = False, search_in: str = "all", max_results: int = 10) -> str:
    """Search documentation chunks by keyword.

    Args:
        query: Search query string
        case_sensitive: Whether search is case sensitive
        search_in: Where to search - "all", "heading", "text", "summary"
        max_results: Maximum number of results to return

    Returns:
        JSON string with search results
    """
    print(f"  [TOOL EXECUTED] chunk_search(query='{query}', search_in='{search_in}', max={max_results})")

    docs = load_docs_data()
    results = []

    # Prepare query for searching
    search_query = query if case_sensitive else query.lower()
    query_terms = search_query.split()

    for idx, chunk in enumerate(docs):
        # Prepare search fields
        heading = chunk.get('chunk_heading', '')
        text = chunk.get('text', '')
        link = chunk.get('chunk_link', '')

        if not case_sensitive:
            heading = heading.lower()
            text = text.lower()

        # Determine what to search
        if search_in == "heading":
            search_text = heading
        elif search_in == "text":
            search_text = text
        else:  # "all"
            search_text = heading + " " + text

        # Count matches for each query term
        match_count = 0
        for term in query_terms:
            match_count += search_text.count(term)

        if match_count > 0:
            # Calculate match score (simple: matches / length of text)
            match_score = min(1.0, match_count / max(1, len(search_text.split()) / 100))

            # Extract snippet around first match
            first_term = query_terms[0]
            snippet_start = search_text.find(first_term)
            if snippet_start != -1:
                # Get context around match
                snippet_start = max(0, snippet_start - 50)
                snippet_end = min(len(text), snippet_start + 200)
                snippet = "..." + text[snippet_start:snippet_end] + "..."
            else:
                snippet = text[:200] + "..."

            results.append({
                "chunk_id": idx,
                "chunk_link": link,
                "chunk_heading": chunk.get('chunk_heading', ''),
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

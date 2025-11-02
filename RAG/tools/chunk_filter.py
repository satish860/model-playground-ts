"""Tool to filter chunks by heading or link patterns."""

import json
import re
from .data_loader import load_docs_data


def chunk_filter(heading_pattern: str = None, link_pattern: str = None) -> str:
    """Filter chunks by heading or link pattern.

    Args:
        heading_pattern: Regex pattern to match against headings
        link_pattern: Regex pattern to match against links

    Returns:
        JSON string with list of matching chunk IDs
    """
    print(f"  [TOOL EXECUTED] chunk_filter(heading='{heading_pattern}', link='{link_pattern}')")

    docs = load_docs_data()
    matching_ids = []

    for idx, chunk in enumerate(docs):
        heading = chunk.get('chunk_heading', '')
        link = chunk.get('chunk_link', '')

        # Check if chunk matches patterns
        heading_match = True
        link_match = True

        if heading_pattern:
            try:
                heading_match = bool(re.search(heading_pattern, heading, re.IGNORECASE))
            except re.error:
                # Invalid regex pattern
                heading_match = heading_pattern.lower() in heading.lower()

        if link_pattern:
            try:
                link_match = bool(re.search(link_pattern, link, re.IGNORECASE))
            except re.error:
                # Invalid regex pattern
                link_match = link_pattern.lower() in link.lower()

        if heading_match and link_match:
            matching_ids.append(idx)

    result = json.dumps(matching_ids)
    print(f"  [TOOL RESULT] Found {len(matching_ids)} matching chunks")
    return result

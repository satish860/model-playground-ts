"""Simple documentation search agent (Level 1).

Agentic search agent that uses tool calling to search and synthesize
information from Claude documentation.

Phase 1a: Agent loop infrastructure with stub tools
Phase 1b: Implement real tool functions
"""

import sys
import os
import json
import re

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python.utils.config import Config


# ============================================================================
# DATA LOADING
# ============================================================================

# Global variable to cache documentation data
_DOCS_DATA = None

def load_docs_data():
    """Load documentation data from JSON file (cached)."""
    global _DOCS_DATA
    if _DOCS_DATA is None:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'anthropic_docs.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            _DOCS_DATA = json.load(f)
    return _DOCS_DATA


# ============================================================================
# REAL TOOL IMPLEMENTATIONS
# ============================================================================

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


def chunk_read(chunk_ids: list) -> str:
    """Read full content of specific chunks.

    Args:
        chunk_ids: List of chunk IDs to read

    Returns:
        JSON string with full chunk contents
    """
    print(f"  [TOOL EXECUTED] chunk_read(chunk_ids={chunk_ids})")

    docs = load_docs_data()
    chunks = []

    for chunk_id in chunk_ids:
        if 0 <= chunk_id < len(docs):
            chunk = docs[chunk_id]
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_link": chunk.get('chunk_link', ''),
                "chunk_heading": chunk.get('chunk_heading', ''),
                "text": chunk.get('text', '')
            })
        else:
            chunks.append({
                "chunk_id": chunk_id,
                "error": f"Chunk ID {chunk_id} out of range (0-{len(docs)-1})"
            })

    result = json.dumps(chunks, indent=2)
    print(f"  [TOOL RESULT] Read {len(chunks)} chunks")
    return result


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


# ============================================================================
# TOOL SCHEMAS
# ============================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "chunk_search",
            "description": "Search documentation chunks by keyword. Returns chunk previews with IDs, headings, snippets, and match scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (keywords to find in documentation)"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search should be case sensitive (default: false)"
                    },
                    "search_in": {
                        "type": "string",
                        "enum": ["all", "heading", "text", "summary"],
                        "description": "Where to search - all fields, just headings, just text, or just summaries (default: all)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "chunk_read",
            "description": "Read the full content of specific documentation chunks. Use this after searching to get complete information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of chunk IDs to read (from search results)"
                    }
                },
                "required": ["chunk_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "chunk_filter",
            "description": "Filter chunks by heading or link patterns. Returns list of chunk IDs matching the pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "heading_pattern": {
                        "type": "string",
                        "description": "Regex pattern to match against chunk headings (e.g., 'API.*streaming')"
                    },
                    "link_pattern": {
                        "type": "string",
                        "description": "Regex pattern to match against chunk links (e.g., 'docs/api-reference')"
                    }
                }
            }
        }
    }
]

# Map function names to actual functions
available_functions = {
    "chunk_search": chunk_search,
    "chunk_read": chunk_read,
    "chunk_filter": chunk_filter
}


# ============================================================================
# SYSTEM PROMPT (Level 1)
# ============================================================================

SYSTEM_PROMPT = """You are a helpful documentation search assistant.

Your task: Answer questions about Claude documentation using the available tools.

Tools:
- chunk_search: Find chunks containing keywords (returns previews with IDs)
- chunk_read: Read full content of specific chunks (use IDs from search)
- chunk_filter: Filter chunks by heading/link patterns

Strategy:
1. Search for relevant keywords
2. Read the most promising chunks (based on match scores and snippets)
3. Provide a clear answer with source citations

Always cite your sources using chunk links from the documentation.
"""


# ============================================================================
# AGENT LOOP
# ============================================================================

def run_agent(user_question: str, max_iterations: int = 5):
    """Run documentation search agent.

    Args:
        user_question: The user's question about Claude documentation
        max_iterations: Maximum number of agent loop iterations

    Returns:
        The agent's final answer
    """
    client = Config.get_client()

    print("=" * 80)
    print("DOCUMENTATION SEARCH AGENT")
    print("=" * 80)
    print(f"Question: {user_question}")
    print(f"Max Iterations: {max_iterations}")
    print("=" * 80)
    print()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question}
    ]

    for iteration in range(max_iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*80}")

        # Make API call with tools
        response = client.chat.completions.create(
            model=Config.MODEL_FREE,
            messages=messages,
            tools=tools
        )

        response_message = response.choices[0].message

        # Check for thinking/reasoning (MiniMax M2 special feature)
        if hasattr(response_message, 'reasoning') and response_message.reasoning:
            reasoning_text = response_message.reasoning

            print(f"\n{'─'*80}")
            print("AGENT'S THINKING:")
            print(f"{'─'*80}")

            if '<think>' in reasoning_text and '</think>' in reasoning_text:
                thinking = reasoning_text.split('<think>')[1].split('</think>')[0].strip()
                print(f"\n{thinking}\n")
                print(f"{'─'*80}")
            else:
                print(f"\n{reasoning_text}\n")
                print(f"{'─'*80}")

        # Debug metadata
        print(f"\nResponse Metadata:")
        print(f"  finish_reason: {response.choices[0].finish_reason}")
        print(f"  has tool_calls: {response_message.tool_calls is not None}")
        print(f"  has content: {bool(response_message.content)}")

        # Handle tool calls
        if response_message.tool_calls:
            print(f"\nTool Calls: {len(response_message.tool_calls)}")

            # Add assistant's response to messages
            messages.append(response_message)

            # Execute each tool call
            for idx, tool_call in enumerate(response_message.tool_calls, 1):
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"\n  Tool {idx}: {function_name}")
                print(f"  Arguments: {json.dumps(function_args, indent=2)}")

                # Call the function
                if function_name in available_functions:
                    function_response = available_functions[function_name](**function_args)
                else:
                    function_response = f"Error: Function '{function_name}' not found"

                # Add function response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(function_response)
                })

        else:
            # No more tool calls - return final answer
            final_text = response_message.content

            # Handle MiniMax M2 reasoning field
            if not final_text and hasattr(response_message, 'reasoning') and response_message.reasoning:
                reasoning_text = response_message.reasoning

                if '</think>' in reasoning_text:
                    parts = reasoning_text.split('</think>')
                    final_text = parts[-1].strip()
                else:
                    final_text = reasoning_text

            print("\n" + "=" * 80)
            print("FINAL ANSWER")
            print("=" * 80)
            print(f"\n{final_text}\n")
            print("=" * 80)
            print(f"Total Iterations: {iteration + 1}")
            print(f"Total Tokens: {response.usage.total_tokens}")
            print("=" * 80)

            return final_text

    print("\n" + "=" * 80)
    print("Max iterations reached without final response")
    print("=" * 80)
    return None


# ============================================================================
# MAIN - TEST THE AGENT
# ============================================================================

def main():
    """Test the agent with sample questions."""
    print("Testing Documentation Search Agent (Level 1)")
    print("=" * 80)
    print("All tools implemented with real data from anthropic_docs.json")
    print("=" * 80)
    print()

    try:
        # Test question
        run_agent("How do I enable streaming in Claude API?",max_iterations=15)

    except Exception as error:
        print(f"Error: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

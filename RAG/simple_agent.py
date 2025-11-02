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
# PLANNING PHASE
# ============================================================================

def create_plan(user_question: str) -> str:
    """Create a search plan before executing agent loop.

    This planning phase helps the agent:
    - Understand the question intent
    - Identify key concepts to search for
    - Determine which documentation sections to target
    - Plan search strategy

    Args:
        user_question: The user's question

    Returns:
        A search plan as string
    """
    client = Config.get_client()

    planning_prompt = f"""You are planning how to answer a question about Claude documentation.

Question: {user_question}

Analyze this question and create a search plan. Consider:
1. What is the user really asking? (e.g., UI walkthrough, API code, conceptual explanation, tutorial steps)
2. What are the EXACT key phrases to search for? (extract important quoted terms, product names, feature names)
3. Which documentation sections are most relevant? (e.g., "API reference", "tutorials", "eval-tool", "test-and-evaluate")
4. Are there specific UI elements mentioned (buttons, forms, screens)?

Provide a search plan with:
- Intent: What type of answer is needed
- Key search terms: 2-3 specific phrases to search for
- Target section: Which docs section to focus on

Example:
Question: "How do I enable streaming in Claude API?"
Plan:
- Intent: API implementation method/code example
- Key search terms: "streaming", "stream parameter", "API"
- Target: API reference documentation

Now create a plan for the given question:"""

    response = client.chat.completions.create(
        model=Config.MODEL_FREE,
        messages=[{"role": "user", "content": planning_prompt}],
        max_tokens=300
    )

    plan = response.choices[0].message.content
    return plan


# ============================================================================
# AGENT LOOP
# ============================================================================

def run_agent(user_question: str, max_iterations: int = 5, use_planning: bool = True):
    """Run documentation search agent.

    Args:
        user_question: The user's question about Claude documentation
        max_iterations: Maximum number of agent loop iterations
        use_planning: Whether to use planning phase before search

    Returns:
        The agent's final answer
    """
    client = Config.get_client()

    print("=" * 80)
    print("DOCUMENTATION SEARCH AGENT")
    print("=" * 80)
    print(f"Question: {user_question}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Planning: {'Enabled' if use_planning else 'Disabled'}")
    print("=" * 80)
    print()

    # PLANNING PHASE (runs before the loop)
    search_plan = None
    if use_planning:
        print(f"\n{'-'*80}")
        print("PLANNING PHASE")
        print(f"{'-'*80}")
        search_plan = create_plan(user_question)
        print(f"\nSearch Plan:\n{search_plan}")
        print(f"\n{'-'*80}\n")

    # Build system prompt with plan
    system_prompt = SYSTEM_PROMPT
    if search_plan:
        system_prompt = f"""{SYSTEM_PROMPT}

SEARCH PLAN (follow this plan to guide your search):
{search_plan}

Remember to follow the search plan above when deciding which tools to use and what to search for."""

    messages = [
        {"role": "system", "content": system_prompt},
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

            print(f"\n{'-'*80}")
            print("AGENT'S THINKING:")
            print(f"{'-'*80}")

            if '<think>' in reasoning_text and '</think>' in reasoning_text:
                thinking = reasoning_text.split('<think>')[1].split('</think>')[0].strip()
                print(f"\n{thinking}\n")
                print(f"{'-'*80}")
            else:
                print(f"\n{reasoning_text}\n")
                print(f"{'-'*80}")

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
# EVALUATION
# ============================================================================

def load_evaluation_data():
    """Load evaluation dataset."""
    eval_path = os.path.join(os.path.dirname(__file__), 'data', 'evaluation', 'docs_evaluation_dataset.json')
    with open(eval_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def llm_judge_answer(question: str, agent_answer: str, reference_answer: str) -> dict:
    """Use LLM to judge the quality of agent's answer.

    Args:
        question: The original question
        agent_answer: Agent's generated answer
        reference_answer: Reference answer from dataset

    Returns:
        dict with score and reasoning
    """
    client = Config.get_client()

    judge_prompt = f"""You are evaluating the quality of an AI agent's answer to a documentation question.

Question: {question}

Reference Answer (Ground Truth):
{reference_answer}

Agent's Answer:
{agent_answer}

Rate the agent's answer on a scale of 1-5 based on:
- Accuracy: Is the information factually correct?
- Completeness: Does it cover all key points from the reference?
- Clarity: Is it well-structured and easy to understand?

Respond in this format:
Score: [1-5]
Reasoning: [Brief explanation of the score]

1 = Wrong/Incomplete
2 = Partially correct but missing key information
3 = Correct but could be more complete
4 = Good answer, covers most points well
5 = Excellent answer, comprehensive and accurate"""

    response = client.chat.completions.create(
        model=Config.MODEL_FREE,
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=500
    )

    result = response.choices[0].message.content

    # Parse score
    score = 0
    reasoning = result
    if "Score:" in result:
        try:
            score_line = [line for line in result.split('\n') if 'Score:' in line][0]
            score = int(score_line.split(':')[1].strip().split()[0])
        except:
            pass

    return {
        "score": score,
        "reasoning": result
    }


def run_evaluation(num_questions=3, start_index=0):
    """Run evaluation on questions from the dataset.

    Args:
        num_questions: Number of questions to evaluate
        start_index: Starting index in the dataset
    """
    eval_data = load_evaluation_data()
    questions = eval_data[start_index:start_index + num_questions]

    print("=" * 80)
    print(f"EVALUATION: Testing on {len(questions)} questions from dataset")
    print("=" * 80)
    print()

    scores = []
    total_iterations = 0
    total_tokens = 0

    for idx, qa in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {idx}/{len(questions)} (ID: {qa['id']})")
        print(f"{'='*80}")
        print(f"Q: {qa['question']}")
        print(f"\nExpected chunks: {', '.join(qa['correct_chunks'])}")
        print()

        try:
            answer = run_agent(qa['question'], max_iterations=15)

            if answer:
                # Use LLM to judge the answer
                print(f"\n{'-'*80}")
                print("JUDGING ANSWER QUALITY...")
                print(f"{'-'*80}")

                judgment = llm_judge_answer(qa['question'], answer, qa['correct_answer'])

                print(f"\nLLM JUDGE SCORE: {judgment['score']}/5")
                print(f"\n{judgment['reasoning']}")

                print(f"\n{'-'*80}")
                print("REFERENCE ANSWER:")
                print(f"{'-'*80}")
                print(qa['correct_answer'])

                scores.append(judgment['score'])
            else:
                print("ERROR: No answer generated")
                scores.append(0)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            scores.append(0)

        print(f"\n{'='*80}\n")

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Questions evaluated: {len(questions)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}/5.0" if scores else "N/A")
    print(f"Score distribution:")
    for score in range(1, 6):
        count = scores.count(score)
        print(f"  {score}/5: {count} questions")
    print(f"{'='*80}")


# ============================================================================
# MAIN - TEST THE AGENT
# ============================================================================

def main():
    """Test the agent with sample questions or run evaluation."""
    import sys

    print("Documentation Search Agent (Level 1.5 - With Planning)")
    print("=" * 80)
    print("Features:")
    print("  - Planning phase before search (new!)")
    print("  - 3 core tools: chunk_search, chunk_read, chunk_filter")
    print("  - LLM-as-judge evaluation")
    print("=" * 80)
    print()

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--eval':
        # Run evaluation mode
        num_questions = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        run_evaluation(num_questions=num_questions)
    else:
        # Manual test mode
        print("Manual test mode. Use --eval [N] to run on evaluation dataset.")
        print()
        try:
            # Test with planning enabled
            run_agent("How can you create multiple test cases for an evaluation in the Anthropic Evaluation tool?", max_iterations=15, use_planning=True)
        except Exception as error:
            print(f"Error: {error}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

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

# Import all tools from tools module
from tools import chunk_grep, chunk_search, chunk_read, chunk_filter, load_docs_data


# ============================================================================
# TOOLS IMPORTED FROM tools/ FOLDER
# ============================================================================
# Tool implementations are in:
# - tools/chunk_grep.py: Regex and exact phrase search
# - tools/chunk_search.py: Keyword search
# - tools/chunk_read.py: Read full chunk content
# - tools/chunk_filter.py: Filter by patterns
# - tools/data_loader.py: Load documentation data



# ============================================================================
# TOOL SCHEMAS
# ============================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "chunk_grep",
            "description": "Grep-style search with regex and exact phrase matching. Use quotes for exact phrases (e.g., \"Add Test Case\"). Supports regex patterns. Best for finding specific terms or UI elements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern. Use quotes for exact phrases (\"exact phrase\") or regex patterns (eval.*tool)"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search should be case sensitive (default: false)"
                    },
                    "search_in": {
                        "type": "string",
                        "enum": ["all", "heading", "text"],
                        "description": "Where to search - all fields, just headings, or just text (default: all)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "chunk_search",
            "description": "Search documentation chunks by keywords. Splits query into words and matches individually. Good for broad exploratory searches.",
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
    "chunk_grep": chunk_grep,
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
- chunk_grep: Grep-style search for exact phrases or regex patterns (PREFER THIS FOR PRECISION!)
  * Use quotes for exact phrases: chunk_grep('"Add Test Case"')
  * Use for product names: chunk_grep('"Anthropic Evaluation tool"')
  * Use for UI elements: chunk_grep('"button"'), chunk_grep('"screen"')
  * Use for features: chunk_grep('"streaming"')
  * Supports regex: chunk_grep('eval.*tool')

- chunk_search: Keyword search (splits into words)
  * Only use for exploratory searches when you don't know exact terms
  * Less precise than grep

- chunk_read: Read full content of specific chunks (use IDs from search)
- chunk_filter: Filter chunks by heading/link patterns

Strategy - IMPORTANT:
1. ALWAYS START WITH GREP for any specific terms:
   - Question mentions "test cases"? → chunk_grep('"test cases"')
   - Question mentions "evaluation tool"? → chunk_grep('"evaluation tool"')
   - Question mentions button/screen? → chunk_grep('"button"') or chunk_grep('"screen"')

2. Examples of CORRECT grep usage:
   Q: "How to create test cases?"
   → chunk_grep('"test cases"')  ✓ NOT chunk_search("test cases") ✗

   Q: "What is the Evaluation tool?"
   → chunk_grep('"Evaluation tool"')  ✓ NOT chunk_search("Evaluation tool") ✗

3. Only use chunk_search for vague/exploratory queries

4. Read the top-scoring chunks (grep gives better scores for exact matches)

5. Always cite sources with chunk links

REMEMBER: Grep with quotes finds exact phrases. Use it!
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

Create a search plan that extracts EXACT PHRASES to search with grep.

IMPORTANT: Identify specific terms that should be searched with exact phrase matching using quotes.

Analyze:
1. What EXACT PHRASES are mentioned? (product names, features, UI elements)
   - Extract 2-3 key phrases that should be searched exactly
   - These will be used with chunk_grep('"exact phrase"')

2. What type of answer is needed?
   - UI instructions (buttons, screens)
   - API code examples
   - Conceptual explanation
   - Configuration steps

3. Which documentation section? (api-reference, test-and-evaluate, build-with-claude, etc.)

Provide a search plan with:
- Grep Terms: List 2-3 exact phrases to search (e.g., ["Add Test Case", "Evaluation tool"])
- Intent: What type of answer is needed
- Target Section: Which docs to focus on

Example 1:
Question: "How do I enable streaming in Claude API?"
Plan:
- Grep Terms: ["streaming", "stream parameter"]
- Intent: API code example
- Target Section: API reference

Example 2:
Question: "How to create test cases in the Evaluation tool?"
Plan:
- Grep Terms: ["test cases", "Evaluation tool", "Add Test Case"]
- Intent: UI instructions
- Target Section: test-and-evaluate

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

SEARCH PLAN FOR THIS QUESTION:
{search_plan}

CRITICAL: If the plan lists "Grep Terms", you MUST use chunk_grep with quotes for those exact phrases!
Example: If plan says 'Grep Terms: ["test cases"]' → Use chunk_grep('"test cases"') as your FIRST search.

Follow the search plan above when deciding which tools to use."""

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


def run_evaluation(num_questions=3, start_index=0, save_results=True):
    """Run evaluation on questions from the dataset.

    Args:
        num_questions: Number of questions to evaluate
        start_index: Starting index in the dataset
        save_results: Whether to save detailed results to file
    """
    eval_data = load_evaluation_data()
    questions = eval_data[start_index:start_index + num_questions]

    print("=" * 80)
    print(f"EVALUATION: Testing on {len(questions)} questions from dataset")
    print("=" * 80)
    print()

    scores = []
    results_detail = []
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

                # Save detailed result
                results_detail.append({
                    'id': qa['id'],
                    'question': qa['question'],
                    'score': judgment['score'],
                    'reasoning': judgment['reasoning'],
                    'agent_answer': answer,
                    'reference_answer': qa['correct_answer'],
                    'expected_chunks': qa['correct_chunks']
                })
            else:
                print("ERROR: No answer generated")
                scores.append(0)
                results_detail.append({
                    'id': qa['id'],
                    'question': qa['question'],
                    'score': 0,
                    'error': 'No answer generated'
                })

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            scores.append(0)
            results_detail.append({
                'id': qa['id'],
                'question': qa['question'],
                'score': 0,
                'error': str(e)
            })

        print(f"\n{'='*80}\n")

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Questions evaluated: {len(questions)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}/5.0" if scores else "N/A")
    print(f"Score distribution:")
    for score in range(0, 6):
        count = scores.count(score)
        percentage = (count / len(scores) * 100) if scores else 0
        print(f"  {score}/5: {count} questions ({percentage:.1f}%)")
    print(f"{'='*80}")

    # Save results
    if save_results:
        results_file = os.path.join(os.path.dirname(__file__), f'evaluation_results_{num_questions}q.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_questions': len(questions),
                    'average_score': sum(scores)/len(scores) if scores else 0,
                    'score_distribution': {str(i): scores.count(i) for i in range(0, 6)}
                },
                'results': results_detail
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return scores


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

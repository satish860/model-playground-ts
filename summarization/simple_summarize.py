"""Simple Lease Summarization using MiniMax M2.

Basic prompt-based summarization approach with visual comparison to reference summaries.
Based on Claude Cookbook: https://github.com/anthropics/claude-cookbooks/tree/main/capabilities/summarization
"""

import os
import time
from typing import Dict
from dotenv import load_dotenv
from openai import OpenAI
from rouge_score import rouge_scorer

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "minimax/minimax-m2:free"


def get_client() -> OpenAI:
    """Get configured OpenAI client for OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    return OpenAI(
        base_url=BASE_URL,
        api_key=OPENROUTER_API_KEY
    )


def load_lease(filename: str) -> str:
    """Load lease document content from data folder.

    Args:
        filename: Name of the lease file (e.g., 'sample-lease1.txt')

    Returns:
        Full content of the lease document
    """
    filepath = os.path.join(os.path.dirname(__file__), 'data', filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Lease file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


def load_reference(filename: str) -> str:
    """Load reference summary from data folder.

    Args:
        filename: Name of the reference summary file (e.g., 'sample-lease1-summary.txt')

    Returns:
        Reference summary content
    """
    filepath = os.path.join(os.path.dirname(__file__), 'data', filename)

    if not os.path.exists(filepath):
        return "[Reference summary not found]"

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


def create_summarization_prompt(lease_content: str) -> str:
    """Create a simple summarization prompt.

    Args:
        lease_content: Full text of the lease agreement

    Returns:
        Formatted prompt string
    """
    prompt = f"""Please summarize this commercial sublease agreement in clear bullet points.

Focus on:
- The parties involved (who is leasing to whom)
- The property being leased (location, size, description)
- The lease term and important dates
- Rent and financial terms (monthly rent, security deposit, increases)
- Key obligations and responsibilities
- Important clauses and restrictions

Document to summarize:
{lease_content}

Provide a concise bullet-point summary covering the key aspects listed above."""

    return prompt


def extract_thinking(response_message) -> tuple[str | None, str]:
    """Extract thinking process and summary from model response.

    Args:
        response_message: The response message from the API

    Returns:
        Tuple of (thinking_process, summary)
    """
    thinking = None
    summary = ""

    # Get content and reasoning fields
    content = response_message.content if response_message.content else ""
    reasoning = ""
    if hasattr(response_message, 'reasoning') and response_message.reasoning:
        reasoning = response_message.reasoning

    # Combine both fields for extraction
    full_text = content + "\n" + reasoning

    # Extract thinking from <think> tags (MiniMax native thinking)
    if '<think>' in full_text and '</think>' in full_text:
        thinking = full_text.split('<think>')[1].split('</think>')[0].strip()
        # Summary is what comes after </think>
        if '</think>' in full_text:
            summary = full_text.split('</think>')[-1].strip()

    # If no structured thinking found, use content as summary
    if not summary:
        summary = content.strip()

    return thinking, summary


def calculate_rouge_scores(generated: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores comparing generated summary to reference.

    Args:
        generated: Generated summary from model
        reference: Reference summary from file

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)

    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def summarize_lease(client: OpenAI, lease_content: str) -> tuple[str, str | None]:
    """Summarize a lease document using MiniMax M2.

    Args:
        client: OpenAI client instance
        lease_content: Full text of the lease

    Returns:
        Tuple of (summary, thinking_process)
    """
    prompt = create_summarization_prompt(lease_content)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # Deterministic output
        )

        response_message = response.choices[0].message

        # Extract thinking and summary
        thinking, summary = extract_thinking(response_message)

        return summary, thinking

    except Exception as error:
        print(f"\n❌ Error generating summary: {error}")
        return "[Error: Could not generate summary]", None


def display_comparison(
    generated: str,
    reference: str,
    thinking: str | None,
    rouge_scores: Dict[str, float],
    stats: Dict,
    lease_num: int,
    total_leases: int
):
    """Display generated summary alongside reference summary.

    Args:
        generated: Generated summary from model
        reference: Reference summary from file
        thinking: Model's thinking process (if available)
        rouge_scores: Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        stats: Dictionary with processing stats (time, size, etc.)
        lease_num: Current lease number
        total_leases: Total number of leases
    """
    print(f"\n{'='*80}")
    print(f"SIMPLE SUMMARIZATION - Lease {lease_num}/{total_leases}")
    print(f"{'='*80}\n")

    # File info
    print(f"📄 File: {stats['file']}")
    print(f"📏 Size: {stats['size']:,} characters (~{stats['size']//4:,} tokens)")
    print(f"⏱️  Processing time: {stats['time']:.2f} seconds\n")

    # Show thinking process if available
    if thinking:
        print("🧠 MODEL'S THINKING PROCESS:")
        print("┄" * 80)
        print(thinking)
        print("┄" * 80)
        print()

    # Generated summary
    print("✅ GENERATED SUMMARY:")
    print("━" * 80)
    print(generated)
    print("━" * 80)

    # Reference summary
    print("\n📚 REFERENCE SUMMARY:")
    print("━" * 80)
    print(reference)
    print("━" * 80)

    # ROUGE scores
    print("\n📊 ROUGE SCORES:")
    print("┄" * 80)
    print(f"   ROUGE-1: {rouge_scores['rouge1']:.4f}  (word overlap)")
    print(f"   ROUGE-2: {rouge_scores['rouge2']:.4f}  (phrase overlap)")
    print(f"   ROUGE-L: {rouge_scores['rougeL']:.4f}  (sentence structure)")
    print("┄" * 80)

    # Quick visual assessment hints
    print("\n💡 COMPARISON TIPS:")
    print("   • Are all key parties mentioned?")
    print("   • Is the property described accurately?")
    print("   • Are financial terms captured?")
    print("   • Are important dates included?")
    print("   • Is the summary concise yet complete?")


def main():
    """Run simple summarization on all lease documents."""
    print("=" * 80)
    print("INSURANCE LEASE SUMMARIZATION - SIMPLE PROMPT APPROACH")
    print("Model: MiniMax M2 Free")
    print("=" * 80)
    print("\nThis will process 9 lease agreements and compare summaries to references.\n")

    # Get configured client
    try:
        client = get_client()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nPlease set OPENROUTER_API_KEY in your .env file")
        return

    # Process each lease document
    total_start = time.time()
    successful = 0
    failed = 0
    all_scores = []  # Track ROUGE scores for all leases

    for i in range(1, 10):  # Process leases 1-9
        lease_file = f"sample-lease{i}.txt"
        reference_file = f"sample-lease{i}-summary.txt"

        try:
            # Load lease content
            lease_content = load_lease(lease_file)
            file_size = len(lease_content)

            # Generate summary
            print(f"\n{'─'*80}")
            print(f"Processing {lease_file}... ⏳")
            print(f"{'─'*80}")

            start_time = time.time()
            generated_summary, thinking = summarize_lease(client, lease_content)
            elapsed = time.time() - start_time

            # Load reference summary
            reference_summary = load_reference(reference_file)

            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(generated_summary, reference_summary)

            # Track scores
            all_scores.append({
                'lease': lease_file,
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL']
            })

            # Display comparison
            display_comparison(
                generated=generated_summary,
                reference=reference_summary,
                thinking=thinking,
                rouge_scores=rouge_scores,
                stats={
                    'time': elapsed,
                    'file': lease_file,
                    'size': file_size
                },
                lease_num=i,
                total_leases=9
            )

            successful += 1

            # Wait before next lease (rate limiting and user pacing)
            if i < 9:
                print(f"\n{'─'*80}")
                input("Press Enter to continue to next lease... ")
                time.sleep(0.5)  # Rate limiting

        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ Unexpected error processing {lease_file}: {e}")
            failed += 1

    # Final summary
    total_elapsed = time.time() - total_start

    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_elapsed:.2f} seconds")
    print(f"⏱️  Average per lease: {total_elapsed/9:.2f} seconds")
    print(f"{'='*80}\n")

    # Calculate and display ROUGE score statistics
    if all_scores:
        print(f"{'='*80}")
        print("EVALUATION SUMMARY - ROUGE SCORES")
        print(f"{'='*80}\n")

        # Calculate averages and standard deviations
        avg_rouge1 = sum(s['rouge1'] for s in all_scores) / len(all_scores)
        avg_rouge2 = sum(s['rouge2'] for s in all_scores) / len(all_scores)
        avg_rougeL = sum(s['rougeL'] for s in all_scores) / len(all_scores)

        # Calculate standard deviations
        import math
        std_rouge1 = math.sqrt(sum((s['rouge1'] - avg_rouge1)**2 for s in all_scores) / len(all_scores))
        std_rouge2 = math.sqrt(sum((s['rouge2'] - avg_rouge2)**2 for s in all_scores) / len(all_scores))
        std_rougeL = math.sqrt(sum((s['rougeL'] - avg_rougeL)**2 for s in all_scores) / len(all_scores))

        print("📊 Average Scores Across All Leases:")
        print("━" * 80)
        print(f"   ROUGE-1: {avg_rouge1:.4f} ± {std_rouge1:.4f}")
        print(f"   ROUGE-2: {avg_rouge2:.4f} ± {std_rouge2:.4f}")
        print(f"   ROUGE-L: {avg_rougeL:.4f} ± {std_rougeL:.4f}")
        print("━" * 80)

        print("\n📋 Individual Lease Scores:")
        print("━" * 80)
        print(f"{'Lease':<20} {'ROUGE-1':<12} {'ROUGE-2':<12} {'ROUGE-L':<12}")
        print("─" * 80)
        for score in all_scores:
            print(f"{score['lease']:<20} {score['rouge1']:<12.4f} {score['rouge2']:<12.4f} {score['rougeL']:<12.4f}")
        print("━" * 80)

        print(f"\n{'='*80}\n")

    print("📊 Next Steps:")
    print("   • Review the generated summaries above")
    print("   • Compare them to reference summaries")
    print("   • Note what information is captured vs missed")
    print("   • Consider trying guided or chunking approaches next")
    print("   • These ROUGE scores will serve as baseline for comparison\n")


if __name__ == "__main__":
    main()

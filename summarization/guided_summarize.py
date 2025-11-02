"""Guided Lease Summarization with Structured Field Extraction.

Structured prompt-based approach with explicit field extraction using XML templates.
Based on Claude Cookbook: https://github.com/anthropics/claude-cookbooks/tree/main/capabilities/summarization
"""

import os
import time
import math
from typing import Dict, Tuple
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


def extract_thinking(response_message) -> Tuple[str | None, str]:
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


def create_guided_prompt(lease_content: str) -> str:
    """Create a structured guided summarization prompt with XML field templates.

    Args:
        lease_content: Full text of the lease agreement

    Returns:
        Formatted prompt string with XML extraction template
    """
    prompt = f"""Please analyze this commercial sublease agreement and extract the following specific information.
Organize your response using the XML tags provided below:

<parties involved>
Identify the landlord/sublessor and tenant/sublessee. Include their full legal names and entity types (Corporation, LLC, etc.).
</parties involved>

<property details>
Describe the property being leased. Include the complete address, size (square footage), and type of space (office, lab, commercial, etc.).
</property details>

<term and rent>
Extract all lease term information including start date, end date, duration, monthly rent amounts (including any escalation schedules), security deposit, additional rent/operating expenses, and any renewal options.
</term and rent>

<responsibilities>
List the key obligations and responsibilities for both parties. Include who pays utilities, who handles maintenance and repairs, insurance requirements, and any hazardous materials provisions.
</responsibilities>

<consent and notices>
Document any landlord consent requirements, notice delivery methods, and the official notice addresses for both parties.
</consent and notices>

<special provisions>
Identify any special clauses, restrictions, or provisions such as furniture/services included, parking, assignment/subletting restrictions, damage/destruction clauses, default remedies, or holdover terms.
</special provisions>

Document to analyze:
{lease_content}

Please provide a comprehensive extraction using the XML tags above. Be specific with dates, amounts, and names."""

    return prompt


def parse_guided_response(response_text: str) -> Dict[str, str]:
    """Parse XML fields from the model's guided response.

    Args:
        response_text: Full response text containing XML tags

    Returns:
        Dictionary mapping field names to extracted content
    """
    fields = {}

    xml_tags = [
        'parties involved',
        'property details',
        'term and rent',
        'responsibilities',
        'consent and notices',
        'special provisions'
    ]

    for tag in xml_tags:
        start_tag = f'<{tag}>'
        end_tag = f'</{tag}>'

        if start_tag in response_text and end_tag in response_text:
            start = response_text.find(start_tag) + len(start_tag)
            end = response_text.find(end_tag)
            fields[tag.replace(' ', '_')] = response_text[start:end].strip()
        else:
            fields[tag.replace(' ', '_')] = "[Not extracted]"

    return fields


def format_structured_summary(fields: Dict[str, str]) -> str:
    """Format parsed XML fields into a readable structured summary.

    Args:
        fields: Dictionary of extracted field content

    Returns:
        Formatted summary string with section headers
    """
    summary_parts = []

    section_headers = {
        'parties_involved': 'PARTIES INVOLVED',
        'property_details': 'PROPERTY DETAILS',
        'term_and_rent': 'TERM AND RENT',
        'responsibilities': 'KEY RESPONSIBILITIES',
        'consent_and_notices': 'CONSENT & NOTICES',
        'special_provisions': 'SPECIAL PROVISIONS'
    }

    for field_key, header in section_headers.items():
        content = fields.get(field_key, "[Not extracted]")
        summary_parts.append(f"{header}:")
        summary_parts.append(content)
        summary_parts.append("")  # Blank line between sections

    return "\n".join(summary_parts)


def extract_lease_fields(client: OpenAI, lease_content: str) -> Tuple[Dict[str, str], str, str | None]:
    """Extract structured fields from a lease document using guided prompting.

    Args:
        client: OpenAI client instance
        lease_content: Full text of the lease

    Returns:
        Tuple of (fields_dict, formatted_summary, thinking_process)
    """
    prompt = create_guided_prompt(lease_content)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # Deterministic output
        )

        response_message = response.choices[0].message

        # Extract thinking and response
        thinking, full_response = extract_thinking(response_message)

        # Parse XML fields from response
        fields = parse_guided_response(full_response)

        # Format into readable summary
        formatted_summary = format_structured_summary(fields)

        return fields, formatted_summary, thinking

    except Exception as error:
        print(f"\n❌ Error extracting fields: {error}")
        return {}, "[Error: Could not extract fields]", None


def display_guided_comparison(
    formatted_summary: str,
    reference: str,
    thinking: str | None,
    rouge_scores: Dict[str, float],
    stats: Dict,
    lease_num: int,
    total_leases: int
):
    """Display guided summary alongside reference summary.

    Args:
        formatted_summary: Formatted structured summary from model
        reference: Reference summary from file
        thinking: Model's thinking process (if available)
        rouge_scores: Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        stats: Dictionary with processing stats (time, size, etc.)
        lease_num: Current lease number
        total_leases: Total number of leases
    """
    print(f"\n{'='*80}")
    print(f"GUIDED SUMMARIZATION - Lease {lease_num}/{total_leases}")
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

    # Generated structured summary
    print("✅ GENERATED STRUCTURED SUMMARY:")
    print("━" * 80)
    print(formatted_summary)
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
    print("   • Are all 6 field sections extracted?")
    print("   • Is specific data (dates, amounts) accurate?")
    print("   • Does structure match reference format?")
    print("   • Are any critical fields missing?")
    print("   • Is the extraction complete and comprehensive?")


def main():
    """Run guided summarization on all lease documents."""
    print("=" * 80)
    print("LEASE SUMMARIZATION - GUIDED STRUCTURED EXTRACTION")
    print("Model: MiniMax M2 Free")
    print("=" * 80)
    print("\nThis will process 9 lease agreements using structured field extraction.\\n")

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

            # Extract structured fields
            print(f"\n{'─'*80}")
            print(f"Processing {lease_file}... ⏳")
            print(f"{'─'*80}")

            start_time = time.time()
            fields, formatted_summary, thinking = extract_lease_fields(client, lease_content)
            elapsed = time.time() - start_time

            # Load reference summary
            reference_summary = load_reference(reference_file)

            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(formatted_summary, reference_summary)

            # Track scores
            all_scores.append({
                'lease': lease_file,
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL']
            })

            # Display comparison
            display_guided_comparison(
                formatted_summary=formatted_summary,
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

        # Comparison to simple baseline
        print("\n📈 IMPROVEMENT OVER SIMPLE BASELINE:")
        print("━" * 80)
        # Baseline from simple_summarize.py
        baseline_r1, baseline_r2, baseline_rL = 0.54, 0.26, 0.28
        improvement_r1 = ((avg_rouge1 - baseline_r1) / baseline_r1) * 100
        improvement_r2 = ((avg_rouge2 - baseline_r2) / baseline_r2) * 100
        improvement_rL = ((avg_rougeL - baseline_rL) / baseline_rL) * 100

        print(f"   ROUGE-1: {baseline_r1:.4f} → {avg_rouge1:.4f} ({improvement_r1:+.1f}%)")
        print(f"   ROUGE-2: {baseline_r2:.4f} → {avg_rouge2:.4f} ({improvement_r2:+.1f}%)")
        print(f"   ROUGE-L: {baseline_rL:.4f} → {avg_rougeL:.4f} ({improvement_rL:+.1f}%)")
        print("━" * 80)

        print(f"\n{'='*80}\n")

    print("📊 Next Steps:")
    print("   • Review the structured field extraction quality")
    print("   • Compare ROUGE improvements over simple baseline")
    print("   • Verify all 6 field sections are consistently extracted")
    print("   • Consider trying chunking approach for very long documents")
    print("   • These guided scores establish the mid-tier baseline\n")


if __name__ == "__main__":
    main()

"""Insurance Support Ticket Classification using MiniMax M2.

Simple prompt-based classification approach.
Based on Claude Cookbook: https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/classification/guide.ipynb
"""

import os
import pandas as pd
import time
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "minimax/minimax-m2:free"


# Define the 10 insurance support ticket categories
CATEGORIES = """<category>
    <label>Billing Inquiries</label>
    <content>Questions about invoices, charges, fees, and premiums. Requests for clarification on billing statements. Inquiries about payment methods and due dates.</content>
</category>
<category>
    <label>Policy Administration</label>
    <content>Requests for policy changes, updates, or cancellations. Questions about policy renewals and reinstatements. Inquiries about adding or removing coverage options.</content>
</category>
<category>
    <label>Claims Assistance</label>
    <content>Questions about the claims process and filing procedures. Requests for help with submitting claim documentation. Inquiries about claim status and payout timelines.</content>
</category>
<category>
    <label>Coverage Explanations</label>
    <content>Questions about what is covered under specific policy types. Requests for clarification on coverage limits and exclusions. Inquiries about deductibles and out-of-pocket expenses.</content>
</category>
<category>
    <label>Quotes and Proposals</label>
    <content>Requests for new policy quotes and price comparisons. Questions about available discounts and bundling options. Inquiries about switching from another insurer.</content>
</category>
<category>
    <label>Account Management</label>
    <content>Requests for login credentials or password resets. Questions about online account features and functionality. Inquiries about updating contact or personal information.</content>
</category>
<category>
    <label>Billing Disputes</label>
    <content>Complaints about unexpected or incorrect charges. Requests for refunds or premium adjustments. Inquiries about late fees or collection notices.</content>
</category>
<category>
    <label>Claims Disputes</label>
    <content>Complaints about denied or underpaid claims. Requests for reconsideration of claim decisions. Inquiries about appealing a claim outcome.</content>
</category>
<category>
    <label>Policy Comparisons</label>
    <content>Questions about the differences between policy options. Requests for help deciding between coverage levels. Inquiries about how policies compare to competitors' offerings.</content>
</category>
<category>
    <label>General Inquiries</label>
    <content>Questions about company contact information or hours of operation. Requests for general information about products or services. Inquiries that don't fit neatly into other categories.</content>
</category>"""


def get_client() -> OpenAI:
    """Get configured OpenAI client for OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    return OpenAI(
        base_url=BASE_URL,
        api_key=OPENROUTER_API_KEY
    )


def create_classification_prompt(ticket_text: str) -> str:
    """Create a simple classification prompt for a support ticket.

    Args:
        ticket_text: The customer support ticket text to classify

    Returns:
        Formatted prompt string
    """
    prompt = f"""You will classify a customer support ticket into one of the following categories:

<categories>
{CATEGORIES}
</categories>

Here is the customer support ticket:
<ticket>
{ticket_text}
</ticket>

Respond with just the label of the category. Do not include any explanation, just the category label exactly as it appears above."""

    return prompt


def classify_ticket(client, ticket_text: str) -> str:
    """Classify a single support ticket.

    Args:
        client: OpenAI client instance
        ticket_text: The ticket text to classify

    Returns:
        Predicted category label
    """
    prompt = create_classification_prompt(ticket_text)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # Use deterministic output
        )

        prediction = response.choices[0].message.content.strip()
        return prediction

    except Exception as error:
        print(f"Error classifying ticket: {error}")
        return "Error"


def load_test_data(file_path: str) -> pd.DataFrame:
    """Load test data from TSV file.

    Args:
        file_path: Path to the TSV file

    Returns:
        DataFrame with text and label columns
    """
    df = pd.read_csv(file_path, sep='\t')
    return df


def calculate_accuracy(predictions: List[str], actual_labels: List[str]) -> Dict:
    """Calculate classification accuracy metrics.

    Args:
        predictions: List of predicted labels
        actual_labels: List of actual labels

    Returns:
        Dictionary with accuracy metrics
    """
    total = len(predictions)
    correct = sum(1 for pred, actual in zip(predictions, actual_labels) if pred == actual)
    accuracy = correct / total if total > 0 else 0

    return {
        'total': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': accuracy
    }


def main():
    """Run simple prompt-based classification on test data."""
    print("=" * 80)
    print("Insurance Support Ticket Classification - Simple Prompt Approach")
    print("Model: MiniMax M2 Free")
    print("=" * 80)
    print()

    # Get configured client
    client = get_client()

    # Load test data (local data folder)
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'test.tsv')
    print(f"Loading test data from: {data_path}")
    df = load_test_data(data_path)
    print(f"Loaded {len(df)} test examples\n")

    # Classify each ticket
    predictions = []
    actual_labels = df['label'].tolist()

    print("Classifying tickets...")
    print("-" * 80)

    start_time = time.time()

    for idx, row in df.iterrows():
        ticket_text = row['text']
        actual_label = row['label']

        # Classify the ticket
        predicted_label = classify_ticket(client, ticket_text)
        predictions.append(predicted_label)

        # Show progress
        is_correct = predicted_label == actual_label
        status = "✓" if is_correct else "✗"

        print(f"{status} [{idx + 1}/{len(df)}] Predicted: {predicted_label:<25} Actual: {actual_label}")

        # Add small delay to avoid rate limiting
        time.sleep(0.5)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 80)
    print()

    # Calculate accuracy
    metrics = calculate_accuracy(predictions, actual_labels)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total examples:     {metrics['total']}")
    print(f"Correct:            {metrics['correct']}")
    print(f"Incorrect:          {metrics['incorrect']}")
    print(f"Accuracy:           {metrics['accuracy']:.2%}")
    print(f"Time elapsed:       {elapsed_time:.2f} seconds")
    print(f"Avg time/example:   {elapsed_time/len(df):.2f} seconds")
    print("=" * 80)

    # Show some examples of errors
    print("\nExample Misclassifications:")
    print("-" * 80)
    error_count = 0
    for idx, (pred, actual, text) in enumerate(zip(predictions, actual_labels, df['text'])):
        if pred != actual and error_count < 5:
            print(f"\nTicket: {text[:100]}...")
            print(f"Predicted: {pred}")
            print(f"Actual:    {actual}")
            error_count += 1

    if error_count == 0:
        print("No misclassifications! Perfect score!")

    print("-" * 80)


if __name__ == "__main__":
    main()

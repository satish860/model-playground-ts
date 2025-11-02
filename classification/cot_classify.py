"""Insurance Support Ticket Classification using MiniMax M2 with RAG + Chain-of-Thought.

Combines vector-based RAG with Chain-of-Thought reasoning to achieve highest accuracy.
Leverages MiniMax M2's native thinking capability (<think> tags).
Based on Claude Cookbook: https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/classification/guide.ipynb
"""

import os
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "minimax/minimax-m2:free"

# Embedding model (lightweight and fast)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, ~80MB


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


class VectorStore:
    """In-memory vector store for similarity search."""

    def __init__(self, embedding_model: SentenceTransformer):
        """Initialize the vector store.

        Args:
            embedding_model: Sentence transformer model for embeddings
        """
        self.embedding_model = embedding_model
        self.vectors = None
        self.texts = []
        self.labels = []

    def add_documents(self, texts: List[str], labels: List[str]):
        """Add documents to the vector store.

        Args:
            texts: List of text documents
            labels: List of corresponding labels
        """
        print(f"Generating embeddings for {len(texts)} training examples...")
        self.texts = texts
        self.labels = labels

        # Generate embeddings
        self.vectors = self.embedding_model.encode(texts, show_progress_bar=True)
        print(f"Generated {len(self.vectors)} embeddings of dimension {self.vectors.shape[1]}")

    def search(self, query_text: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search for similar documents.

        Args:
            query_text: Query text to search for
            top_k: Number of top results to return

        Returns:
            List of tuples: (text, label, similarity_score)
        """
        if self.vectors is None:
            raise ValueError("Vector store is empty. Call add_documents first.")

        # Generate query embedding
        query_vector = self.embedding_model.encode([query_text])[0]

        # Calculate cosine similarities
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            # Cosine similarity = 1 - cosine distance
            similarity = 1 - cosine(query_vector, doc_vector)
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top-k results
        results = []
        for idx, similarity in similarities[:top_k]:
            results.append((self.texts[idx], self.labels[idx], similarity))

        return results


def get_client() -> OpenAI:
    """Get configured OpenAI client for OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    return OpenAI(
        base_url=BASE_URL,
        api_key=OPENROUTER_API_KEY
    )


def create_cot_prompt(ticket_text: str, similar_examples: List[Tuple[str, str, float]]) -> str:
    """Create a Chain-of-Thought enhanced classification prompt.

    Args:
        ticket_text: The customer support ticket text to classify
        similar_examples: List of similar examples (text, label, similarity)

    Returns:
        Formatted prompt string
    """
    # Format examples
    examples_text = ""
    for i, (text, label, similarity) in enumerate(similar_examples, 1):
        examples_text += f"""
<example>
    <query>{text}</query>
    <label>{label}</label>
</example>"""

    prompt = f"""You will classify a customer support ticket into one of the following categories:

<categories>
{CATEGORIES}
</categories>

Here is the customer support ticket:
<ticket>
{ticket_text}
</ticket>

Use the following similar examples from our training data to help you classify:
<examples>{examples_text}
</examples>

Before providing your classification, think step-by-step about the problem:
1. What is the main issue or request in the ticket?
2. Which categories seem most relevant based on the ticket content?
3. How do the similar examples help narrow down the category?
4. What key words or phrases indicate the correct category?

Respond using this format:
<response>
    <scratchpad>Your step-by-step analysis and reasoning goes here</scratchpad>
    <category>The category label you chose goes here</category>
</response>

Provide ONLY the response in the format above, nothing else."""

    return prompt


def extract_thinking_and_category(response_message) -> Tuple[Optional[str], Optional[str]]:
    """Extract thinking process and category from model response.

    Args:
        response_message: The response message from the API

    Returns:
        Tuple of (thinking, category)
    """
    thinking = None
    category = None

    # First check content field
    content = response_message.content if response_message.content else ""

    # Also check reasoning field (MiniMax M2 specific)
    reasoning = ""
    if hasattr(response_message, 'reasoning') and response_message.reasoning:
        reasoning = response_message.reasoning

    # Combine both fields
    full_text = content + "\n" + reasoning

    # Extract thinking from <think> tags (MiniMax native thinking)
    if '<think>' in full_text and '</think>' in full_text:
        thinking = full_text.split('<think>')[1].split('</think>')[0].strip()

    # Extract thinking from <scratchpad> tags (our prompted thinking)
    if '<scratchpad>' in full_text and '</scratchpad>' in full_text:
        scratchpad = full_text.split('<scratchpad>')[1].split('</scratchpad>')[0].strip()
        # Combine with <think> if exists, otherwise use scratchpad
        if thinking:
            thinking = f"{thinking}\n\n{scratchpad}"
        else:
            thinking = scratchpad

    # Extract category
    if '<category>' in full_text and '</category>' in full_text:
        category = full_text.split('<category>')[1].split('</category>')[0].strip()

    # Fallback: if no structured response, try to extract from content
    if not category and content:
        category = content.strip()

    return thinking, category


def classify_ticket_with_cot(
    client: OpenAI,
    vector_store: VectorStore,
    ticket_text: str,
    top_k: int = 5,
    verbose: bool = True
) -> Tuple[str, Optional[str]]:
    """Classify a ticket using RAG + Chain-of-Thought approach.

    Args:
        client: OpenAI client instance
        vector_store: Vector store with training examples
        ticket_text: The ticket text to classify
        top_k: Number of similar examples to retrieve
        verbose: Whether to print thinking process

    Returns:
        Tuple of (predicted_category, thinking_process)
    """
    # Retrieve similar examples
    similar_examples = vector_store.search(ticket_text, top_k=top_k)

    # Create CoT prompt
    prompt = create_cot_prompt(ticket_text, similar_examples)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # Use deterministic output
        )

        response_message = response.choices[0].message

        # Extract thinking and category
        thinking, category = extract_thinking_and_category(response_message)

        if not category:
            category = "Error: Could not extract category"

        return category, thinking

    except Exception as error:
        print(f"Error classifying ticket: {error}")
        return "Error", None


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from TSV file.

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
    """Run RAG + CoT classification on test data."""
    print("=" * 80)
    print("Insurance Support Ticket Classification - RAG + Chain-of-Thought")
    print("Model: MiniMax M2 Free")
    print(f"Embeddings: {EMBEDDING_MODEL}")
    print("=" * 80)
    print()

    # Get configured client
    client = get_client()

    # Load embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Model loaded: {EMBEDDING_MODEL}\n")

    # Load training data
    train_path = os.path.join(os.path.dirname(__file__), 'data', 'train.tsv')
    print(f"Loading training data from: {train_path}")
    train_df = load_data(train_path)
    print(f"Loaded {len(train_df)} training examples\n")

    # Build vector store
    vector_store = VectorStore(embedding_model)
    vector_store.add_documents(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist()
    )
    print()

    # Load test data
    test_path = os.path.join(os.path.dirname(__file__), 'data', 'test.tsv')
    print(f"Loading test data from: {test_path}")
    test_df = load_data(test_path)
    print(f"Loaded {len(test_df)} test examples\n")

    # Classify each ticket
    predictions = []
    thinking_processes = []
    actual_labels = test_df['label'].tolist()

    print("Classifying tickets with RAG + Chain-of-Thought...")
    print("=" * 80)

    start_time = time.time()

    for idx, row in test_df.iterrows():
        ticket_text = row['text']
        actual_label = row['label']

        print(f"\n{'─' * 80}")
        print(f"[{idx + 1}/{len(test_df)}] Ticket: {ticket_text[:80]}...")
        print(f"{'─' * 80}")

        # Classify with RAG + CoT
        predicted_label, thinking = classify_ticket_with_cot(
            client, vector_store, ticket_text, top_k=5
        )
        predictions.append(predicted_label)
        thinking_processes.append(thinking)

        # Show thinking process
        if thinking:
            print("\n🧠 MODEL'S THINKING:")
            print(f"{'┄' * 80}")
            print(thinking)
            print(f"{'┄' * 80}")

        # Show classification result
        is_correct = predicted_label == actual_label
        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        color = "" if is_correct else ""

        print(f"\n📊 CLASSIFICATION:")
        print(f"  Predicted: {predicted_label}")
        print(f"  Actual:    {actual_label}")
        print(f"  Status:    {status}")

        # Add small delay to avoid rate limiting
        time.sleep(0.5)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'=' * 80}")
    print()

    # Calculate accuracy
    metrics = calculate_accuracy(predictions, actual_labels)

    print("=" * 80)
    print("FINAL RESULTS - RAG + CHAIN-OF-THOUGHT")
    print("=" * 80)
    print(f"Total examples:     {metrics['total']}")
    print(f"Correct:            {metrics['correct']}")
    print(f"Incorrect:          {metrics['incorrect']}")
    print(f"Accuracy:           {metrics['accuracy']:.2%}")
    print(f"Time elapsed:       {elapsed_time:.2f} seconds")
    print(f"Avg time/example:   {elapsed_time/len(test_df):.2f} seconds")
    print("=" * 80)

    # Show detailed misclassifications with thinking
    print("\n" + "=" * 80)
    print("DETAILED MISCLASSIFICATIONS")
    print("=" * 80)
    error_count = 0
    for idx, (pred, actual, text, thinking) in enumerate(
        zip(predictions, actual_labels, test_df['text'], thinking_processes)
    ):
        if pred != actual and error_count < 5:
            print(f"\n{'─' * 80}")
            print(f"Misclassification #{error_count + 1}")
            print(f"{'─' * 80}")
            print(f"Ticket: {text}")
            print(f"\nPredicted: {pred}")
            print(f"Actual:    {actual}")
            if thinking:
                print(f"\n🧠 Model's Reasoning:")
                print(f"{'┄' * 80}")
                print(thinking)
                print(f"{'┄' * 80}")
            error_count += 1

    if error_count == 0:
        print("\n🎉 No misclassifications! Perfect score!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

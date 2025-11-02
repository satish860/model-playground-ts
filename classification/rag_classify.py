"""Insurance Support Ticket Classification using MiniMax M2 with RAG.

Vector-based Retrieval-Augmented Generation approach.
Uses sentence-transformers for embeddings and in-memory vector storage.
Based on Claude Cookbook: https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/classification/guide.ipynb
"""

import os
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple
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


def create_rag_prompt(ticket_text: str, similar_examples: List[Tuple[str, str, float]]) -> str:
    """Create a RAG-enhanced classification prompt.

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

Based on the categories and similar examples above, respond with just the label of the category. Do not include any explanation, just the category label exactly as it appears in the categories list."""

    return prompt


def classify_ticket_with_rag(
    client: OpenAI,
    vector_store: VectorStore,
    ticket_text: str,
    top_k: int = 5
) -> str:
    """Classify a ticket using RAG approach.

    Args:
        client: OpenAI client instance
        vector_store: Vector store with training examples
        ticket_text: The ticket text to classify
        top_k: Number of similar examples to retrieve

    Returns:
        Predicted category label
    """
    # Retrieve similar examples
    similar_examples = vector_store.search(ticket_text, top_k=top_k)

    # Create RAG prompt
    prompt = create_rag_prompt(ticket_text, similar_examples)

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
    """Run RAG-based classification on test data."""
    print("=" * 80)
    print("Insurance Support Ticket Classification - RAG Approach")
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
    actual_labels = test_df['label'].tolist()

    print("Classifying tickets with RAG (Top-5 similar examples)...")
    print("-" * 80)

    start_time = time.time()

    for idx, row in test_df.iterrows():
        ticket_text = row['text']
        actual_label = row['label']

        # Classify with RAG
        predicted_label = classify_ticket_with_rag(client, vector_store, ticket_text, top_k=5)
        predictions.append(predicted_label)

        # Show progress
        is_correct = predicted_label == actual_label
        status = "✓" if is_correct else "✗"

        print(f"{status} [{idx + 1}/{len(test_df)}] Predicted: {predicted_label:<25} Actual: {actual_label}")

        # Add small delay to avoid rate limiting
        time.sleep(0.5)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 80)
    print()

    # Calculate accuracy
    metrics = calculate_accuracy(predictions, actual_labels)

    print("=" * 80)
    print("RESULTS - RAG APPROACH")
    print("=" * 80)
    print(f"Total examples:     {metrics['total']}")
    print(f"Correct:            {metrics['correct']}")
    print(f"Incorrect:          {metrics['incorrect']}")
    print(f"Accuracy:           {metrics['accuracy']:.2%}")
    print(f"Time elapsed:       {elapsed_time:.2f} seconds")
    print(f"Avg time/example:   {elapsed_time/len(test_df):.2f} seconds")
    print("=" * 80)

    # Show some examples of errors
    print("\nExample Misclassifications:")
    print("-" * 80)
    error_count = 0
    for idx, (pred, actual, text) in enumerate(zip(predictions, actual_labels, test_df['text'])):
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

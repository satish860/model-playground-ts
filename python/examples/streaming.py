"""Streaming text generation example with MiniMax M2.

Equivalent to TypeScript src/stream.ts
"""

import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import Config


def main():
    """Stream text response from MiniMax M2."""
    # Get configured client
    client = Config.get_client()

    print("Testing MiniMax M2 Free Model - Streaming\n")
    print("=" * 60)

    prompt = "Write a short story about AI."

    print(f"Prompt: {prompt}\n")
    print("Response (streaming):\n")

    try:
        # Create streaming completion
        stream = client.chat.completions.create(
            model=Config.MODEL_FREE,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        # Stream and print chunks as they arrive
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n\n" + "=" * 60)
        print("Streaming complete!")
        print("=" * 60)

    except Exception as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()

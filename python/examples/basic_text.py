"""Basic text generation example with MiniMax M2.

Equivalent to TypeScript src/index.ts
"""

import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import Config


def main():
    """Generate basic text response from MiniMax M2."""
    # Get configured client
    client = Config.get_client()

    print("Testing MiniMax M2 Free Model - Basic Text Generation\n")
    print("=" * 60)

    prompt = "What is OpenRouter?"

    print(f"Prompt: {prompt}\n")

    try:
        # Generate text completion
        response = client.chat.completions.create(
            model=Config.MODEL_FREE,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract response
        text = response.choices[0].message.content
        usage = response.usage

        print(f"Response:\n{text}\n")
        print("=" * 60)
        print(f"Tokens Used: {usage.total_tokens} "
              f"(Prompt: {usage.prompt_tokens}, "
              f"Completion: {usage.completion_tokens})")
        print("=" * 60)

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()

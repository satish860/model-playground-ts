"""CLI chatbot with conversation history.

Equivalent to TypeScript src/chat.ts
"""

import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import Config
from typing import List, Dict


class ChatSession:
    """Manage a chat session with conversation history."""

    def __init__(self, model: str = None):
        self.client = Config.get_client()
        self.model = model or Config.MODEL_FREE
        self.messages: List[Dict[str, str]] = []

    def send_message(self, user_message: str) -> str:
        """Send a message and get streaming response."""
        self.messages.append({"role": "user", "content": user_message})

        # Stream the response
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True
        )

        print("\nAssistant: ", end="", flush=True)

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print()  # New line after response

        # Add assistant response to history
        self.messages.append({"role": "assistant", "content": full_response})

        return full_response

    def clear_history(self):
        """Clear conversation history."""
        self.messages = []


def main():
    """Run interactive CLI chat."""
    print("=" * 60)
    print("MiniMax M2 Command Line Chat")
    print("=" * 60)
    print("Type your message and press Enter to chat.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'clear' to clear conversation history.")
    print("=" * 60)
    print()

    chat = ChatSession()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break

            if user_input.lower() == 'clear':
                chat.clear_history()
                print("\nConversation history cleared.\n")
                continue

            if not user_input:
                continue

            chat.send_message(user_input)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()

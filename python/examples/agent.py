"""Agent with tool calling example.

Equivalent to TypeScript src/agent.ts and src/agent-simple.ts
Note: Includes debug logging to see actual responses from the model
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import Config


# Define tool functions
def calculator(operation: str, num1: float, num2: float) -> str:
    """Perform basic arithmetic operations."""
    print(f"  [TOOL EXECUTED] calculator({operation}, {num1}, {num2})")

    if operation == "add":
        result = num1 + num2
    elif operation == "subtract":
        result = num1 - num2
    elif operation == "multiply":
        result = num1 * num2
    elif operation == "divide":
        if num2 == 0:
            return "Error: Cannot divide by zero"
        result = num1 / num2
    else:
        return f"Error: Unknown operation '{operation}'"

    print(f"  [TOOL RESULT] {num1} {operation} {num2} = {result}")
    return f"The result of {num1} {operation} {num2} is {result}"


def get_weather(city: str) -> str:
    """Get simulated weather for a city."""
    import random

    print(f"  [TOOL EXECUTED] get_weather('{city}')")

    weather_conditions = ["sunny", "cloudy", "rainy", "snowy"]
    weather = {
        "temperature": random.randint(10, 30),
        "condition": random.choice(weather_conditions),
        "humidity": random.randint(40, 80)
    }

    result = (f"The weather in {city} is {weather['condition']} "
              f"with a temperature of {weather['temperature']}°C "
              f"and {weather['humidity']}% humidity.")

    print(f"  [TOOL RESULT] {result}")
    return result


def get_current_time() -> str:
    """Get the current date and time."""
    print("  [TOOL EXECUTED] get_current_time()")

    now = datetime.now()
    result = now.strftime("%Y-%m-%d %H:%M:%S")

    print(f"  [TOOL RESULT] {result}")
    return f"Current time is {result}"


# Define tool schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic arithmetic operations (add, subtract, multiply, divide)",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform"
                    },
                    "num1": {
                        "type": "number",
                        "description": "First number"
                    },
                    "num2": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["operation", "num1", "num2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g., 'London' or 'New York'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

# Map function names to actual functions
available_functions = {
    "calculator": calculator,
    "get_weather": get_weather,
    "get_current_time": get_current_time
}


def run_agent(user_message: str, max_iterations: int = 5):
    """Run agent with tool calling capabilities."""
    client = Config.get_client()

    print("=" * 80)
    print("AGENT EXECUTION")
    print("=" * 80)
    print(f"User Prompt: {user_message}")
    print(f"Max Iterations: {max_iterations}")
    print("=" * 80)
    print()

    messages = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*80}")

        # Make API call
        response = client.chat.completions.create(
            model=Config.MODEL_FREE,
            messages=messages,
            tools=tools
        )

        response_message = response.choices[0].message

        # FIRST: Check for thinking/reasoning in this iteration
        if hasattr(response_message, 'reasoning') and response_message.reasoning:
            reasoning_text = response_message.reasoning

            print(f"\n{'─'*80}")
            print("🧠 MODEL'S THINKING (this iteration):")
            print(f"{'─'*80}")

            # Extract and show thinking block
            if '<think>' in reasoning_text and '</think>' in reasoning_text:
                thinking = reasoning_text.split('<think>')[1].split('</think>')[0].strip()
                print(f"\n{thinking}\n")
                print(f"{'─'*80}")
            else:
                print(f"\n{reasoning_text}\n")
                print(f"{'─'*80}")

        # Debug: Print the full response
        print(f"\n📊 Response Metadata:")
        print(f"  finish_reason: {response.choices[0].finish_reason}")
        print(f"  has tool_calls: {response_message.tool_calls is not None}")
        print(f"  has content: {bool(response_message.content)}")
        print(f"  has reasoning: {hasattr(response_message, 'reasoning') and bool(response_message.reasoning)}")

        if response_message.tool_calls:
            print(f"\n🔧 Tool Calls: {len(response_message.tool_calls)}")

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
            # No more tool calls, return final response
            final_text = response_message.content

            # MiniMax M2 special handling: Check reasoning field if content is empty
            if not final_text and hasattr(response_message, 'reasoning') and response_message.reasoning:
                reasoning_text = response_message.reasoning

                print("\n" + "=" * 80)
                print("MODEL'S INTERNAL THINKING PROCESS")
                print("=" * 80)

                # Extract text after </think> tag if present
                if '</think>' in reasoning_text:
                    # Split into thinking and response
                    parts = reasoning_text.split('</think>')
                    thinking = parts[0].replace('<think>', '').strip()
                    final_text = parts[-1].strip()

                    print(f"\n{thinking}\n")
                    print("=" * 80)
                    print("END OF THINKING - MODEL'S FINAL RESPONSE BELOW")
                else:
                    final_text = reasoning_text
                    print(f"\n{reasoning_text}\n")

                print("=" * 80)

            print("\n" + "=" * 80)
            print("FINAL RESPONSE")
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


def main():
    """Run agent examples."""
    print("Testing MiniMax M2 Free Model - Agent with Tool Calling\n")
    print("This will show the model's internal thinking process!\n")

    try:
        # Example 1: Multi-step calculation
        run_agent("Calculate (15 + 25) * 2 and tell me if the result is greater than 75")

        print("\n\n")

        # Example 2: Weather with analysis
        run_agent("What's the weather in Paris? Based on the conditions, should I bring an umbrella?")

        print("\n\n")

        # Example 3: Simple query
        run_agent("What is 100 divided by 4?")

    except Exception as error:
        print(f"Error: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

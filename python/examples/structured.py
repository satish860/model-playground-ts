"""Structured output (JSON mode) example with MiniMax M2.

Equivalent to TypeScript src/structured.ts
Note: Free tier uses JSON mode instead of formal structured outputs
"""

import sys
import os
import json

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import Config
from pydantic import BaseModel
from typing import List, Literal


# Define Pydantic models
class Ingredient(BaseModel):
    name: str
    amount: str


class Recipe(BaseModel):
    name: str
    ingredients: List[Ingredient]
    steps: List[str]


class RecipeResponse(BaseModel):
    recipe: Recipe


def example1_basic_structured():
    """Example 1: Basic structured data - Recipe."""
    print("\n=== Example 1: Basic Structured Data (Recipe) ===\n")

    client = Config.get_client()

    # Define the expected schema
    schema = RecipeResponse.model_json_schema()

    response = client.chat.completions.create(
        model=Config.MODEL_FREE,
        messages=[
            {
                "role": "system",
                "content": f"Output valid JSON matching this schema: {json.dumps(schema)}"
            },
            {
                "role": "user",
                "content": "Generate a recipe for chocolate chip cookies. Output as JSON."
            }
        ],
        response_format={"type": "json_object"}
    )

    # Parse and validate with Pydantic
    result_dict = json.loads(response.choices[0].message.content)
    result = RecipeResponse(**result_dict)

    print("Recipe:", json.dumps(result.model_dump(), indent=2))


class Task(BaseModel):
    id: int
    title: str
    priority: Literal["low", "medium", "high"]
    estimatedHours: float
    tags: List[str]


class TaskList(BaseModel):
    tasks: List[Task]


def example2_array_data():
    """Example 2: Array of structured items."""
    print("\n=== Example 2: Array of Tasks ===\n")

    client = Config.get_client()

    schema = TaskList.model_json_schema()

    response = client.chat.completions.create(
        model=Config.MODEL_FREE,
        messages=[
            {
                "role": "system",
                "content": f"Output valid JSON matching this schema: {json.dumps(schema)}"
            },
            {
                "role": "user",
                "content": "Generate 5 software development tasks for building a todo app. Output as JSON."
            }
        ],
        response_format={"type": "json_object"}
    )

    result_dict = json.loads(response.choices[0].message.content)
    result = TaskList(**result_dict)

    print("Tasks:", json.dumps(result.model_dump(), indent=2))


def main():
    """Run all structured output examples."""
    print("Testing MiniMax M2 Free Model - Structured Output (JSON Mode)\n")
    print("=" * 60)

    try:
        example1_basic_structured()
        example2_array_data()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()

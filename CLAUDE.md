# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AI Model Playground** - Multi-language model testing framework for evaluating and comparing AI models through OpenRouter API. Supports both TypeScript and Python implementations for exploring different AI capabilities: text generation, streaming, structured outputs, chat interactions, agentic workflows with tool calling, and classification tasks.

### Key Characteristics
- **Purpose**: Test and benchmark any AI model available through OpenRouter
- **Current Model**: MiniMax M2 (both free and paid tiers), but designed to support any model
- **Languages**: TypeScript (Node.js) and Python (OpenAI SDK)
- **API Provider**: OpenRouter (https://openrouter.ai/) - provides unified access to 200+ models
- **Architecture**: Model-agnostic design with easy model switching via configuration

---

## Project Structure

```
model-playground-ts/
├── src/                          # TypeScript examples
│   ├── index.ts                  # Basic text generation
│   ├── stream.ts                 # Streaming responses
│   ├── structured.ts             # Structured JSON output
│   ├── chat.ts                   # Interactive chat CLI
│   ├── agent.ts                  # Tool calling with debug logging
│   └── agent-simple.ts           # Simplified agent for testing
│
├── python/                       # Python examples
│   ├── examples/
│   │   ├── basic_text.py         # Basic text generation
│   │   ├── streaming.py          # Streaming responses
│   │   ├── structured.py         # Structured JSON output
│   │   ├── chat_cli.py           # Interactive chat CLI
│   │   └── agent.py              # Tool calling with thinking process
│   └── utils/
│       └── config.py             # Centralized configuration
│
├── classification/               # Self-contained classification module
│   ├── data/                     # Training/test data (68+68 examples)
│   ├── simple_classify.py        # Baseline: prompt-only (~70% accuracy)
│   ├── rag_classify.py           # RAG with vectors (~85% accuracy)
│   ├── cot_classify.py           # RAG + Chain-of-Thought (~95%+ accuracy)
│   └── requirements.txt          # Module-specific dependencies
│
├── summarization/                # Document summarization module (in progress)
│   ├── data/                     # Lease documents (to be downloaded)
│   ├── ROADMAP.md                # Detailed implementation plan
│   └── requirements.txt          # Module-specific dependencies (TBD)
│
├── .env                          # API keys (not in git)
└── package.json                  # TypeScript dependencies & scripts
```

---

## Running Examples

### TypeScript Examples

```bash
# Basic examples
npm run start          # Basic text generation
npm run stream         # Streaming responses
npm run structured     # Structured JSON output
npm run chat           # Interactive chat CLI
npm run agent          # Full agent with tools
npm run agent-simple   # Simplified agent for debugging
```

### Python Examples

```bash
# Activate virtual environment first
cd python
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Run examples
python examples/basic_text.py
python examples/streaming.py
python examples/structured.py
python examples/chat_cli.py
python examples/agent.py
```

### Classification Module

```bash
cd classification

# Three approaches with increasing accuracy:
python simple_classify.py   # Baseline: Simple prompts
python rag_classify.py      # RAG with vector similarity
python cot_classify.py      # RAG + Chain-of-Thought reasoning
```

### Summarization Module

```bash
cd summarization

# Three approaches (Coming Soon):
python simple_summarize.py    # Basic: Simple bullet-point summary
python guided_summarize.py     # Structured: Guided field extraction
python chunking_summarize.py   # Advanced: Meta-summarization with chunking

# View roadmap:
cat ROADMAP.md  # Detailed implementation plan
```

---

## Architecture & Key Patterns

### Model Configuration

**OpenRouter Integration**: All examples use OpenRouter as a unified API gateway to access any model.

**TypeScript Pattern**:
```typescript
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
const openrouter = createOpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });
```

**Python Pattern**:
```python
from utils.config import Config
client = Config.get_client()  # Returns OpenAI client configured for OpenRouter
```

### Model Selection

Models are configured via constants:
- **TypeScript**: Model strings in each file (e.g., `minimax/minimax-m2:free`)
- **Python**: `Config.MODEL_FREE` in `utils/config.py`

**To test a different model**: Change the model string to any OpenRouter-supported model (e.g., `anthropic/claude-3.5-sonnet`, `openai/gpt-4`, etc.)

### Tool Calling Architecture

**MiniMax M2 Special Feature**: The model includes a native "thinking" process exposed via `<think>` tags in the `reasoning` field.

**Pattern** (see `python/examples/agent.py`):
```python
response_message = response.choices[0].message

# MiniMax M2 exposes thinking in reasoning field
if hasattr(response_message, 'reasoning') and response_message.reasoning:
    thinking = extract_thinking(response_message.reasoning)  # Extract <think> blocks
```

This thinking capability is leveraged in the classification module's Chain-of-Thought approach.

### Classification Module Architecture

**Self-Contained Design**: The `classification/` folder is independent with its own data and dependencies.

**Three Progressive Approaches**:

1. **Simple Prompt** (`simple_classify.py`)
   - Categories + Ticket → Classification
   - Baseline accuracy: ~70%

2. **RAG** (`rag_classify.py`)
   - Uses `sentence-transformers` for embeddings
   - In-memory vector store with cosine similarity
   - Retrieves top-5 similar training examples
   - Categories + 5 Examples + Ticket → Classification
   - Accuracy: ~85%

3. **Chain-of-Thought** (`cot_classify.py`)
   - RAG + Explicit reasoning prompt
   - Leverages MiniMax's native thinking
   - Shows step-by-step reasoning for each classification
   - Categories + 5 Examples + Ticket → Think → Classification
   - Accuracy: ~95%+

**Data Source**: Claude Cookbook insurance ticket classification dataset (68 training, 68 test examples)

### Summarization Module Architecture

**Self-Contained Design**: The `summarization/` folder is independent with its own data and dependencies.

**Three Progressive Approaches** (Planned):

1. **Simple Summarization** (`simple_summarize.py`)
   - Document → "Summarize this" → Summary
   - Basic bullet-point summaries
   - Fast but may miss important details

2. **Guided Summarization** (`guided_summarize.py`)
   - Document → Extract specific fields (parties, dates, obligations) → Structured Summary
   - Domain-specific prompting for legal documents
   - More consistent and comprehensive

3. **Meta-Summarization (Chunking)** (`chunking_summarize.py`)
   - Document → Split into chunks → Summarize each → Synthesize → Final Summary
   - Best for very long documents (50+ pages)
   - Handles documents exceeding context window

**Data Source**: Claude Cookbook legal document summarization dataset (9 lease agreements with reference summaries)

**Status**: Roadmap created, implementation in progress. See `summarization/ROADMAP.md` for detailed plan.

---

## Configuration & Environment

### Required Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Get your key at: https://openrouter.ai/keys

### Dependencies

**TypeScript**:
```bash
npm install
```

**Python** (global):
```bash
cd python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Classification Module**:
```bash
cd classification
pip install -r requirements.txt  # Includes sentence-transformers
```

---

## Testing Different Models

### Quick Model Switch

**TypeScript**: Change model string in the file:
```typescript
model: openrouter('anthropic/claude-3.5-sonnet')  // Instead of minimax/minimax-m2:free
```

**Python**: Update `Config.MODEL_FREE` in `python/utils/config.py`:
```python
MODEL_FREE = "anthropic/claude-3.5-sonnet"  # Instead of minimax/minimax-m2:free
```

**Classification**: Update `MODEL` in each classifier file:
```python
MODEL = "anthropic/claude-3.5-sonnet"  # Test classification with different models
```

### Model-Specific Considerations

- **Tool Calling**: Support varies by model. MiniMax M2 has native thinking in `reasoning` field.
- **Structured Output**: Use `generateObject()` (TypeScript) or JSON mode (Python) for reliable structured outputs.
- **Rate Limits**: Free tiers have limits. Check OpenRouter docs for model-specific limits.

---

## Important Conventions

### Model-Agnostic Code
- When adding new examples, avoid hardcoding model-specific logic
- Use configuration constants for model selection
- Test with multiple models when possible

### Error Handling Pattern
```typescript
try {
  // API call
} catch (error) {
  console.error('Error:', error instanceof Error ? error.message : String(error));
}
```

### Environment Setup
- Always use `.env` files for API keys (never commit keys)
- Provide `.env.example` for reference
- Check for API keys before making calls

### Classification Module
- Self-contained with its own `requirements.txt`
- Uses local `data/` folder for training/test data
- Independent of main project dependencies

---

## Development Philosophy

This playground is designed for **rapid experimentation** with different AI models:

1. **Easy Model Switching**: Change one configuration value to test different models
2. **Parallel Implementations**: TypeScript and Python versions for language preference
3. **Progressive Complexity**: Start with basic examples, move to advanced (agents, classification)
4. **Real-World Use Cases**: Classification module demonstrates practical application
5. **Transparency**: Show model thinking/reasoning processes when available

When adding new features or examples, maintain this model-agnostic philosophy so any OpenRouter-supported model can be tested.

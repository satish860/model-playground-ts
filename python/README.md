# Python Model Playground - MiniMax M2 with OpenRouter

Python examples for testing MiniMax M2 model through OpenRouter API using the OpenAI Python SDK.

## Project Structure

```
python/
├── examples/
│   ├── basic_text.py      # Basic text generation
│   ├── streaming.py       # Streaming responses
│   ├── structured.py      # Structured output (JSON mode)
│   ├── chat_cli.py        # Interactive CLI chatbot
│   └── agent.py           # Tool calling / agent examples
├── utils/
│   ├── config.py          # Configuration management
│   └── __init__.py
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Setup

### 1. Create Virtual Environment

```bash
cd python
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# Get your key at: https://openrouter.ai/keys
```

Your `.env` file should contain:
```env
OPENROUTER_API_KEY=your_api_key_here
```

## Running Examples

### Basic Text Generation

```bash
python examples/basic_text.py
```

Generates a single text response from MiniMax M2.

### Streaming

```bash
python examples/streaming.py
```

Streams the response in real-time as it's generated.

### Structured Output (JSON Mode)

```bash
python examples/structured.py
```

Generates structured JSON output validated with Pydantic models.

### Interactive Chat CLI

```bash
python examples/chat_cli.py
```

Starts an interactive chat session with conversation history.

Commands:
- Type your message and press Enter
- Type `exit` or `quit` to end
- Type `clear` to clear conversation history

### Agent with Tool Calling

```bash
python examples/agent.py
```

Demonstrates tool calling capabilities (with debug logging).

**Note:** Tool calling may have limitations on the free tier. See Known Issues below.

## Dependencies

- **openai** (>= 2.6.1) - OpenAI Python SDK
- **python-dotenv** (>= 1.0.0) - Environment variable management
- **pydantic** (>= 2.0.0) - Data validation and structured outputs

## Features & Limitations

### ✅ Working Features

- **Basic Text Generation** - Standard chat completions
- **Streaming** - Real-time response streaming
- **Structured Output** - JSON mode with Pydantic validation
- **Chat History** - Multi-turn conversations with context
- **CLI Interface** - Interactive command-line chat

### ⚠️ Known Issues

- **Tool Calling on Free Tier** - The `minimax/minimax-m2:free` model may have limited or inconsistent tool calling support through OpenRouter
- **Character Encoding** - Some emoji characters may cause encoding errors in Windows console
- **Rate Limits** - Free tier has rate limits (50-1000 requests/day depending on account status)

## Comparison with TypeScript Version

This Python implementation provides equivalent functionality to the TypeScript version:

| Feature | TypeScript | Python |
|---------|------------|--------|
| Basic Text | `src/index.ts` | `examples/basic_text.py` |
| Streaming | `src/stream.ts` | `examples/streaming.py` |
| Structured Output | `src/structured.ts` | `examples/structured.py` |
| Chat CLI | `src/chat.ts` | `examples/chat_cli.py` |
| Agent/Tools | `src/agent.ts` | `examples/agent.py` |

## API Reference

### Configuration (`utils/config.py`)

```python
from utils.config import Config

# Get configured OpenAI client
client = Config.get_client()

# Access configuration
Config.MODEL_FREE  # "minimax/minimax-m2:free"
Config.BASE_URL    # "https://openrouter.ai/api/v1"
```

### Basic Usage

```python
from utils.config import Config

client = Config.get_client()

response = client.chat.completions.create(
    model=Config.MODEL_FREE,
    messages=[
        {"role": "user", "content": "Your prompt here"}
    ]
)

print(response.choices[0].message.content)
```

### Streaming Usage

```python
from utils.config import Config

client = Config.get_client()

stream = client.chat.completions.create(
    model=Config.MODEL_FREE,
    messages=[{"role": "user", "content": "Your prompt"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Troubleshooting

### "OPENROUTER_API_KEY not found"

Make sure you've created a `.env` file with your API key:
```bash
cp .env.example .env
# Edit .env and add your key
```

### Rate Limit Errors

If you hit rate limits, consider:
- Purchasing credits on OpenRouter for higher limits
- Implementing rate limit handling with retries
- Using the paid tier (`minimax/minimax-m2`)

### Import Errors

Make sure you've activated the virtual environment and installed dependencies:
```bash
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [MiniMax M2 Model Info](https://openrouter.ai/minimax/minimax-m2)

## License

This project is for educational and testing purposes.
